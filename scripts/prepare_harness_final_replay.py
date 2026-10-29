#!/usr/bin/env python3
"""Freeze tools-disabled final-prompt replay inputs; preparation never uses GPUs.

Each saved pre-final trajectory is evaluated with BOTH final prompts. Original
search tokens, steps and remaining wall time are charged to each branch. Qrels
are read only by the host scorer after generation, never by the replay engine.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

from prepare_harness_pilot_continuation import (
    BENCH_PYTHON, SERVING_PYTHON, file_hash, inventory, json_hash,
    read_json, relative_file, verify_group, write_json,
)

PROFILES = ("budgeted-stable", "budgeted-selective")
CONTRACT_HASHES = {
    "budgeted-stable": "e155daec787f39c79ce6f3a6b18383ca792f452b863ec5ef819ea346ac0d12e9",
    "budgeted-selective": "c6f4a9e46de4f0ebf9b21aca6f73c86a7b1dfd726783362bc3c476807be6419f",
}
ENTRY = 'if (resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url)) main().catch'


def contracts(runtime):
    base = read_json(runtime / "dci_bench/protocol/contract.json")
    profiles = read_json(runtime / "dci_bench/protocol/pilot_profiles.json")
    result = {}
    for name in PROFILES:
        profile, contract = profiles[name], deepcopy(base)
        contract.update(benchmark_contract_version=profile["benchmark_contract_version"], protocol_profile=name)
        contract["generation_config"].update(profile.get("generation_config", {}))
        contract["budget_control"] = deepcopy(profiles[profile["budget_control"]]["policy"])
        contract["budget_control"].update(profile.get("budget_control_overrides", {}))
        contract["search_policy"] = deepcopy(profiles[profile["search_policy"]]["search_policy"])
        if json_hash(contract) != CONTRACT_HASHES[name]:
            raise ValueError(f"frozen contract mismatch: {name}")
        result[name] = contract
    a, b = (deepcopy(result[p]) for p in PROFILES)
    for value in (a, b):
        for key in ("benchmark_contract_version", "protocol_profile"):
            value.pop(key)
        for key in ("final_prompt", "final_repair_prompt", "version"):
            value["budget_control"].pop(key)
    if a != b:
        raise ValueError("profiles differ beyond final prompts and version labels")
    return result


def text_content(message):
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(p["text"] for p in content if p["type"] == "text")


def extract_prefix(trace, contract, filenames):
    events = [event for event in trace["control_events"] if event["action"] == "finalize"]
    if not events:
        return None
    if len(events) != 1 or trace.get("failure") or trace.get("partial"):
        raise ValueError("replay requires one completed, successful forced-final transition")
    event = events[0]
    prompt = contract["budget_control"]["final_prompt"].replace("{reason}", event["reason"]).replace(
        "{remaining_tokens}", str(event["remaining_tokens"]))
    matches = [i for i, m in enumerate(trace["messages"]) if m["role"] == "user" and text_content(m) == prompt]
    if len(matches) != 1:
        raise ValueError("cannot identify a unique first final prompt")
    cut = matches[0]
    prefix = trace["messages"][:cut]
    assistants = [m for m in prefix if m["role"] == "assistant"]
    if len(assistants) != event["turn"] or not assistants:
        raise ValueError("final transition assistant-turn mismatch")
    if any(not m["usage"].get("usageAvailable", True) or m["stopReason"] == "error" for m in assistants):
        raise ValueError("prefix has missing usage or a failed model call")
    used = sum(m["usage"]["input"] + m["usage"]["output"] for m in assistants)
    if used + event["remaining_tokens"] != contract["max_total_model_tokens"]:
        raise ValueError("prefix cumulative usage does not match final transition")
    if any(m["role"] == "toolResult" for m in trace["messages"][cut:]):
        raise ValueError("source executed tools after the final transition")
    calls = [p["id"] for m in assistants for p in m["content"] if p["type"] == "toolCall"]
    results = [m["toolCallId"] for m in prefix if m["role"] == "toolResult"]
    if len(set(calls)) != len(calls) or sorted(calls) != sorted(results):
        raise ValueError("prefix has missing, duplicate or unmatched tool results")
    receipts = [m["details"]["receipt"] for m in prefix
                if m["role"] == "toolResult" and m["toolName"] == "read"
                and not m.get("isError") and m.get("details", {}).get("bodyEvidence")]
    if not receipts or receipts != trace["body_evidence"]["read_receipts"]:
        raise ValueError("prefix read receipts differ from original evidence")
    if any(r["returned_bytes"] <= 0 for r in receipts):
        raise ValueError("empty read receipt")
    if not 0 < event["remaining_seconds"] <= contract["sample_timeout_seconds"]:
        raise ValueError("invalid remaining wall budget")
    return {"schema_version": "dci-final-replay-input-v1", "task": trace["task"],
            "query_id": trace["query_id"], "origin_profile": trace["protocol_profile"],
            "messages": prefix, "prefix_sha256": json_hash(prefix), "read_receipts": receipts,
            "corpus_filenames": filenames, "control": event, "original_final_prompt": prompt}


def saved_wire_request(path, prefix, contract):
    # inspect-ai's installed zip extension supports the saved zstd ZIP format.
    if sys.version_info < (3, 14):
        import zipfile_zstd  # noqa: F401
    with zipfile.ZipFile(path) as archive:
        sample = json.loads(archive.read(f"samples/{prefix['query_id']}_epoch_1.json"))
    if str(sample["id"]) != prefix["query_id"]:
        raise ValueError("Inspect sample ID mismatch")

    def resolve(value):
        if isinstance(value, str) and value.startswith("attachment://"):
            return sample["attachments"][value.removeprefix("attachment://")]
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return value

    models = [e for e in sample["events"] if e["event"] == "model"]
    call = models[prefix["control"]["turn"]]["call"]
    if call["call_key"] != "messages":
        raise ValueError("unknown Inspect request storage")
    calls, wire = sample["events_data"]["calls"], []
    for lo, hi in call["call_refs"]:
        if not 0 <= lo <= hi <= len(calls):
            raise ValueError("Inspect call range is invalid")
        wire.extend(resolve(calls[lo:hi]))
    request = resolve(call["request"])
    if (wire[-1] != {"role": "user", "content": prefix["original_final_prompt"]}
            or wire[0] != {"role": "system", "content": contract["system_prompt"]}
            or request.get("tools") is not None or request.get("tool_choice") is not None):
        raise ValueError("saved forced-final request cannot be matched")
    # Cross-check semantic identity while retaining the original wire strings,
    # especially spaces within serialized tool-call arguments (token-visible).
    if len(wire) != len(prefix["messages"]) + 2:
        raise ValueError("Inspect changed the prefix message count")
    for actual, original in zip(wire[1:-1], prefix["messages"], strict=True):
        role = "tool" if original["role"] == "toolResult" else original["role"]
        expected_text = text_content(original)
        if role == "assistant":
            expected_text = expected_text.strip()
        if actual["role"] != role or (actual.get("content") or "") != expected_text:
            raise ValueError("Inspect/trace message content mismatch")
        if role == "tool" and actual["tool_call_id"] != original["toolCallId"]:
            raise ValueError("Inspect/trace tool result mismatch")
        if role == "assistant":
            a = [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in actual.get("tool_calls", [])]
            b = [(c["id"], c["name"], c["arguments"]) for c in original["content"] if c["type"] == "toolCall"]
            if a != b:
                raise ValueError("Inspect/trace tool arguments differ")
    template = {k: v for k, v in request.items() if k != "extra_headers"}
    extra = template.pop("extra_body", {})
    if set(template) & set(extra):
        raise ValueError("ambiguous saved extra_body")
    template.update(extra)  # OpenAI SDK merges extra_body into the JSON body.
    expected = {"model": "Qwen3.8-27B", "tools": None, "tool_choice": None,
                "max_tokens": 1024, "temperature": 0, "seed": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
    if template != expected:
        raise ValueError("saved serving generation contract changed")
    # JSON.stringify represents integral floats as integers. Canonicalize this
    # non-token-visible parameter before sharing hashes between Python and Pi.
    template["temperature"] = 0
    return {"wire_prefix": wire[:-1], "wire_prefix_sha256": json_hash(wire[:-1]),
            "request_template": template, "original_request_sha256": json_hash({**template, "messages": wire})}


def generate_engine(source, fragment):
    if source.count(ENTRY) != 1:
        raise ValueError("frozen Pi entrypoint changed; adapter needs review")
    return source.replace(ENTRY, ENTRY.replace("main().catch", "replayMain().catch")) + "\n" + fragment


def verify_runtime(root, manifest):
    if inventory(root / "runtime", Path(manifest["shared_workspaces"])) != manifest["runtime_inventory"]:
        raise ValueError("frozen dependency runtime changed")


def prepare(runtime_bundle, pilots, output):
    if {'results', 'eval_data', 'raw_data', 'eval_data_code'} & set(output.resolve().parts):
        raise ValueError('Executable bundles belong under the code repository releases/ directory')
    print("[replay] verifying frozen dependencies and extracting saved requests (CPU only)", flush=True)
    frozen = read_json(runtime_bundle / "continuation.json")
    runtime = runtime_bundle / "runtime"
    verify_runtime(runtime_bundle, frozen)
    resolved = contracts(runtime)
    source = runtime / "pi-dci/scripts/dci-run.ts"
    if file_hash(source) != frozen["source_hashes"]["pi-dci/scripts/dci-run.ts"]:
        raise ValueError("original Pi runner hash changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "host").mkdir()
    engine = output / "pi-dci/scripts/dci-run.ts"
    engine.parent.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    fragment = (here / "harness_final_replay_entry.ts.inc").read_text()
    engine.write_text(generate_engine(source.read_text(), fragment))
    shutil.copytree(runtime / "dci_bench/protocol", output / "dci_bench/protocol")
    shutil.copyfile(runtime / "pi-dci/package.json", output / "pi-dci/package.json")
    (output / "pi-dci/node_modules").symlink_to(runtime / "pi-dci/node_modules", target_is_directory=True)
    shutil.copyfile(Path(__file__), output / "replay.py")
    shutil.copyfile(here / "prepare_harness_pilot_continuation.py", output / "prepare_harness_pilot_continuation.py")
    shutil.copyfile(here / "run_harness_final_replay.sh", output / "serve.sh")
    shutil.copyfile(runtime_bundle / "continuation.json", output / "dependency-runtime.json")
    launch = output / "launch.sh"
    launch.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\n'
                      'bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
                      'mkdir -p "$bundle/logs"\n'
                      'driver_log="$bundle/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"\n'
                      'exec > >(tee "$driver_log") 2>&1\n'
                      'echo "[replay] driver log: $driver_log"\n'
                      f'exec "${{BENCH_PYTHON:-{BENCH_PYTHON}}}" "$bundle/replay.py" --bundle "$bundle" "$@"\n')
    rows, exclusions, groups, references, identities = [], [], [], {}, set()
    expected_queries = read_json(pilots[0] / "evaluation/plan.json")["query_ids"]
    for pilot in pilots:
        plan = read_json(pilot / "evaluation/plan.json")
        if plan["query_suite"] != "validation":
            raise ValueError("only frozen validation pilots are supported")
        references[str(pilot / "evaluation/plan.json")] = file_hash(pilot / "evaluation/plan.json")
        references[str(pilot / "backend.json")] = file_hash(pilot / "backend.json")
        for result_path in sorted((pilot / "evaluation/runs").glob("*/*/runs/*/task-result.json")):
            run, result = result_path.parent, read_json(result_path)
            task, origin = result["task"], result["run_id"].removeprefix("pilot-")
            if origin not in PROFILES or result["query_ids"] != expected_queries[task]:
                raise ValueError("unexpected source profile or query set")
            groups.append(verify_group(run, plan, task, origin))
            if result["code"]["source_tree_fingerprint"] != groups[0]["source_tree_fingerprint"]:
                raise ValueError("source pilots used different implementations")
            for name in ("task-result.json", "run-manifest.json", "_COMPLETE"):
                references[str(run / name)] = file_hash(run / name)
            mapping = read_json(runtime / "data/metadata" / task / "doc_id_files.json")
            filenames = sorted(mapping.values())
            if sorted(p.name for p in (runtime / "data/workspaces" / task / "corpus").iterdir()) != filenames:
                raise ValueError(f"corpus filename membership changed: {task}")
            qrels_path = runtime / "data/metadata" / task / "qrels.json"
            if json_hash(read_json(qrels_path)) != result["data"]["qrels_hash"]:
                raise ValueError("source qrels changed")
            if not (output / "host" / f"{task}.qrels.json").exists():
                shutil.copyfile(qrels_path, output / "host" / f"{task}.qrels.json")
            for entry in result["samples"]:
                sample_path = relative_file(run, entry["result"]["path"])
                trace_path = relative_file(run, entry["trace"]["path"])
                sample, trace = read_json(sample_path), read_json(trace_path)
                identity = task, sample["query_id"], origin
                if identity in identities or not sample["valid_output"]:
                    raise ValueError("duplicate or invalid source sample")
                identities.add(identity)
                if trace["task"] != task or trace["query_id"] != sample["query_id"] or trace["protocol_profile"] != origin:
                    raise ValueError("source trace identity mismatch")
                if trace["base_contract_sha256"] != CONTRACT_HASHES[origin]:
                    raise ValueError("source trace contract hash mismatch")
                for path in (sample_path, trace_path):
                    references[str(path)] = file_hash(path)
                prefix = extract_prefix(trace, resolved[origin], filenames)
                if prefix is None:
                    exclusions.append({"task": task, "query_id": sample["query_id"], "origin_profile": origin,
                                       "reason": "natural final; no final-prompt intervention"})
                    continue
                eval_path = relative_file(run, sample["artifacts"]["inspect_log"]["path"])
                references[str(eval_path)] = file_hash(eval_path)
                prefix.update(saved_wire_request(eval_path, prefix, resolved[origin]))
                key = f"{task}--{sample['query_id']}--{origin}"
                path = output / "inputs" / f"{key}.json"
                write_json(path, prefix)
                rows.append({"key": key, "task": task, "query_id": sample["query_id"], "origin_profile": origin,
                             "input": str(path), "prefix_sha256": prefix["prefix_sha256"], "source_trace": str(trace_path)})
    expected = {(t, q, p) for t, ids in expected_queries.items() for q in ids for p in PROFILES}
    if identities != expected or len(rows) != 118 or len(exclusions) != 2 or any(
            (e["task"], e["query_id"]) != ("LLMFQuADRetrieval", "62") for e in exclusions):
        raise ValueError("expected 120 validation sources, 118 forced prefixes and two FQuAD62 natural finals")
    write_json(output / "input-paths.json", [r["input"] for r in rows])
    backend = read_json(pilots[-1] / "backend.json")
    write_json(output / "expected-backend.json", backend)
    plan = {"schema_version": "dci-final-replay-plan-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "dependency_bundle": str(runtime_bundle), "runtime": str(runtime), "profiles": list(PROFILES),
            "contract_hashes": CONTRACT_HASHES, "source_pilots": [str(p) for p in pilots], "source_groups": groups,
            "source_references": references, "prefixes": rows, "exclusions": exclusions,
            "unique_queries": 59, "prefix_count": len(rows), "planned_final_chains": len(rows) * 2,
            "max_concurrency": 4, "max_model_calls_per_chain": 4,
            "design": "Both source origins crossed with both final prompts; sequential alternating arms within each prefix; query is the analysis unit, not origin or branch. No new search/read/tool execution. Original usage, turns and remaining wall time retained.",
            "transport": "Saved Inspect wire messages and generation parameters sent directly to vLLM; bypasses Inspect scheduling. Only final/repair user prompts change within each prefix. Record every outgoing request. Compare prompt treatments within replay, not latency against the original full run.",
            "files": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file() and not path.is_symlink():
            plan["files"][str(path.relative_to(output))] = file_hash(path)
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(file_hash(output / "plan.json") + "\n")
    for path in output.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    print(f"[replay] frozen {len(rows)} prefixes; {len(rows) * 2} final chains; {len(exclusions)} natural finals excluded")
    print(f"bash {launch}")


def bundle_env(bundle, plan):
    env = os.environ.copy()
    backend = read_json(bundle / "expected-backend.json")
    env.update({"BENCH_PYTHON": env.get("BENCH_PYTHON", BENCH_PYTHON),
                "SERVING_PYTHON": env.get("SERVING_PYTHON", SERVING_PYTHON),
                "PYTHONPATH": plan["runtime"], "PYTHONDONTWRITEBYTECODE": "1", "TSX_DISABLE_CACHE": "1",
                "DCI_EVALUATION_TRACK": "document", "DCI_PROTOCOL_PROFILE": "v3", "MODEL_PATH": backend["model"]["source_path"],
                "SERVED_MODEL_NAME": backend["model"]["served_model_name"],
                "TENSOR_PARALLEL_SIZE": str(backend["backend"]["tensor_parallel_size"]),
                "VLLM_MAX_NUM_SEQS": "4", "MAX_CONCURRENCY": "4", "MAX_MODEL_LEN": "65536", "DTYPE": "bfloat16",
                "VLLM_GPU_MEMORY_UTILIZATION": str(backend["backend"]["gpu_memory_utilization"]),
                "VLLM_EXTRA_ARGS": backend["backend"]["runtime"]["launch_environment"]["VLLM_EXTRA_ARGS"],
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
                "VLLM_HOST": "127.0.0.1", "VLLM_PORT": env.get("VLLM_PORT", "28099"),
                "TOOL_CALL_PARSER": "qwen3_coder", "REASONING_PARSER": "qwen3", "LANGUAGE_MODEL_ONLY": "1"})
    for key in ("NODE_PATH", "NODE_OPTIONS", "DCI_CODE_RUNTIME_CONTRACT", "DCI_PROTOCOL_CONTRACT_SHA256",
                "KV_CACHE_DTYPE", "VLLM_NVFP4_GEMM_BACKEND", "DRY_RUN"):
        env.pop(key, None)
    env["PATH"] = str(Path(env["BENCH_PYTHON"]).parent) + os.pathsep + env.get("PATH", "")
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env["BACKEND_BASE_URL"] = f"http://127.0.0.1:{env['VLLM_PORT']}/v1"
    return env


def verify_bundle(bundle, plan):
    print("[replay] checking frozen bundle, dependency inventory and source artifacts", flush=True)
    if (bundle / "plan.sha256").read_text().strip() != file_hash(bundle / "plan.json"):
        raise ValueError("replay plan changed")
    for relative, expected in plan["files"].items():
        if file_hash(relative_file(bundle, relative)) != expected:
            raise ValueError(f"replay bundle changed: {relative}")
    dependency = read_json(bundle / "dependency-runtime.json")
    verify_runtime(Path(plan["dependency_bundle"]), dependency)
    if (bundle / "pi-dci/node_modules").resolve() != (Path(plan["runtime"]) / "pi-dci/node_modules").resolve():
        raise ValueError("replay Node dependency link changed")
    for path, expected in plan["source_references"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f"source result changed: {path}")
    contracts(Path(plan["runtime"]))


def check_inputs(bundle, plan, env):
    for profile in PROFILES:
        checked = subprocess.check_output(["node", "--import", "tsx", "scripts/dci-run.ts", "--check-inputs",
                                           str(bundle / "input-paths.json")], cwd=bundle / "pi-dci",
                                          env={**env, "DCI_PROTOCOL_PROFILE": profile,
                                               "DCI_PROTOCOL_CONTRACT_SHA256": CONTRACT_HASHES[profile]}, text=True)
        print(checked, end="", flush=True)
    print("[replay] frozen source, 120 source samples, 118 prefixes, contracts and budgets checked; no GPU started", flush=True)


def run_evaluation(bundle, plan, env, output):
    sys.path.insert(0, plan["runtime"])
    from dci_bench.scoring.retrieval import precision_at_k, score_query, zero_metrics

    cancelled = threading.Event()

    def run_pair(index_row):
        index, row = index_row
        results = []
        for profile in (PROFILES if index % 2 == 0 else PROFILES[::-1]):
            if cancelled.is_set():
                return results
            destination = output / "samples" / row["key"] / profile
            destination.mkdir(parents=True, exist_ok=False)
            result_path = destination / "final.json"
            started = time.monotonic()
            timeout = read_json(Path(row["input"]))["control"]["remaining_seconds"] + 60
            with (destination / "driver.log").open("x") as log:
                subprocess.run(["node", "--import", "tsx", "scripts/dci-run.ts", row["input"], str(result_path)],
                               cwd=bundle / "pi-dci", env={**env, "DCI_PROTOCOL_PROFILE": profile,
                               "DCI_PROTOCOL_CONTRACT_SHA256": CONTRACT_HASHES[profile]}, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=timeout)
            final = read_json(result_path)
            if final["prefix_sha256"] != row["prefix_sha256"] or final["profile"] != profile or final["tool_executions"] != 0:
                raise ValueError("replay output identity mismatch")
            # Service failures must stop the job, not become apparent retrieval losses.
            if (final.get("failure") or {}).get("kind") in ("agent_error", "usage_unavailable", "infrastructure_error"):
                raise RuntimeError(f"replay transport/runtime failed: {result_path}: {final['failure']}")
            qrels = read_json(bundle / "host" / f"{row['task']}.qrels.json")[row["query_id"]]
            ranked = final["ranked_doc_ids"]
            metrics = score_query(ranked, qrels) if final["valid_output"] else zero_metrics()
            metrics["precision_at_10"] = precision_at_k(ranked, qrels, 10) if final["valid_output"] else 0.0
            gold = {doc for doc, value in qrels.items() if value > 0}
            sample = {**{k: row[k] for k in ("key", "task", "query_id", "origin_profile", "prefix_sha256")},
                      "profile": profile, "valid_output": final["valid_output"], "failure": final["failure"],
                      "metrics": metrics, "ranked_doc_ids": ranked, "gold_hits": len(set(ranked) & gold),
                      "non_gold_returned": len(set(ranked) - gold), "replay_tokens": final["usage"]["total_tokens"],
                      "prefix_tokens": final["prefix_usage"]["total_tokens"], "repair_attempts": final["repair_attempts"],
                      "elapsed_seconds": time.monotonic() - started, "final": str(result_path), "final_sha256": file_hash(result_path)}
            write_json(destination / "result.json", sample)
            write_json(destination / "_COMPLETE", {"result_sha256": json_hash(sample), "final_sha256": sample["final_sha256"]})
            results.append(sample)
            print(f"[replay] {index + 1}/{len(plan['prefixes'])} {row['key']} -> {profile}: valid={final['valid_output']}", flush=True)
        return results

    rows = []
    with ThreadPoolExecutor(max_workers=plan["max_concurrency"]) as pool:
        futures = [pool.submit(run_pair, item) for item in enumerate(plan["prefixes"])]
        try:
            for future in as_completed(futures):
                rows.extend(future.result())
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda row: (row["key"], row["profile"]))
    # Recheck immutable inputs before publishing completion, never the live checkout.
    verify_bundle(bundle, plan)
    index = {(r["key"], r["profile"]): r for r in rows}
    paired = []
    for source in plan["prefixes"]:
        a, b = (index[source["key"], p] for p in PROFILES)
        paired.append({"task": source["task"], "query_id": source["query_id"], "origin_profile": source["origin_profile"],
                       "differences": {m: b["metrics"][m] - a["metrics"][m]
                                       for m in ("recall_at_10", "precision_at_10", "f1_at_10", "ndcg_at_10")},
                       "gold_hits_difference": b["gold_hits"] - a["gold_hits"],
                       "non_gold_difference": b["non_gold_returned"] - a["non_gold_returned"]})
    summary = {"schema_version": "dci-final-replay-summary-v1", "bundle": str(bundle),
               "plan_sha256": file_hash(bundle / "plan.json"), "unique_queries": plan["unique_queries"],
               "prefix_count": len(paired), "completed_chains": len(rows), "exclusions": plan["exclusions"],
               "design": plan["design"], "transport": plan["transport"], "paired_prefixes": paired, "samples": rows,
               "valid_by_profile": {p: sum(r["valid_output"] for r in rows if r["profile"] == p) for p in PROFILES},
               "interpretation": "Saved-trajectory final-selection diagnostic; 59 independent query clusters, each with two origins. Not a full retrieval benchmark or 118 independent queries."}
    write_json(output / "summary.json", summary)
    write_json(output / "_COMPLETE", {"summary_sha256": json_hash(summary), "completed_chains": len(rows),
                                      "valid_by_profile": summary["valid_by_profile"]})
    print(f"[replay] complete: {output / 'summary.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bundle", type=Path)
    parser.add_argument("--pilots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--evaluate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.bundle:
        if not args.runtime_bundle or not args.pilots or not args.output or args.preflight_only or args.dry_run or args.evaluate:
            parser.error("preparation requires --runtime-bundle --pilots ... --output")
        prepare(args.runtime_bundle.resolve(), [p.resolve() for p in args.pilots], args.output.resolve())
        return
    if args.runtime_bundle or args.pilots:
        parser.error("cannot mix preparation and launch options")
    if args.output and not args.evaluate:
        parser.error("set REPLAY_ROOT for a fresh launch output; --output is internal to --evaluate")
    bundle = args.bundle.resolve()
    plan = read_json(bundle / "plan.json")
    verify_bundle(bundle, plan)
    env = bundle_env(bundle, plan)
    check_inputs(bundle, plan, env)
    if args.preflight_only or args.dry_run or (os.environ.get("DRY_RUN") == "1" and not args.evaluate):
        print(json.dumps({"planned_final_chains": plan["planned_final_chains"], "queries": plan["unique_queries"],
                          "TP": env["TENSOR_PARALLEL_SIZE"], "concurrency": 4, "runtime": plan["runtime"]}))
        return
    if args.evaluate:
        if not args.output or not args.output.is_dir():
            parser.error("internal --evaluate requires a new serving output directory")
        run_evaluation(bundle, plan, env, args.output.resolve())
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default = Path(plan["source_pilots"][-1]).parents[1] / "harness-replay-runs" / f"{timestamp}-final-replay-{os.getpid()}"
    output = Path(os.environ.get("REPLAY_ROOT", str(default))).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    write_json(output / "lineage.json", {"bundle": str(bundle), "plan_sha256": file_hash(bundle / "plan.json"),
                                         "planned_final_chains": plan["planned_final_chains"]})
    print(f"[replay] output={output}", flush=True)
    os.execvpe("bash", ["bash", str(bundle / "serve.sh"), str(bundle), str(output), plan["runtime"]], env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[replay] refused: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
