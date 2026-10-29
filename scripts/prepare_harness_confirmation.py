#!/usr/bin/env python3
"""Freeze a new-query, end-to-end stable/selective comparison without using GPUs.

The source is a previously verified continuation, not the development checkout.
Only the pilot orchestrator is extended; protocol, scorer, agent, dependencies,
and all six document workspaces are physically copied and made read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from prepare_harness_pilot_continuation import (
    BENCH_PYTHON, SERVING_PYTHON, file_hash, inventory, json_hash,
    read_json, verify_group, write_json,
)

PROFILES = ["budgeted-stable", "budgeted-selective"]
CONTRACTS = {
    "budgeted-stable": "e155daec787f39c79ce6f3a6b18383ca792f452b863ec5ef819ea346ac0d12e9",
    "budgeted-selective": "c6f4a9e46de4f0ebf9b21aca6f73c86a7b1dfd726783362bc3c476807be6419f",
}
SEED = "dci-selection-confirmation-20261015-v1"
DESCRIPTION = (
    "End-to-end confirmation after final-prompt replay; metadata-only selection, "
    "excluding all archived harness-pilot queries and diagnostic/holdout/validation suites. "
    "Historical full benchmark runs may contain these queries; not absolutely unseen."
)
FREEZE_FILE = "scripts/harness_confirmation_queries.json"
HISTORY_FILE = "scripts/harness_confirmation_history.json"
SELF_FILE = "scripts/prepare_harness_confirmation.py"


def query_ids_from_record(value, tasks):
    """Read ID fields only; never use relevance labels or historical scores."""
    result = {}
    ids = value.get("query_ids")
    if isinstance(ids, dict):
        result = {task: list(ids[task]) for task in tasks if task in ids}
    elif isinstance(ids, list) and value.get("task") in tasks:
        result = {value["task"]: list(ids)}
    elif all(task in value and isinstance(value[task], list) for task in tasks):
        result = {task: list(value[task]) for task in tasks}
    argv = value.get("argv", [])
    if "--task" in argv and "--query-ids" in argv:
        task = argv[argv.index("--task") + 1]
        start = argv.index("--query-ids") + 1
        stop = next((i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv))
        if task in tasks:
            result[task] = argv[start:stop]
    for task, selected in result.items():
        if not all(isinstance(q, str) and q for q in selected) or len(set(selected)) != len(selected):
            raise ValueError(f"invalid historical query IDs: {task}")
    return result


def collect_history(repository, tasks):
    paths = [repository / "scripts" / name for name in (
        "harness_pilot_queries.json", "harness_pilot_holdout_queries.json",
        "harness_pilot_validation_queries.json")]
    listing = subprocess.check_output([
        "rg", "--files", "results/harness-pilots", "-g", "plan.json",
        "-g", "run-manifest.json", "-g", "command-*.json",
    ], cwd=repository, text=True)
    paths.extend(repository / line for line in listing.splitlines())
    evidence = []
    for path in sorted(set(paths)):
        ids = query_ids_from_record(read_json(path), tasks)
        if ids:
            evidence.append({"path": str(path.relative_to(repository)),
                             "sha256": file_hash(path), "query_ids": ids})
    if not evidence:
        raise ValueError("no prior query-selection evidence found")
    return {"schema_version": "dci-confirmation-history-v1", "selection_inputs": evidence,
            "historical_exposure": "Historical full benchmark queries are not excluded; no absolutely-unseen claim."}


def compute_freeze(runtime, history, take):
    if not 1 <= take <= 20:
        raise ValueError("confirmation take must be between 1 and 20")
    tasks = list(read_json(runtime / "scripts/harness_pilot_queries.json"))
    sources, excluded, selected, pools = {}, {}, {}, {}
    for task in tasks:
        path = runtime / "data/metadata" / task / "queries.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [row["_id"] for row in records]
        if not ids or not all(isinstance(q, str) and q for q in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid or duplicate metadata query IDs: {task}")
        used = {q for entry in history["selection_inputs"] for q in entry["query_ids"].get(task, [])}
        # Some experimental primers can refer to a different dataset revision.
        # Preserve those IDs in the evidence; exclude every matching canonical ID.
        excluded[task] = sorted(used & set(ids))
        candidates = set(ids) - used
        if len(candidates) < take:
            raise ValueError(f"{task}: only {len(candidates)} unused pilot queries, need {take}")
        selected[task] = sorted(candidates, key=lambda q: (
            hashlib.sha256(f"{SEED}:{task}:{q}".encode()).hexdigest(), q))[:take]
        sources[task] = {"path": str(path.relative_to(runtime)), "sha256": file_hash(path), "count": len(ids)}
        pools[task] = {"total": len(ids), "excluded": len(excluded[task]), "remaining": len(candidates)}
    return {"schema_version": "dci-harness-confirmation-v1", "suite": "confirmation",
            "description": DESCRIPTION, "seed": SEED, "take": take,
            "selection_method": "ascending SHA256(seed + ':' + task + ':' + query_id)",
            "query_ids": selected, "excluded_query_ids": excluded, "pools": pools,
            "source_queries": sources, "history_sha256": json_hash(history),
            "selected_sha256": {t: json_hash(ids) for t, ids in selected.items()},
            "profile_contract_sha256": CONTRACTS}


def load_confirmation(runtime):
    frozen = read_json(runtime / FREEZE_FILE)
    history = read_json(runtime / HISTORY_FILE)
    if frozen != compute_freeze(runtime, history, frozen["take"]):
        raise ValueError("confirmation query freeze changed")
    return frozen


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"archived launcher patch anchor changed: {old[:85]}")
    return text.replace(old, new, 1)


def patch_pilot(text):
    edits = [
        ('QUERY_SUITE_CHOICES = ("diagnostic", "holdout", "validation")',
         'QUERY_SUITE_CHOICES = ("diagnostic", "holdout", "validation", "confirmation")'),
        ('choices=range(1, 11), default=10', 'choices=range(1, 21), default=10'),
        ('    if args.preflight_only:\n        try:\n',
         '    if args.preflight_only:\n        try:\n'
         '            if args.query_suite == "confirmation":\n'
         '                from prepare_harness_confirmation import load_confirmation\n'
         '                frozen = load_confirmation(REPO)\n'
         '                _check_validation_profiles(_describe_profiles(args.profiles), frozen)\n'
         '                print("[pilot] confirmation query and protocol freeze passed", flush=True)\n'),
        ('    validation = None\n    if args.query_suite == "holdout":',
         '    validation = None\n    confirmation = None\n'
         '    if args.query_suite == "confirmation":\n'
         '        from prepare_harness_confirmation import load_confirmation\n'
         '        confirmation = load_confirmation(REPO)\n'
         '        if args.query_limit != confirmation["take"] or args.tasks:\n'
         '            parser.error("confirmation requires all frozen tasks and the exact frozen take")\n'
         '        suite = confirmation["query_ids"]\n'
         '        selection = confirmation["description"]\n'
         '        freeze_path = REPO / "scripts/harness_confirmation_queries.json"\n'
         '        suite_source = {"path": _relative_source_path(freeze_path), "sha256": _file_sha256(freeze_path)}\n'
         '    elif args.query_suite == "holdout":'),
        ('if args.query_suite in ("holdout", "validation"):',
         'if args.query_suite in ("holdout", "validation", "confirmation"):'),
        ('    profiles = _describe_profiles(args.profiles)\n',
         '    profiles = _describe_profiles(args.profiles)\n'
         '    if confirmation is not None:\n'
         '        _check_validation_profiles(profiles, confirmation)\n'),
        ('    if args.dry_run:\n        print(json.dumps(plan, ensure_ascii=False, indent=2))',
         '    if confirmation is not None:\n        plan["confirmation_freeze"] = confirmation\n'
         '    if args.dry_run:\n        print(json.dumps(plan, ensure_ascii=False, indent=2))'),
        ('    for item in execution_order:\n        profile, task = item["profile"], item["task"]',
         '    if confirmation is not None:\n        write_json(output / "confirmation-freeze.json", confirmation)\n'
         '    for item in execution_order:\n        profile, task = item["profile"], item["task"]'),
    ]
    for old, new in edits:
        text = replace_once(text, old, new)
    return text


def patch_launcher(text):
    text = replace_once(text,
        '"$SERVING_PYTHON" scripts/harness_pilot_runtime.py --output "$PILOT_ROOT/runtime.json"\n',
        '"$BENCH_PYTHON" scripts/prepare_harness_confirmation.py --bundle "$DCI_CONFIRMATION_BUNDLE" --check-ports\n'
        '"$SERVING_PYTHON" scripts/harness_pilot_runtime.py --output "$PILOT_ROOT/runtime.json"\n'
        '"$BENCH_PYTHON" scripts/prepare_harness_confirmation.py --bundle "$DCI_CONFIRMATION_BUNDLE" --check-runtime "$PILOT_ROOT/runtime.json"\n')
    text = replace_once(text,
        'echo "[pilot] synthetic serving warmup: concurrency=$MAX_CONCURRENCY deadline=${WARMUP_DEADLINE_SECONDS}s"',
        '"$BENCH_PYTHON" scripts/prepare_harness_confirmation.py --bundle "$DCI_CONFIRMATION_BUNDLE" --check-backend "$PILOT_ROOT/backend.json"\n'
        'echo "[pilot] synthetic serving warmup: concurrency=$MAX_CONCURRENCY deadline=${WARMUP_DEADLINE_SECONDS}s"')
    return replace_once(text,
        '  --backend-artifact "$PILOT_ROOT/warmup.json" "$@"\n',
        '  --backend-artifact "$PILOT_ROOT/warmup.json" "$@"\n'
        '"$BENCH_PYTHON" scripts/prepare_harness_confirmation.py --bundle "$DCI_CONFIRMATION_BUNDLE" --finish "$PILOT_ROOT"\n')


def python_identity(python):
    return json.loads(subprocess.check_output([python, "-c", (
        "import json,platform; from importlib.metadata import version; "
        "print(json.dumps({'python':platform.python_version(), 'packages':"
        "{k:version(k) for k in ('inspect-ai','openai','httpx','aiohttp')}}))"
    )], text=True))


def prepare(source, repository, output, backend_path, take):
    if {'results', 'eval_data', 'raw_data', 'eval_data_code'} & set(output.resolve().parts):
        raise ValueError('Executable bundles belong under the code repository releases/ directory')
    preparer_source = Path(__file__).read_bytes()
    source_manifest = read_json(source / "continuation.json")
    original = source / "runtime"
    if inventory(original, Path(source_manifest["shared_workspaces"])) != source_manifest["runtime_inventory"]:
        raise ValueError("source continuation runtime changed")
    tasks = list(read_json(original / "scripts/harness_pilot_queries.json"))
    history = collect_history(repository, tasks)
    frozen = compute_freeze(original, history, take)
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    print("[confirmation] copying verified code, installed dependencies, and six document workspaces", flush=True)
    # Native copy avoids Python overhead for the many small dependency files.
    # cp -a preserves internal relative links without following the data link.
    subprocess.run(["cp", "-a", "--", str(original), str(runtime)], check=True)
    # Make only new directories writable during assembly, never the source.
    for base, dirs, _ in os.walk(runtime, followlinks=False):
        Path(base).chmod(Path(base).stat().st_mode | 0o200)
    (runtime / "data/workspaces").unlink()  # Newly copied external link, not source data.
    (runtime / "data/workspaces").mkdir()
    for task in tasks:
        shutil.copytree(original / "data/workspaces" / task, runtime / "data/workspaces" / task)
    diffs = []
    for name, patcher in (("scripts/run_harness_pilot.py", patch_pilot),
                          ("scripts/run_harness_pilot.sh", patch_launcher)):
        path = runtime / name
        before = path.read_text()
        after = patcher(before)
        path.chmod(path.stat().st_mode | 0o200)
        path.write_text(after)
        diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile="original/" + name, tofile="confirmation/" + name))
    (runtime / SELF_FILE).write_bytes(preparer_source)
    shutil.copyfile(source / "continue.py", runtime / "scripts/prepare_harness_pilot_continuation.py")
    write_json(runtime / HISTORY_FILE, history)
    write_json(runtime / FREEZE_FILE, frozen)
    shutil.copyfile(backend_path, output / "expected-backend.json")
    (output / "orchestration.patch").write_text("".join(diffs))
    write_json(output / "analysis-plan.json", {
        "primary": "all-sample task-macro F1@10, invalid samples scored zero",
        "pairing": "one stable/selective pair per unique query; six tasks weighted equally",
        "uncertainty": "paired bootstrap stratified by task, 100000 draws, seed 20261015",
        "secondary": ["Recall@10", "Precision@10 (actual returned window)", "nDCG@10",
                      "valid_output rate", "failure reasons", "total model tokens", "wall seconds"],
        "decision": {"F1": "paired 95% interval lower bound > 0",
                     "recall": "paired 95% interval lower bound >= -0.02 (two absolute percentage points)",
                     "validity": "selective validity no more than one percentage point below stable",
                     "cost": "total token ratio <= 1.10 when all usage is known; otherwise report inconclusive",
                     "otherwise": "retain stable as control; no automatic protocol change or sample replacement"},
        "scope": DESCRIPTION,
    })
    launch = output / "launch.sh"
    launch.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\n'
        'bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'mkdir -p "$bundle/logs"\n'
        'driver_log="$bundle/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"\n'
        'exec > >(tee "$driver_log") 2>&1\n'
        'echo "[confirmation] driver log: $driver_log"\n'
        f'exec {BENCH_PYTHON} -B "$bundle/runtime/{SELF_FILE}" --bundle "$bundle" "$@"\n')
    launch.chmod(0o755)
    current = inventory(runtime, runtime / "data/workspaces")
    changed = [name for name, sha in source_manifest["runtime_inventory"]["files"].items()
               if current["files"].get(name) != sha]
    if sorted(changed) != ["scripts/run_harness_pilot.py", "scripts/run_harness_pilot.sh"]:
        raise ValueError(f"unexpected inherited file changes: {changed}")
    manifest = {
        "schema_version": "dci-confirmation-bundle-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(source), "source_manifest_sha256": file_hash(source / "continuation.json"),
        "source_inventory_sha256": json_hash(source_manifest["runtime_inventory"]),
        "inherited_changed_files": changed, "runtime_inventory": current,
        "query_ids": frozen["query_ids"], "take": take, "profiles": PROFILES,
        "contract_sha256": CONTRACTS, "planned_model_evaluations": take * len(tasks) * len(PROFILES),
        "output_parent": str(repository / "results/harness-confirmation-runs"),
        "bench_python": BENCH_PYTHON, "serving_python": SERVING_PYTHON,
        "bench_identity": python_identity(BENCH_PYTHON),
        "node_version": subprocess.check_output([str(Path(BENCH_PYTHON).with_name("node")), "--version"], text=True).strip(),
        "files": {name: file_hash(output / name) for name in (
            "launch.sh", "expected-backend.json", "orchestration.patch", "analysis-plan.json")},
    }
    if Path(__file__).read_bytes() != preparer_source:
        raise ValueError("preparation script changed during copying; prepare a fresh bundle")
    write_json(output / "confirmation.json", manifest)
    (output / "confirmation.sha256").write_text(file_hash(output / "confirmation.json") + "\n")
    for base, dirs, names in os.walk(runtime, topdown=False, followlinks=False):
        for name in dirs + names:
            path = Path(base) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
    runtime.chmod(0o555)
    print(json.dumps({"pools": frozen["pools"], "planned_evaluations": manifest["planned_model_evaluations"],
                      "launch": str(launch)}, ensure_ascii=False, indent=2), flush=True)


def verify_bundle(bundle):
    if file_hash(bundle / "confirmation.json") != (bundle / "confirmation.sha256").read_text().strip():
        raise ValueError("confirmation manifest changed")
    manifest = read_json(bundle / "confirmation.json")
    for name, sha in manifest["files"].items():
        if file_hash(bundle / name) != sha:
            raise ValueError(f"bundle file changed: {name}")
    runtime = bundle / "runtime"
    if inventory(runtime, runtime / "data/workspaces") != manifest["runtime_inventory"]:
        raise ValueError("frozen code, dependencies or data changed")
    frozen = load_confirmation(runtime)
    if frozen["query_ids"] != manifest["query_ids"] or frozen["profile_contract_sha256"] != manifest["contract_sha256"]:
        raise ValueError("bundle query or profile identity changed")
    return manifest


def launch_environment(bundle, manifest):
    backend = read_json(bundle / "expected-backend.json")
    env = os.environ.copy()
    for key in ("NODE_PATH", "NODE_OPTIONS", "DCI_CODE_RUNTIME_CONTRACT", "DCI_PROTOCOL_CONTRACT_SHA256",
                "KV_CACHE_DTYPE", "VLLM_NVFP4_GEMM_BACKEND", "PYTHONHOME"):
        env.pop(key, None)
    env.update({"PYTHONPATH": str(bundle / "runtime"), "PYTHONDONTWRITEBYTECODE": "1", "TSX_DISABLE_CACHE": "1",
                "DCI_EVALUATION_TRACK": "document", "DCI_PROTOCOL_PROFILE": "v3",
                "DCI_CONFIRMATION_BUNDLE": str(bundle), "BENCH_PYTHON": manifest["bench_python"],
                "SERVING_PYTHON": manifest["serving_python"], "TP_SIZE": "4", "TENSOR_PARALLEL_SIZE": "4",
                "MAX_CONCURRENCY": "4", "VLLM_MAX_NUM_SEQS": "4", "VLLM_GPU_MEMORY_UTILIZATION": "0.9",
                "VLLM_EXTRA_ARGS": "--enforce-eager", "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
                "MODEL_PATH": backend["model"]["source_path"], "SERVED_MODEL_NAME": backend["model"]["served_model_name"],
                "VLLM_PORT": env.get("VLLM_PORT", "28099"), "WARMUP_DEADLINE_SECONDS": "600",
                "WARMUP_HTTP_TIMEOUT_SECONDS": "600"})
    env["PATH"] = str(Path(manifest["bench_python"]).parent) + os.pathsep + env.get("PATH", "")
    return env


def check_ports(env):
    serving = int(env.get("VLLM_PORT", "28099"))
    ports = [serving, *range(28131, 28131 + 12 * 4)]
    if len(ports) != len(set(ports)) or not 1024 <= serving <= 65535:
        raise ValueError("invalid or overlapping serving and bridge ports")
    opened = []
    try:
        for port in ports:
            sock = socket.socket()
            opened.append(sock)
            sock.bind(("127.0.0.1", port))
    finally:
        for sock in opened:
            sock.close()


def check_runtime(bundle, actual):
    expected = read_json(bundle / "expected-backend.json")["backend"]["runtime"]
    for field in ("packages", "torch_cuda", "python"):
        if actual[field] != expected[field]:
            raise ValueError(f"serving runtime changed: {field}")
    identity = lambda x: [(g["name"], g["capability"], g["total_memory_bytes"]) for g in x["cuda"]["gpus"]]
    if identity(actual) != identity(expected):
        raise ValueError("GPU configuration must match the previous four-H100 run")


def check_backend(bundle, actual):
    expected = read_json(bundle / "expected-backend.json")
    if actual["model"] != expected["model"]:
        raise ValueError("model configuration or weights changed")
    for field in ("kind", "version", "tensor_parallel_size", "context_length", "dtype", "max_num_seqs",
                  "gpu_memory_utilization", "generation_config", "tool_call_parser", "reasoning_parser",
                  "language_model_only", "sampling_backend"):
        if actual["backend"][field] != expected["backend"][field]:
            raise ValueError(f"backend changed: {field}")


def verify_plan(manifest, plan):
    if plan["query_ids"] != manifest["query_ids"] or plan["query_suite"] != "confirmation":
        raise ValueError("planned query IDs or suite changed")
    if {p: v["sha256"] for p, v in plan["profiles"].items()} != manifest["contract_sha256"]:
        raise ValueError("planned profile contracts changed")
    from prepare_harness_pilot_continuation import execution_order
    if (plan["max_concurrency"] != 4 or plan["profile_order"] != "alternating"
            or plan["execution_order"] != execution_order(list(manifest["query_ids"]), PROFILES, "alternating")):
        raise ValueError("planned concurrency or paired order changed")


def finish_run(bundle, output, manifest):
    verify_bundle(bundle)
    plan = read_json(output / "evaluation/plan.json")
    verify_plan(manifest, plan)
    marker = read_json(output / "evaluation/_COMPLETE")
    if marker["rows"] != 12 or marker["summary_sha256"] != file_hash(output / "evaluation/summary.json"):
        raise ValueError("pilot summary completion mismatch")
    valid, completed = 0, 0
    groups = []
    for item in plan["execution_order"]:
        run = output / "evaluation/runs" / plan["model_key"] / item["task"] / "runs" / f"pilot-{item['profile']}"
        group = verify_group(run, plan, item["task"], item["profile"])
        contract = read_json(run / "run-manifest.json")["benchmark_contract"]
        context = contract.pop("task_context")
        if context["task_name"] != item["task"] or contract != plan["profiles"][item["profile"]]["contract"]:
            raise ValueError("completed task protocol differs from its frozen plan")
        result = read_json(run / "task-result.json")
        valid += result["counts"]["valid"]
        completed += result["counts"]["completed"]
        groups.append(group)
    if completed != manifest["planned_model_evaluations"]:
        raise ValueError("completed sample count differs from the frozen plan")
    write_json(output / "completion-audit.json", {"groups": groups, "completed": completed, "valid": valid,
               "invalid": completed - valid, "interpretation": "Completion does not imply every sample is valid."})
    write_json(output / "_COMPLETE", {"schema_version": "dci-confirmation-complete-v1", "completed": completed,
               "valid": valid, "completion_audit_sha256": file_hash(output / "completion-audit.json"),
               "bundle_manifest_sha256": file_hash(bundle / "confirmation.json")})
    print(f"[confirmation] complete: {output}; completed={completed}, valid={valid}", flush=True)


def run_bundle(bundle, *, dry_run=False, preflight_only=False):
    manifest = verify_bundle(bundle)
    runtime = bundle / "runtime"
    env = launch_environment(bundle, manifest)
    python = manifest["bench_python"]
    if python_identity(python) != manifest["bench_identity"]:
        raise ValueError("benchmark Python packages changed")
    if subprocess.check_output([str(Path(python).with_name("node")), "--version"], text=True).strip() != manifest["node_version"]:
        raise ValueError("Node runtime changed")
    subprocess.run([python, "-c", "from pathlib import Path; import dci_bench; from dci_bench.agents import pi_agent; "
                    "assert Path(dci_bench.__file__).resolve().is_relative_to(Path.cwd()); "
                    "assert pi_agent.repo_root()==Path.cwd()"], cwd=runtime, env=env, check=True)
    subprocess.run(["node", "--input-type=module", "-e", (
        "import {fileURLToPath} from 'node:url'; import {realpathSync} from 'node:fs'; "
        "const root=realpathSync(process.cwd())+'/'; "
        "for (const spec of ['@earendil-works/pi-agent-core','@earendil-works/pi-ai',"
        "'@earendil-works/pi-telemetry','typebox','tsx']) "
        "if (!realpathSync(fileURLToPath(import.meta.resolve(spec))).startsWith(root)) throw Error('Dependency escapes snapshot');"
    )], cwd=runtime / "pi-dci", env=env, check=True)
    args = ["--profiles", *PROFILES, "--query-suite", "confirmation", "--query-limit", str(manifest["take"]),
            "--profile-order", "alternating", "--max-concurrency", "4"]
    command = [python, "scripts/run_harness_pilot.py"]
    subprocess.run([*command, "--preflight-only", *args], cwd=runtime, env=env, check=True)
    plan = json.loads(subprocess.check_output([*command, "--backend-manifest", str(bundle / "expected-backend.json"),
                                              "--dry-run", *args], cwd=runtime, env=env, text=True))
    verify_plan(manifest, plan)
    if dry_run or preflight_only or env.get("DRY_RUN") == "1":
        subprocess.run(["bash", "scripts/run_harness_pilot.sh", *args], cwd=runtime,
                       env={**env, "DRY_RUN": "1"}, check=True)
        if preflight_only:
            subprocess.run([python, "scripts/run_all.py", "--preflight-only"], cwd=runtime, env=env, check=True)
            for task in manifest["query_ids"]:
                for profile in PROFILES:
                    subprocess.run(["npm", "exec", "--", "tsx", "scripts/dci-run.ts", "--preflight-only",
                                    "--workspace", str(runtime / "data/workspaces" / task)],
                                   cwd=runtime / "pi-dci", env={**env, "DCI_PROTOCOL_PROFILE": profile}, check=True)
            check_ports(env)
        print(json.dumps({"mode": "CPU-preflight" if preflight_only else "dry-run", "gpu_started": False,
                          "query_ids": plan["query_ids"], "planned_evaluations": manifest["planned_model_evaluations"]}, indent=2), flush=True)
        return
    check_ports(env)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(env.get("PILOT_ROOT", str(Path(manifest["output_parent"]) / f"{stamp}-confirmation-{os.getpid()}"))).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse pilot output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_name(output.name + ".lineage.json"), {
        "bundle": str(bundle), "bundle_manifest_sha256": file_hash(bundle / "confirmation.json"),
        "planned_evaluations": manifest["planned_model_evaluations"], "output": str(output),
    })
    env["PILOT_ROOT"] = str(output)
    env.pop("DRY_RUN", None)
    print(f"[confirmation] launching full search/read evaluation: {output}", flush=True)
    # The shell retains its original signal traps and owns the serving process.
    # Its final command calls --finish only after all evaluation groups finish.
    os.chdir(runtime)
    os.execvpe("bash", ["bash", "scripts/run_harness_pilot.sh", *args], env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-backend", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--take", type=int, default=20)
    parser.add_argument("--bundle", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--check-runtime", type=Path)
    modes.add_argument("--check-backend", type=Path)
    modes.add_argument("--check-ports", action="store_true")
    modes.add_argument("--finish", type=Path)
    args = parser.parse_args(argv)
    if args.bundle:
        bundle = args.bundle.resolve()
        if args.check_runtime:
            check_runtime(bundle, read_json(args.check_runtime))
        elif args.check_backend:
            check_backend(bundle, read_json(args.check_backend))
        elif args.check_ports:
            check_ports(os.environ)
        elif args.finish:
            finish_run(bundle, args.finish.resolve(), read_json(bundle / "confirmation.json"))
        else:
            run_bundle(bundle, dry_run=args.dry_run, preflight_only=args.preflight_only)
    elif args.source_bundle and args.expected_backend and args.output:
        prepare(args.source_bundle.resolve(), args.repository.resolve(), args.output.resolve(),
                args.expected_backend.resolve(), args.take)
    else:
        parser.error("provide --bundle, or --source-bundle/--expected-backend/--output")


if __name__ == "__main__":
    main()
