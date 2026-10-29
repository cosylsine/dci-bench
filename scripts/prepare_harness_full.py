#!/usr/bin/env python3
"""Freeze and launch the accepted selective document protocol on all six tasks.

Preparation and --preflight-only are CPU-only. The inherited solver, scorer,
protocol, corpus, and dependencies are never changed. Only orchestration changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys

import prepare_harness_confirmation as confirmation
from prepare_harness_pilot_continuation import (
    BENCH_PYTHON, SERVING_PYTHON, execution_order, file_hash, inventory,
    json_hash, read_json, verify_group, write_json,
)

PROFILE = "budgeted-selective"
CONTRACT_SHA256 = "c6f4a9e46de4f0ebf9b21aca6f73c86a7b1dfd726783362bc3c476807be6419f"
COUNTS = {
    "LLMAILAStatutes": 50, "LLMFQuADRetrieval": 100, "LLMHC3FinanceRetrieval": 100,
    "LLMLegalBenchConsumerContractsQA": 100, "LLMPublicHealthQA": 100,
    "LLMTwitterHjerneRetrieval": 77,
}
MODEL_KEY = "Qwen3.8-27B"
SELF_FILE = "scripts/prepare_harness_full.py"
FREEZE_FILE = "scripts/harness_full_queries.json"
DESCRIPTION = "All 527 canonical queries from the frozen six-task document datasets, in metadata order; one selective arm. Includes prior development and confirmation queries; not an unseen holdout."


def compute_full_freeze(runtime):
    selected, sources = {}, {}
    for task, expected_count in COUNTS.items():
        path = runtime / "data/metadata" / task / "queries.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [row["_id"] for row in records]
        if (len(ids) != expected_count or len(set(ids)) != expected_count
                or not all(isinstance(q, str) and q for q in ids)):
            raise ValueError(f"full query count/identity mismatch: {task}")
        selected[task] = ids
        sources[task] = {"path": str(path.relative_to(runtime)), "sha256": file_hash(path), "count": len(ids)}
    return {"schema_version": "dci-selective-full-query-freeze-v1", "suite": "full",
            "description": DESCRIPTION, "take": max(COUNTS.values()),
            "query_ids": selected, "counts": COUNTS, "source_queries": sources,
            "selected_sha256": {t: json_hash(ids) for t, ids in selected.items()},
            "profile_contract_sha256": {PROFILE: CONTRACT_SHA256}}


def load_full(runtime):
    frozen = read_json(runtime / FREEZE_FILE)
    if frozen != compute_full_freeze(runtime):
        raise ValueError("full query freeze changed")
    return frozen


def patch_pilot(source):
    # This is a separate copy of the successful confirmation orchestrator.
    if 'from prepare_harness_confirmation import load_confirmation' not in source:
        raise ValueError("expected archived confirmation orchestrator")
    result = source.replace("confirmation", "full")
    result = confirmation.replace_once(result, 'choices=range(1, 21), default=10',
                                        'choices=range(1, 101), default=10')
    return result.replace("validation requires both frozen stable/selective profile contracts; hash or profile mismatch",
                          "selected suite requires exactly its frozen profile contracts; hash or profile mismatch")


def patch_launcher(source):
    result = source.replace("prepare_harness_confirmation.py", "prepare_harness_full.py")
    result = result.replace("DCI_CONFIRMATION_BUNDLE", "DCI_FULL_BUNDLE")
    start = result.index('for profile in "${preflight_profiles[@]}"; do\n')
    end = result.index('mkdir -p "$PILOT_ROOT"\n', start)
    result = result[:start] + (
        'PATH="$(dirname "$BENCH_PYTHON"):$PATH" "$BENCH_PYTHON" scripts/prepare_harness_full.py '
        '--bundle "$DCI_FULL_BUNDLE" --check-sandboxes\n'
    ) + result[end:]
    return result


def prepare(source, output, backend_path, results_root):
    if {'results', 'eval_data', 'raw_data', 'eval_data_code'} & set(output.resolve().parts):
        raise ValueError('Executable bundles belong under the code repository releases/ directory')
    preparer_source = Path(__file__).read_bytes()
    original = source / "runtime"
    # Helpers used to prepare the new snapshot must be the same verified ones.
    for module in (confirmation, sys.modules["prepare_harness_pilot_continuation"]):
        if file_hash(Path(module.__file__)) != file_hash(original / "scripts" / Path(module.__file__).name):
            raise ValueError(f"preparation helper differs from verified source: {module.__file__}")
    source_manifest = confirmation.verify_bundle(source)
    backend = read_json(backend_path)
    confirmation.check_backend(source, backend)
    confirmation.check_runtime(source, backend["backend"]["runtime"])
    if backend["model"]["model_key"] != MODEL_KEY or results_root.name != MODEL_KEY:
        raise ValueError("this frozen entry is for the Qwen3.8-27B result root")
    frozen = compute_full_freeze(original)
    if confirmation.python_identity(BENCH_PYTHON) != source_manifest["bench_identity"]:
        raise ValueError("benchmark Python identity differs from confirmation")
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    print("[full] copying verified runtime, dependencies, metadata and six corpora", flush=True)
    subprocess.run(["cp", "-a", "--", str(original), str(runtime)], check=True)
    for base, _, _ in os.walk(runtime, followlinks=False):
        Path(base).chmod(Path(base).stat().st_mode | 0o200)
    diffs = []
    for name, patcher in (("scripts/run_harness_pilot.py", patch_pilot),
                          ("scripts/run_harness_pilot.sh", patch_launcher)):
        path = runtime / name
        before, after = path.read_text(), patcher(path.read_text())
        path.chmod(path.stat().st_mode | 0o200)
        path.write_text(after)
        diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile="confirmation/" + name, tofile="full/" + name))
    (runtime / SELF_FILE).write_bytes(preparer_source)
    write_json(runtime / FREEZE_FILE, frozen)
    shutil.copyfile(backend_path, output / "expected-backend.json")
    (output / "orchestration.patch").write_text("".join(diffs))
    env = {**os.environ, "PYTHONPATH": str(runtime), "PYTHONDONTWRITEBYTECODE": "1",
           "DCI_PROTOCOL_PROFILE": PROFILE, "DCI_EVALUATION_TRACK": "document"}
    env.pop("DCI_PROTOCOL_CONTRACT_SHA256", None)
    contract = json.loads(subprocess.check_output([
        BENCH_PYTHON, "-B", "scripts/run_harness_pilot.py", "--describe-profile"
    ], cwd=runtime, env=env, text=True))
    if contract["sha256"] != CONTRACT_SHA256 or json_hash(contract["contract"]) != CONTRACT_SHA256:
        raise ValueError("selective contract differs from accepted confirmation")
    write_json(output / "selective-contract.json", contract)
    write_json(output / "evaluation-plan.json", {
        "profile": PROFILE, "base_contract_sha256": CONTRACT_SHA256, "model_key": MODEL_KEY,
        "task_counts": COUNTS, "planned_evaluations": sum(COUNTS.values()), "scope": DESCRIPTION,
        "primary_report": "all-sample equal-task macro F1@10; invalid samples remain zero",
        "secondary_report": ["query-weighted means", "valid_only", "Recall@10", "actual-window Precision@10",
                             "nDCG@10", "validity and repair counts", "tokens and latency"],
        "sample_failures": "finish all six tasks, preserve all samples, mark complete_with_sample_failures and exit 1",
    })
    logs = shlex.quote(str(results_root / "job-logs"))
    launch = output / "launch.sh"
    launch.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\n'
        'bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'log_dir={logs}\nmkdir -p "$log_dir"\n'
        'export DCI_FULL_DRIVER_LOG="$(mktemp "$log_dir/selective-full-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")"\n'
        'exec > >(tee "$DCI_FULL_DRIVER_LOG") 2>&1\n'
        'echo "[full] driver log: $DCI_FULL_DRIVER_LOG"\n'
        f'exec {shlex.quote(BENCH_PYTHON)} -B "$bundle/runtime/{SELF_FILE}" --bundle "$bundle" "$@"\n')
    launch.chmod(0o755)
    current = inventory(runtime, runtime / "data/workspaces")
    changed = sorted(name for name, sha in source_manifest["runtime_inventory"]["files"].items()
                     if current["files"].get(name) != sha)
    if changed != ["scripts/run_harness_pilot.py", "scripts/run_harness_pilot.sh"]:
        raise ValueError(f"unexpected inherited file changes: {changed}")
    manifest = {
        "schema_version": "dci-selective-full-bundle-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(source), "source_manifest_sha256": file_hash(source / "confirmation.json"),
        "inherited_changed_files": changed, "runtime_inventory": current,
        "query_ids": frozen["query_ids"], "counts": COUNTS, "take": frozen["take"], "profiles": [PROFILE],
        "contract_sha256": {PROFILE: CONTRACT_SHA256}, "planned_model_evaluations": sum(COUNTS.values()),
        "model_key": MODEL_KEY, "results_root": str(results_root),
        "output_parent": str(results_root / "full-selective-runs"),
        "bench_python": BENCH_PYTHON, "serving_python": SERVING_PYTHON,
        "bench_identity": source_manifest["bench_identity"], "node_version": source_manifest["node_version"],
        "files": {name: file_hash(output / name) for name in
                  ("launch.sh", "expected-backend.json", "orchestration.patch", "selective-contract.json", "evaluation-plan.json")},
    }
    if Path(__file__).read_bytes() != preparer_source:
        raise ValueError("preparer changed during preparation; create a new bundle")
    write_json(output / "frozen.json", manifest)
    (output / "frozen.sha256").write_text(file_hash(output / "frozen.json") + "\n")
    for base, dirs, names in os.walk(runtime, topdown=False, followlinks=False):
        for name in dirs + names:
            path = Path(base) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
    runtime.chmod(0o555)
    print(json.dumps({"tasks": COUNTS, "planned_evaluations": sum(COUNTS.values()),
                      "launch": str(launch), "output_parent": manifest["output_parent"]}, indent=2), flush=True)


def verify_bundle(bundle):
    if file_hash(bundle / "frozen.json") != (bundle / "frozen.sha256").read_text().strip():
        raise ValueError("full bundle manifest changed")
    manifest = read_json(bundle / "frozen.json")
    for name, sha in manifest["files"].items():
        if file_hash(bundle / name) != sha:
            raise ValueError(f"bundle file changed: {name}")
    runtime = bundle / "runtime"
    if inventory(runtime, runtime / "data/workspaces") != manifest["runtime_inventory"]:
        raise ValueError("frozen code, dependencies or data changed")
    frozen = load_full(runtime)
    if (frozen["query_ids"] != manifest["query_ids"]
            or manifest["contract_sha256"] != {PROFILE: CONTRACT_SHA256}
            or manifest["planned_model_evaluations"] != sum(COUNTS.values())):
        raise ValueError("full bundle scope or protocol changed")
    return manifest


def launch_environment(bundle, manifest):
    env = confirmation.launch_environment(bundle, manifest)
    for key in ("DCI_CONFIRMATION_BUNDLE", "PILOT_ROOT", "RESULTS_ROOT", "WORKSPACE_ROOT", "METADATA_ROOT"):
        env.pop(key, None)
    env["DCI_FULL_BUNDLE"] = str(bundle)
    return env


def check_ports(env):
    port = int(env.get("VLLM_PORT", "28099"))
    ports = [port, *range(28131, 28131 + len(COUNTS) * 4)]
    if len(set(ports)) != len(ports) or not 1024 <= port <= 65535:
        raise ValueError("invalid or overlapping serving and bridge ports")
    opened = []
    try:
        for item in ports:
            sock = socket.socket()
            opened.append(sock)
            sock.bind(("127.0.0.1", item))
    finally:
        for sock in opened:
            sock.close()


def check_sandboxes(bundle, manifest):
    runtime = bundle / "runtime"
    env = {**launch_environment(bundle, manifest), "DCI_PROTOCOL_PROFILE": PROFILE}
    for task in COUNTS:
        subprocess.run(["npm", "exec", "--", "tsx", "scripts/dci-run.ts", "--preflight-only",
                        "--workspace", str(runtime / "data/workspaces" / task)],
                       cwd=runtime / "pi-dci", env=env, check=True)


def verify_plan(manifest, plan):
    if plan["query_ids"] != manifest["query_ids"] or plan["query_suite"] != "full":
        raise ValueError("full planned query scope changed")
    if {p: v["sha256"] for p, v in plan["profiles"].items()} != {PROFILE: CONTRACT_SHA256}:
        raise ValueError("full planned profile changed")
    if (plan["model_key"] != MODEL_KEY or plan["max_concurrency"] != 4 or plan["profile_order"] != "as-listed"
            or plan["execution_order"] != execution_order(list(COUNTS), [PROFILE], "as-listed")):
        raise ValueError("full model, concurrency or execution order changed")


def output_path(manifest, env):
    default_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-selective-full-{os.getpid()}"
    run_id = env.get("FULL_RUN_ID", default_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("FULL_RUN_ID must be one safe path component")
    output = (Path(manifest["output_parent"]) / run_id).resolve()
    if not output.is_relative_to(Path(manifest["results_root"]).resolve()):
        raise ValueError("output escapes the frozen Qwen result root")
    if output.exists() or output.with_name(output.name + ".lineage.json").exists():
        raise FileExistsError(f"refusing to reuse full output: {output}")
    return output


def finish_run(bundle, output, manifest):
    verify_bundle(bundle)
    if not output.is_relative_to(Path(manifest["output_parent"]).resolve()):
        raise ValueError("completion output escapes frozen result parent")
    plan = read_json(output / "evaluation/plan.json")
    verify_plan(manifest, plan)
    marker = read_json(output / "evaluation/_COMPLETE")
    if marker["rows"] != len(COUNTS) or marker["summary_sha256"] != file_hash(output / "evaluation/summary.json"):
        raise ValueError("full summary completion mismatch")
    completed = valid = 0
    groups = []
    for task in COUNTS:
        run = output / "evaluation/runs" / MODEL_KEY / task / "runs" / f"pilot-{PROFILE}"
        groups.append(verify_group(run, plan, task, PROFILE))
        contract = read_json(run / "run-manifest.json")["benchmark_contract"]
        context = contract.pop("task_context")
        if context["task_name"] != task or contract != plan["profiles"][PROFILE]["contract"]:
            raise ValueError("completed task protocol differs from frozen plan")
        result = read_json(run / "task-result.json")
        completed += result["counts"]["completed"]
        valid += result["counts"]["valid"]
    if completed != manifest["planned_model_evaluations"]:
        raise ValueError("completed sample count differs from full plan")
    status = "complete" if valid == completed else "complete_with_sample_failures"
    write_json(output / "completion-audit.json", {"groups": groups, "completed": completed, "valid": valid,
               "invalid": completed - valid, "status": status})
    write_json(output / "_COMPLETE", {"schema_version": "dci-selective-full-complete-v1", "status": status,
               "completed": completed, "valid": valid, "invalid": completed - valid,
               "completion_audit_sha256": file_hash(output / "completion-audit.json"),
               "bundle_manifest_sha256": file_hash(bundle / "frozen.json")})
    print(f"[full] {status}: {output}; completed={completed}, valid={valid}", flush=True)
    return 0 if valid == completed else 1


def run_bundle(bundle, *, dry_run=False, preflight_only=False):
    manifest = verify_bundle(bundle)
    runtime, python = bundle / "runtime", manifest["bench_python"]
    env = launch_environment(bundle, manifest)
    if confirmation.python_identity(python) != manifest["bench_identity"]:
        raise ValueError("benchmark Python packages changed")
    if subprocess.check_output([str(Path(python).with_name("node")), "--version"], text=True).strip() != manifest["node_version"]:
        raise ValueError("Node runtime changed")
    subprocess.run([python, "-B", "-c", "from pathlib import Path; import dci_bench; from dci_bench.agents import pi_agent; "
                    "assert Path(dci_bench.__file__).resolve().is_relative_to(Path.cwd()); "
                    "assert pi_agent.repo_root()==Path.cwd()"], cwd=runtime, env=env, check=True)
    subprocess.run(["node", "--input-type=module", "-e", (
        "import {fileURLToPath} from 'node:url'; import {realpathSync} from 'node:fs'; "
        "const root=realpathSync(process.cwd())+'/'; "
        "for (const spec of ['@earendil-works/pi-agent-core','@earendil-works/pi-ai',"
        "'@earendil-works/pi-telemetry','typebox','tsx']) "
        "if (!realpathSync(fileURLToPath(import.meta.resolve(spec))).startsWith(root)) throw Error('Dependency escapes snapshot');"
    )], cwd=runtime / "pi-dci", env=env, check=True)
    args = ["--profiles", PROFILE, "--query-suite", "full", "--query-limit", str(manifest["take"]),
            "--profile-order", "as-listed", "--max-concurrency", "4"]
    command = [python, "-B", "scripts/run_harness_pilot.py"]
    subprocess.run([*command, "--preflight-only", *args], cwd=runtime, env=env, check=True)
    plan = json.loads(subprocess.check_output([*command, "--backend-manifest", str(bundle / "expected-backend.json"),
                                              "--dry-run", *args], cwd=runtime, env=env, text=True))
    verify_plan(manifest, plan)
    output = output_path(manifest, env)
    env["PILOT_ROOT"] = str(output)
    if dry_run or preflight_only or env.get("DRY_RUN") == "1":
        subprocess.run(["bash", "scripts/run_harness_pilot.sh", *args], cwd=runtime,
                       env={**env, "DRY_RUN": "1"}, check=True)
        if preflight_only:
            subprocess.run([python, "-B", "scripts/run_all.py", "--preflight-only"], cwd=runtime, env=env, check=True)
            check_sandboxes(bundle, manifest)
            check_ports(env)
        print(json.dumps({"mode": "CPU-preflight" if preflight_only else "dry-run", "gpu_started": False,
                          "task_counts": COUNTS, "profiles": [PROFILE], "contract_sha256": CONTRACT_SHA256,
                          "planned_evaluations": sum(COUNTS.values()), "output": str(output)}, indent=2), flush=True)
        return 0
    check_ports(env)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_name(output.name + ".lineage.json"), {
        "bundle": str(bundle), "bundle_manifest_sha256": file_hash(bundle / "frozen.json"),
        "base_contract_sha256": CONTRACT_SHA256, "planned_evaluations": sum(COUNTS.values()),
        "output": str(output), "driver_log": env.get("DCI_FULL_DRIVER_LOG"),
    })
    env.pop("DRY_RUN", None)
    print(f"[full] launching 527 document evaluations: {output}", flush=True)
    os.chdir(runtime)
    os.execvpe("bash", ["bash", "scripts/run_harness_pilot.sh", *args], env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--expected-backend", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--check-runtime", type=Path)
    modes.add_argument("--check-backend", type=Path)
    modes.add_argument("--check-ports", action="store_true")
    modes.add_argument("--check-sandboxes", action="store_true")
    modes.add_argument("--finish", type=Path)
    args = parser.parse_args(argv)
    if args.bundle:
        bundle = args.bundle.resolve()
        if args.check_runtime:
            confirmation.check_runtime(bundle, read_json(args.check_runtime))
        elif args.check_backend:
            confirmation.check_backend(bundle, read_json(args.check_backend))
        elif args.check_ports:
            check_ports(os.environ)
        elif args.check_sandboxes:
            check_sandboxes(bundle, read_json(bundle / "frozen.json"))
        elif args.finish:
            return finish_run(bundle, args.finish.resolve(), read_json(bundle / "frozen.json"))
        else:
            return run_bundle(bundle, dry_run=args.dry_run, preflight_only=args.preflight_only)
    elif args.source_bundle and args.expected_backend and args.results_root and args.output:
        prepare(args.source_bundle.resolve(), args.output.resolve(), args.expected_backend.resolve(), args.results_root.resolve())
    else:
        parser.error("provide --bundle, or --source-bundle/--expected-backend/--results-root/--output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
