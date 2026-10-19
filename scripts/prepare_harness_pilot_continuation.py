#!/usr/bin/env python3
"""Prepare an isolated continuation of whole, unstarted pilot task pairs.

Preparation and --preflight-only never start a model. The generated launch.sh
runs the archived launcher with a checked --tasks subset and fresh outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile


BENCH_PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python"
SERVING_PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38/bin/python"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def relative_file(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"unsafe relative artifact path: {name}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact escapes its root: {name}")
    return result


def verify_artifact(root, reference):
    path = relative_file(root, reference["path"])
    if file_hash(path) != reference["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")


def verify_group(run, plan, task, profile):
    marker = read_json(run / "_COMPLETE")
    result = read_json(run / "task-result.json")
    manifest = read_json(run / "run-manifest.json")
    if marker["task_result_sha256"] != json_hash(result):
        raise ValueError(f"task result hash mismatch: {run}")
    ids = plan["query_ids"][task]
    if (result["query_ids"] != ids or manifest["query_ids"] != ids
            or result["task"] != task or manifest["task"] != task
            or result["run_id"] != f"pilot-{profile}"
            or manifest["run_id"] != result["run_id"]
            or marker["run_id"] != result["run_id"]
            or marker["run_fingerprint"] != manifest["run_fingerprint"]
            or manifest["data"] != result["data"]
            or result["counts"]["completed"] != len(ids)):
        raise ValueError(f"completed group identity/count mismatch: {run}")
    if [sample["query_id"] for sample in result["samples"]] != ids:
        raise ValueError(f"sample index mismatch: {run}")
    for entry in result["samples"]:
        sample = read_json(relative_file(run, entry["result"]["path"]))
        if (json_hash(sample) != entry["result"]["sha256"]
                or sample["query_id"] != entry["query_id"]):
            raise ValueError(f"sample hash/identity mismatch: {run}/{entry['query_id']}")
        for reference in sample.get("artifacts", {}).values():
            verify_artifact(run, reference)
        verify_artifact(run, entry["trace"])
    integrity = result.get("integrity", {})
    for reference in [integrity.get("backend_manifest"), *integrity.get("backend_diagnostics", [])]:
        if reference:
            verify_artifact(run, reference)
    return {"task": task, "profile": profile, "run": str(run), "samples": len(ids),
            "task_result_sha256": marker["task_result_sha256"],
            "source_tree_fingerprint": result["code"]["source_tree_fingerprint"],
            "data": result["data"]}


def execution_order(tasks, profiles, mode):
    if mode == "as-listed":
        return [{"task": task, "profile": p} for p in profiles for task in tasks]
    return [{"task": task, "profile": p} for i, task in enumerate(tasks, 1)
            for p in (profiles[::-1] if i % 2 else profiles)]


def remaining_plan(pilot):
    evaluation = pilot / "evaluation"
    plan = read_json(evaluation / "plan.json")
    if plan["query_suite"] != "validation":
        raise ValueError("continuation currently requires a frozen validation suite")
    profiles = list(plan["execution_groups"][0]["profiles"])
    if plan["profile_order"] == "alternating":
        profiles.reverse()
    completed, remaining = [], []
    for group in plan["execution_groups"]:
        task = group["task"]
        runs = [evaluation / "runs" / plan["model_key"] / task / "runs" / f"pilot-{p}"
                for p in profiles]
        present = [(run / "_COMPLETE").is_file() for run in runs]
        if all(present):
            pair = [verify_group(run, plan, task, p) for run, p in zip(runs, profiles)]
            if any(item["data"] != pair[0]["data"] for item in pair):
                raise ValueError(f"paired data mismatch: {task}")
            completed.extend(pair)
        elif any(present) or any(run.exists() for run in runs):
            raise ValueError(f"partially started task needs explicit recovery, refusing to rerun it: {task}")
        else:
            remaining.append(task)
    if not remaining:
        raise ValueError("no whole unstarted task pairs remain")
    expected = [item for item in plan["execution_order"] if item["task"] in remaining]
    # An odd completed prefix flips alternating order. Preserve the saved plan.
    if execution_order(remaining, profiles, plan["profile_order"]) != expected:
        profiles.reverse()
    if execution_order(remaining, profiles, plan["profile_order"]) != expected:
        raise ValueError("task subset cannot preserve the original profile order")
    return plan, profiles, remaining, completed, expected


def restore_sources(evaluation, runtime):
    hashes = read_json(evaluation / "source-hashes.json")
    with tarfile.open(evaluation / "sources.tar.gz") as archive:
        members = archive.getmembers()
        if len(members) != len(hashes) or {m.name for m in members} != set(hashes):
            raise ValueError("source archive entries do not match source-hashes.json")
        for member in members:
            target = relative_file(runtime, member.name)
            if not member.isfile():
                raise ValueError(f"source archive must contain regular files: {member.name}")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != hashes[member.name]:
                raise ValueError(f"source archive hash mismatch: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            target.chmod(member.mode & 0o777)
    return hashes


def inventory(runtime, shared_workspaces):
    files, links = {}, {}
    for base, directories, names in os.walk(runtime, followlinks=False):
        for name in sorted(directories + names):
            path = Path(base) / name
            relative = path.relative_to(runtime).as_posix()
            if path.is_symlink():
                if relative == "data/workspaces":
                    if path.resolve() != shared_workspaces.resolve():
                        raise ValueError("shared workspace target changed")
                elif not path.resolve().is_relative_to(runtime.resolve()):
                    raise ValueError(f"runtime symlink escapes frozen copy: {relative}")
                links[relative] = os.readlink(path)
            elif path.is_file():
                files[relative] = file_hash(path)
    return {"files": files, "symlinks": links}


def check_plan(old, new, remaining, expected_order):
    if new["query_ids"] != {task: old["query_ids"][task] for task in remaining}:
        raise ValueError("continuation changed frozen query IDs")
    for field in ("profiles", "validation_freeze", "query_suite", "model_key", "max_concurrency"):
        if new[field] != old[field]:
            raise ValueError(f"continuation plan changed {field}")
    if new["execution_order"] != expected_order:
        raise ValueError("continuation changed paired execution order")


def prepare(pilot, output, repository):
    plan, profiles, tasks, completed, expected = remaining_plan(pilot)
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    print("[continuation] copying Pi runtime and installed dependencies (no install)", flush=True)
    shutil.copytree(repository / "pi-dci", runtime / "pi-dci", symlinks=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".cache"))
    hashes = restore_sources(pilot / "evaluation", runtime)
    (runtime / "data").mkdir()
    shutil.copytree(repository / "data/metadata", runtime / "data/metadata")
    workspaces = (repository / "data/workspaces").resolve()
    (runtime / "data/workspaces").symlink_to(workspaces, target_is_directory=True)
    copied = output / "continue.py"
    shutil.copyfile(Path(__file__), copied)
    backend = read_json(pilot / "backend.json")
    launch = output / "launch.sh"
    launch.write_text(
        '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
        'bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'mkdir -p "$bundle/logs"\n'
        'driver_log="$bundle/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"\n'
        'exec > >(tee "$driver_log") 2>&1\n'
        'echo "[continuation] driver log: $driver_log"\n'
        f'exec "${{BENCH_PYTHON:-{BENCH_PYTHON}}}" "$bundle/continue.py" --bundle "$bundle" "$@"\n'
    )
    launch.chmod(0o755)
    manifest = {
        "schema_version": "dci-pilot-continuation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pilot": str(pilot), "repository": str(repository),
        "plan_sha256": file_hash(pilot / "evaluation/plan.json"),
        "backend_sha256": file_hash(pilot / "backend.json"),
        "source_archive_sha256": file_hash(pilot / "evaluation/sources.tar.gz"),
        "source_hashes": hashes, "profiles": profiles, "remaining_tasks": tasks,
        "remaining_execution_order": expected, "completed_groups": completed,
        "planned_model_evaluations": sum(len(plan["query_ids"][t]) for t in tasks) * len(profiles),
        "shared_workspaces": str(workspaces), "runtime_inventory": inventory(runtime, workspaces),
        "runner_sha256": file_hash(copied), "launch_sha256": file_hash(launch),
        "model_path": backend["model"]["source_path"],
        "served_model_name": backend["model"]["served_model_name"],
        "tensor_parallel_size": backend["backend"]["tensor_parallel_size"],
        "max_concurrency": plan["max_concurrency"],
        "max_num_seqs": backend["backend"]["max_num_seqs"],
        "gpu_memory_utilization": backend["backend"]["gpu_memory_utilization"],
        "vllm_extra_args": backend["backend"]["runtime"]["launch_environment"]["VLLM_EXTRA_ARGS"],
    }
    write_json(output / "continuation.json", manifest)
    # Only this new runtime copy is made read-only. Never chmod shared inputs.
    for base, directories, names in os.walk(runtime, topdown=False, followlinks=False):
        for name in names:
            path = Path(base) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        for name in directories:
            path = Path(base) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
    runtime.chmod(0o555)
    print(f"[continuation] preserved {sum(x['samples'] for x in completed)} completed evaluations; "
          f"remaining {manifest['planned_model_evaluations']}", flush=True)
    print(f"bash {shlex.quote(str(launch))}", flush=True)


def run_bundle(bundle, *, preflight_only=False, dry_run=False):
    manifest = read_json(bundle / "continuation.json")
    runtime = bundle / "runtime"
    print("[continuation] verifying frozen runtime and preserved results", flush=True)
    if (file_hash(bundle / "continue.py") != manifest["runner_sha256"]
            or file_hash(bundle / "launch.sh") != manifest["launch_sha256"]):
        raise ValueError("continuation runner changed after preparation")
    if inventory(runtime, Path(manifest["shared_workspaces"])) != manifest["runtime_inventory"]:
        raise ValueError("frozen runtime changed after preparation")
    pilot = Path(manifest["source_pilot"])
    if (file_hash(pilot / "evaluation/plan.json") != manifest["plan_sha256"]
            or file_hash(pilot / "backend.json") != manifest["backend_sha256"]
            or file_hash(pilot / "evaluation/sources.tar.gz") != manifest["source_archive_sha256"]):
        raise ValueError("original pilot provenance changed")
    old, profiles, tasks, completed, expected = remaining_plan(pilot)
    if (profiles != manifest["profiles"] or tasks != manifest["remaining_tasks"]
            or completed != manifest["completed_groups"] or expected != manifest["remaining_execution_order"]):
        raise ValueError("original pilot completion state changed; prepare a new continuation")
    env = os.environ.copy()
    env.update({
        "BENCH_PYTHON": env.get("BENCH_PYTHON", BENCH_PYTHON),
        "SERVING_PYTHON": env.get("SERVING_PYTHON", SERVING_PYTHON),
        "PYTHONPATH": str(runtime), "PYTHONDONTWRITEBYTECODE": "1", "TSX_DISABLE_CACHE": "1",
        "DCI_EVALUATION_TRACK": "document", "DCI_PROTOCOL_PROFILE": "v3",
        "MODEL_PATH": manifest["model_path"],
        "SERVED_MODEL_NAME": manifest["served_model_name"],
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        "TP_SIZE": str(manifest["tensor_parallel_size"]),
        "MAX_CONCURRENCY": str(manifest["max_concurrency"]),
        "VLLM_MAX_NUM_SEQS": str(manifest["max_num_seqs"]),
        "VLLM_GPU_MEMORY_UTILIZATION": str(manifest["gpu_memory_utilization"]),
        "VLLM_EXTRA_ARGS": manifest["vllm_extra_args"],
    })
    for key in ("NODE_PATH", "NODE_OPTIONS", "DCI_CODE_RUNTIME_CONTRACT", "DCI_PROTOCOL_CONTRACT_SHA256",
                "KV_CACHE_DTYPE", "VLLM_NVFP4_GEMM_BACKEND"):
        env.pop(key, None)
    env["PATH"] = str(Path(env["BENCH_PYTHON"]).parent) + os.pathsep + env.get("PATH", "")
    args = ["--profiles", *profiles, "--query-suite", old["query_suite"],
            "--profile-order", old["profile_order"], "--max-concurrency", str(old["max_concurrency"]),
            "--tasks", *tasks]
    python = env["BENCH_PYTHON"]
    subprocess.run([python, "-c", (
        "from pathlib import Path; import dci_bench; from dci_bench.agents import pi_agent; "
        "root=Path.cwd().resolve(); "
        "assert Path(dci_bench.__file__).resolve().is_relative_to(root); "
        "assert pi_agent.repo_root()==root; "
        "print('[continuation] Python imports:', dci_bench.__file__)"
    )], cwd=runtime, env=env, check=True)
    subprocess.run(["node", "--input-type=module", "-e", (
        "import {fileURLToPath} from 'node:url'; import {realpathSync} from 'node:fs'; "
        "const root=realpathSync(process.cwd())+'/'; "
        "for (const spec of ['@earendil-works/pi-agent-core','@earendil-works/pi-ai',"
        "'@earendil-works/pi-telemetry','typebox','tsx']) { "
        "const path=realpathSync(fileURLToPath(import.meta.resolve(spec))); "
        "if (!path.startsWith(root)) throw new Error('Module escapes frozen runtime: '+spec); "
        "console.log('[continuation] Node import:', spec, path); }"
    )], cwd=runtime / "pi-dci", env=env, check=True)
    command = [python, "scripts/run_harness_pilot.py"]
    subprocess.run([*command, "--preflight-only", *args], cwd=runtime, env=env, check=True)
    planned = subprocess.check_output([*command, "--backend-manifest", str(pilot / "backend.json"),
                                      "--dry-run", *args], cwd=runtime, env=env, text=True)
    check_plan(old, json.loads(planned), tasks, expected)
    print(f"[continuation] frozen queries/profiles/order verified: {len(tasks)} tasks, "
          f"{manifest['planned_model_evaluations']} evaluations", flush=True)
    if preflight_only or dry_run:
        subprocess.run(["bash", "scripts/run_harness_pilot.sh", *args], cwd=runtime,
                       env={**env, "DRY_RUN": "1"}, check=True)
    if preflight_only:
        subprocess.run([python, "scripts/run_all.py", "--preflight-only"], cwd=runtime, env=env, check=True)
        for profile in profiles:
            subprocess.run(["npm", "exec", "--", "tsx", "scripts/dci-run.ts", "--preflight-only",
                            "--workspace", str(runtime / "data/workspaces/LLMPublicHealthQA")],
                           cwd=runtime / "pi-dci", env={**env, "DCI_PROTOCOL_PROFILE": profile}, check=True)
        print("[continuation] CPU/data/two-profile sandbox preflight passed; no GPU started", flush=True)
        return
    if dry_run:
        print(planned, end="")
        return
    env.pop("DRY_RUN", None)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(env.get("PILOT_ROOT", str(pilot.parent / f"{timestamp}-validation-remaining-{os.getpid()}"))).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse pilot output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env["PILOT_ROOT"] = str(output)
    write_json(output.with_name(output.name + ".continuation.json"), {
        "bundle": str(bundle), "manifest_sha256": file_hash(bundle / "continuation.json"),
        "source_pilot": str(pilot), "preserved_completed_groups": completed,
        "remaining_tasks": tasks, "planned_model_evaluations": manifest["planned_model_evaluations"],
    })
    os.chdir(runtime)
    print(f"[continuation] starting archived launcher from {runtime}; output={output}", flush=True)
    os.execvpe("bash", ["bash", "scripts/run_harness_pilot.sh", *args], env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-pilot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bundle", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.bundle:
        if args.from_pilot or args.output:
            parser.error("--bundle cannot be combined with preparation options")
        run_bundle(args.bundle.resolve(), preflight_only=args.preflight_only, dry_run=args.dry_run)
    else:
        if not args.from_pilot or not args.output or args.preflight_only or args.dry_run:
            parser.error("preparation requires --from-pilot and --output")
        prepare(args.from_pilot.resolve(), args.output.resolve(), args.repository.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"[continuation] refused: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
