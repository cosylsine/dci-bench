#!/usr/bin/env python
"""Run a paired harness pilot against one fixed serving endpoint.

The 60 fixed IDs (10 per task) had valid v2 outputs and failed in v3; a valid
output need not hit a relevant document. These diagnostic cases are not an
unbiased benchmark. The validation suite excludes both development suites.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Keep the original four-arm invocation as the default.  The calibrated arm
# is opt-in because its protocol is supplied by the serving checkout and is
# intentionally not part of the historical comparison.
DEFAULT_PROFILES = ("v3", "v3-temp1", "budgeted", "budgeted-temp1")
PROFILES = DEFAULT_PROFILES + ("budgeted-calibrated", "budgeted-stable", "budgeted-selective")
QUERY_SUITE_CHOICES = ("diagnostic", "holdout", "validation")
PROFILE_ORDER_CHOICES = ("as-listed", "alternating")
HOLDOUT_SEED = "dci-budget-round2-v1"
HOLDOUT_TAKE = 10
VALIDATION_SEED = "dci-selection-validation-v1"
DIAGNOSTIC_SELECTION = (
    "v2-valid-output/v3-failure diagnostic suite; original two IDs per task retained, "
    "eight added by ascending SHA256(task:query_id); not necessarily relevant hits"
)
HOLDOUT_SELECTION = (
    "calibration holdout: fixed ten IDs per task selected from metadata query records "
    "after excluding the diagnostic suite; not used for this round's calibration and "
    "may have appeared in historical full-corpus evaluation; not claimed absolutely unseen"
)
VALIDATION_SELECTION = (
    "selection validation: fixed ten IDs per task selected from query metadata after "
    "excluding both diagnostic and round-2 holdout IDs; no qrels or score filtering; "
    "frozen stable/selective contracts; historical full evaluations may include these IDs"
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_source_path(path):
    """Return an audit-friendly source path without requiring a shared mount."""

    try:
        # Preserve the checkout's logical path before resolving the shared
        # data symlink; the freeze records ``data/metadata/...`` consistently
        # on both local fixtures and cluster mounts.
        return str(path.relative_to(REPO))
    except ValueError:
        try:
            return str(path.resolve().relative_to(REPO.resolve()))
        except ValueError:
            return str(path)


def _load_query_records(task, metadata_root):
    """Load canonical host-side ``queries.jsonl`` without opening qrels."""

    task_root = Path(metadata_root) / task
    source = task_root / "queries.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"missing query metadata for {task}: {source}")
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = []
    seen = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"invalid query record at {source}:{index}")
        query_id = record.get("_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"invalid query id at {source}:{index}")
        if query_id in seen:
            raise ValueError(f"duplicate query id {task}/{query_id} in {source}")
        seen.add(query_id)
        ids.append(query_id)
    if not ids:
        raise ValueError(f"no query ids found in {source}")
    return {
        "path": source,
        "path_display": _relative_source_path(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(ids),
        "query_ids": ids,
    }


def _load_diagnostic_suite():
    path = REPO / "scripts/harness_pilot_queries.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def _holdout_hash(task, query_id):
    return hashlib.sha256(f"{HOLDOUT_SEED}:{task}:{query_id}".encode("utf-8")).hexdigest()


def _canonical_id_hashes(ids_by_task):
    return {task: digest(ids) for task, ids in ids_by_task.items()}


def _compute_holdout_freeze(metadata_root, diagnostic_suite):
    """Compute the expected deterministic six-task holdout from metadata."""

    tasks = list(diagnostic_suite)
    source = {}
    excluded = {}
    selected = {}
    for task in tasks:
        info = _load_query_records(task, metadata_root)
        source[task] = {
            "path": info["path_display"],
            "sha256": info["sha256"],
            "count": info["count"],
        }
        excluded_ids = list(diagnostic_suite[task])
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError(f"diagnostic suite has duplicate IDs for {task}")
        unknown = [query_id for query_id in excluded_ids if query_id not in set(info["query_ids"])]
        if unknown:
            raise ValueError(f"diagnostic suite IDs missing from {task} metadata: {unknown}")
        excluded[task] = excluded_ids
        candidates = [query_id for query_id in info["query_ids"] if query_id not in set(excluded_ids)]
        if len(candidates) < HOLDOUT_TAKE:
            raise ValueError(
                f"{task} has only {len(candidates)} non-diagnostic queries; "
                f"need {HOLDOUT_TAKE} for holdout"
            )
        selected[task] = sorted(candidates, key=lambda query_id: (_holdout_hash(task, query_id), query_id))[:HOLDOUT_TAKE]

    excluded_hashes = _canonical_id_hashes(excluded)
    selected_hashes = _canonical_id_hashes(selected)
    return {
        "schema_version": "dci-harness-pilot-holdout-v1",
        "suite": "holdout",
        "description": HOLDOUT_SELECTION,
        "selection_method": "sha256(seed + ':' + task + ':' + query_id)",
        "seed": HOLDOUT_SEED,
        "take": HOLDOUT_TAKE,
        "query_ids": selected,
        "selected_sha256": selected_hashes,
        "excluded_query_ids": excluded,
        "excluded_sha256": excluded_hashes,
        "source_queries": source,
    }


def _holdout_freeze(metadata_root, diagnostic_suite):
    """Load the required freeze and reject any metadata or freeze drift."""

    freeze_path = REPO / "scripts/harness_pilot_holdout_queries.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"required holdout freeze is missing: {freeze_path}")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = _compute_holdout_freeze(metadata_root, diagnostic_suite)
    # These fields form the complete selection and provenance contract.  A
    # changed source content hash is treated like any other freeze mismatch.
    for field in (
        "schema_version", "seed", "take", "selection_method", "query_ids",
        "selected_sha256", "excluded_query_ids", "excluded_sha256", "source_queries",
    ):
        if frozen.get(field) != expected.get(field):
            raise ValueError(f"holdout freeze mismatch in {freeze_path}: field {field} changed")
    return frozen


def _validate_metadata_ids(task, ids, metadata_root):
    """Validate selected IDs using query metadata only (never qrels/scores)."""

    info = _load_query_records(task, metadata_root)
    available = set(info["query_ids"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate selected query ID for {task}")
    unknown = [query_id for query_id in ids if query_id not in available]
    if unknown:
        raise ValueError(f"unknown query ID(s) for {task}: {unknown}")


def _compute_validation_freeze(metadata_root, diagnostic_suite, holdout):
    """Select new queries using metadata only, after both development suites."""

    if set(diagnostic_suite) != set(holdout["query_ids"]):
        raise ValueError("validation exclusion suites must cover the same tasks")
    source, excluded, selected = {}, {}, {}
    for task in diagnostic_suite:
        info = _load_query_records(task, metadata_root)
        source[task] = {"path": info["path_display"], "sha256": info["sha256"], "count": info["count"]}
        excluded[task] = sorted(set(diagnostic_suite[task]) | set(holdout["query_ids"][task]))
        available = set(info["query_ids"])
        if not set(excluded[task]) <= available:
            raise ValueError(f"validation exclusion IDs missing from {task} metadata")
        candidates = available - set(excluded[task])
        if len(candidates) < 10:
            raise ValueError(f"{task} needs ten queries outside diagnostic and holdout for validation")
        selected[task] = sorted(candidates, key=lambda query_id: (
            hashlib.sha256(f"{VALIDATION_SEED}:{task}:{query_id}".encode("utf-8")).hexdigest(), query_id,
        ))[:10]
    profile_plan = REPO / "scripts/harness_selection_round3_plan.json"
    profiles = json.loads(profile_plan.read_text(encoding="utf-8"))["profiles"]
    return {
        "schema_version": "dci-harness-pilot-validation-v1",
        "suite": "validation", "description": VALIDATION_SELECTION,
        "selection_method": "sha256(seed + ':' + task + ':' + query_id)",
        "seed": VALIDATION_SEED, "take": 10,
        "query_ids": selected, "selected_sha256": _canonical_id_hashes(selected),
        "excluded_query_ids": excluded, "excluded_sha256": _canonical_id_hashes(excluded),
        "source_queries": source,
        "excluded_suites": {name: {"path": _relative_source_path(path), "sha256": _file_sha256(path)}
                            for name, path in (
                                ("diagnostic", REPO / "scripts/harness_pilot_queries.json"),
                                ("holdout", REPO / "scripts/harness_pilot_holdout_queries.json"))},
        "profile_plan": {"path": _relative_source_path(profile_plan), "sha256": _file_sha256(profile_plan)},
        "profile_contract_sha256": {name: value["base_contract_sha256"] for name, value in profiles.items()},
    }


def _validation_freeze(metadata_root, diagnostic_suite):
    path = REPO / "scripts/harness_pilot_validation_queries.json"
    if not path.is_file():
        raise FileNotFoundError(f"required validation freeze is missing: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    holdout = _holdout_freeze(metadata_root, diagnostic_suite)
    expected = _compute_validation_freeze(metadata_root, diagnostic_suite, holdout)
    for field in expected.keys() | frozen.keys():
        if frozen.get(field) != expected.get(field):
            raise ValueError(f"validation freeze mismatch in {path}: field {field} changed")
    return frozen


def _describe_profiles(names):
    profiles = {}
    for profile in names:
        env = {**os.environ, "DCI_PROTOCOL_PROFILE": profile}
        env.pop("DCI_PROTOCOL_CONTRACT_SHA256", None)
        payload = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "--describe-profile"],
                                          cwd=REPO, env=env, text=True)
        profiles[profile] = json.loads(payload)
    return profiles


def _check_validation_profiles(profiles, frozen):
    expected = frozen["profile_contract_sha256"]
    actual = {name: profile["sha256"] for name, profile in profiles.items()}
    if actual != expected:
        raise ValueError("validation requires both frozen stable/selective profile contracts; hash or profile mismatch")


def _execution_order(tasks, profiles, profile_order):
    """Return flat command order and grouped order for plan/audit output."""

    if profile_order == "as-listed":
        sequence = [{"profile": profile, "task": task}
                    for profile in profiles for task in tasks]
        groups = [{"task": task, "profiles": list(profiles)} for task in tasks]
    else:
        groups = []
        sequence = []
        for task_index, task in enumerate(tasks, 1):
            ordered = list(profiles)
            if task_index % 2 == 1:
                ordered.reverse()
            groups.append({"task": task, "profiles": ordered})
            sequence.extend({"profile": profile, "task": task} for profile in ordered)
    return sequence, groups


def git_output(repo, *args):
    # Shared cluster mounts can be owned by a different UID from the container.
    # Match phase4_runner._git_audit: bind commands to this exact checkout.
    # Older cluster Git versions ignore command-line safe.directory overrides.
    repo = repo.resolve()
    git_dir = repo / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir:"):
            raise RuntimeError(f"Invalid gitdir marker: {git_dir}")
        git_dir = (repo / marker.removeprefix("gitdir:").strip()).resolve()
    command = ["git", f"--git-dir={git_dir}", f"--work-tree={repo}", *args]
    try:
        result = subprocess.run(command, cwd=repo, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Source snapshot Git command failed in {repo}: {shlex.join(command)}\n{exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Source snapshot Git command failed (exit {result.returncode}) in {repo}: "
            f"{shlex.join(command)}\n{detail}"
        )
    return result.stdout


def snapshot_sources(output):
    paths = set()
    for root, suffixes in (("dci_bench", {".py", ".json"}), ("scripts", {".py", ".sh", ".json"}),
                           ("schemas", {".json"})):
        paths.update(p for p in (REPO / root).rglob("*") if p.is_file() and p.suffix in suffixes
                     and "__pycache__" not in p.parts)
    paths.update(REPO / p for p in (
        "pi-dci/scripts/dci-run.ts", "pi-dci/scripts/dci-run.test.ts", "pi-dci/tsconfig.dci.json",
        "pi-dci/package.json", "pi-dci/package-lock.json", "tests/test_harness_pilot.py",
        "tests/test_harness_pilot_launcher.py", "tests/test_harness_pilot_warmup.py",
        "tests/test_harness_pilot_query_suites.py", "README.md", "docs/protocol.md",
        "dci_bench/protocol/stable-rg.conf", "pyproject.toml"))
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths) if p.is_file()}
    with tarfile.open(output / "sources.tar.gz", "x:gz") as archive:
        for name in hashes:
            archive.add(REPO / name, arcname=name, recursive=False)
    write_json(output / "source-hashes.json", hashes)
    git_audit = {}
    for label, repo in (("dci", REPO), ("pi", REPO / "pi-dci")):
        try:
            revision = git_output(repo, "rev-parse", "HEAD").decode("utf-8").strip()
            patch = git_output(repo, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--")
        except (OSError, RuntimeError, UnicodeError) as exc:
            git_audit[label] = {"available": False, "error": str(exc)}
            print(f"[pilot] warning: optional Git metadata unavailable for {label}: {exc}",
                  file=sys.stderr, flush=True)
            continue
        (output / f"{label}.patch").write_bytes(patch)
        (output / f"{label}.revision").write_text(revision + "\n")
        git_audit[label] = {"available": True, "revision": revision,
                            "patch_sha256": hashlib.sha256(patch).hexdigest()}
    write_json(output / "git-audit.json", git_audit)
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-profile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify source snapshot and validation freeze, if selected; no backend or GPU required")
    parser.add_argument("--backend-manifest", type=Path)
    parser.add_argument("--backend-artifact", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(DEFAULT_PROFILES))
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--query-limit", type=int, choices=range(1, 11), default=10,
                        help="use the first N fixed queries per task (default: 10)")
    parser.add_argument("--query-suite", choices=QUERY_SUITE_CHOICES, default="diagnostic",
                        help="query selection suite (default: diagnostic)")
    parser.add_argument("--profile-order", choices=PROFILE_ORDER_CHOICES, default="as-listed",
                        help="execution order; alternating groups profiles by task")
    parser.add_argument("--bridge-port", type=int, default=28131)
    parser.add_argument("--max-concurrency", type=int, default=os.environ.get("MAX_CONCURRENCY", "1"),
                        help="simultaneous samples within each task (default: MAX_CONCURRENCY or 1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.describe_profile:
        from dci_bench.protocol.contracts import CONTRACT_SHA256, protocol_contract
        print(json.dumps({"sha256": CONTRACT_SHA256, "contract": protocol_contract()}))
        return 0
    if args.preflight_only:
        try:
            if args.query_suite == "validation":
                diagnostic_suite, _ = _load_diagnostic_suite()
                frozen = _validation_freeze(REPO / "data/metadata", diagnostic_suite)
                _check_validation_profiles(_describe_profiles(args.profiles), frozen)
                print("[pilot] validation query freeze and stable/selective contract hashes passed", flush=True)
            with tempfile.TemporaryDirectory(prefix="dci-harness-preflight-") as temporary:
                hashes = snapshot_sources(Path(temporary))
                git_audit = json.loads((Path(temporary) / "git-audit.json").read_text())
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[pilot] source snapshot preflight failed: {exc}", file=sys.stderr, flush=True)
            return 2
        available = sum(item["available"] for item in git_audit.values())
        print(f"[pilot] source snapshot preflight passed: {len(hashes)} files; "
              f"optional Git metadata available for {available}/2 checkouts", flush=True)
        return 0
    if args.backend_manifest is None:
        parser.error("--backend-manifest is required")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique")
    manifest = json.loads(args.backend_manifest.read_text())
    max_num_seqs = manifest["backend"].get("max_num_seqs")
    if max_num_seqs is not None and max_num_seqs < args.max_concurrency:
        parser.error("backend max_num_seqs must be at least --max-concurrency")
    model_key = manifest["model"]["model_key"]
    diagnostic_suite, diagnostic_path = _load_diagnostic_suite()
    holdout = None
    validation = None
    if args.query_suite == "holdout":
        holdout = _holdout_freeze(REPO / "data/metadata", diagnostic_suite)
        suite = holdout["query_ids"]
        selection = HOLDOUT_SELECTION
        freeze_path = REPO / "scripts/harness_pilot_holdout_queries.json"
        suite_source = {"path": _relative_source_path(freeze_path),
                        "sha256": _file_sha256(freeze_path)}
    elif args.query_suite == "validation":
        validation = _validation_freeze(REPO / "data/metadata", diagnostic_suite)
        suite = validation["query_ids"]
        selection = VALIDATION_SELECTION
        freeze_path = REPO / "scripts/harness_pilot_validation_queries.json"
        suite_source = {"path": _relative_source_path(freeze_path), "sha256": _file_sha256(freeze_path)}
    else:
        suite = diagnostic_suite
        selection = DIAGNOSTIC_SELECTION
        suite_source = {"path": _relative_source_path(diagnostic_path),
                        "sha256": _file_sha256(diagnostic_path)}
    tasks = args.tasks or list(suite)
    if len(set(tasks)) != len(tasks) or any(t not in suite for t in tasks):
        parser.error("tasks must be unique names from the selected query suite")
    job_count = len(args.profiles) * len(tasks)
    if not 1024 <= args.bridge_port <= 65536 - job_count * args.max_concurrency:
        parser.error("bridge port range must fit within 1024..65535")
    # A closed bridge can leave TIME_WAIT sockets behind. Use a fresh port for
    # each subprocess instead of immediately rebinding the preceding task's ports.
    # Each task leases max_concurrency consecutive ports from its base.
    bridge_ports = {profile: {task: args.bridge_port + (arm * len(tasks) + index) * args.max_concurrency
                             for index, task in enumerate(tasks)}
                    for arm, profile in enumerate(args.profiles)}
    selected = {
        # Holdout freeze remains the complete ten-ID artifact; query-limit
        # only controls the planned smoke slice sent to run_task.py.
        t: suite[t][:args.query_limit]
        for t in tasks
    }
    # Diagnostic selection retains the historical host-side qrels validation.
    # Holdout and validation selection use query metadata, never qrels/scores.
    if args.query_suite in ("holdout", "validation"):
        for task, ids in selected.items():
            _validate_metadata_ids(task, ids, REPO / "data/metadata")
    else:
        from dci_bench.tasks.mteb_llm_retrieval import _select_query_ids
        for task, ids in selected.items():
            _select_query_ids(task, metadata_root=REPO / "data/metadata", query_ids=ids, all_queries=False)
    profiles = _describe_profiles(args.profiles)
    if validation is not None:
        _check_validation_profiles(profiles, validation)
    arm_manifests = {}
    for profile, contract in profiles.items():
        arm = deepcopy(manifest)
        arm["backend"]["generation_config"] = contract["contract"]["generation_config"]
        arm["manifest_sha256"] = digest({k: v for k, v in arm.items() if k != "manifest_sha256"})
        arm_manifests[profile] = arm
    execution_order, execution_groups = _execution_order(tasks, args.profiles, args.profile_order)
    plan = {
        "model_key": model_key,
        "selection": selection,
        "query_suite": args.query_suite,
        "query_suite_source": suite_source,
        "query_suite_source_sha256": suite_source["sha256"],
        "query_ids": selected,
        "profiles": profiles,
        "profile_order": args.profile_order,
        "execution_order": execution_order,
        "execution_groups": execution_groups,
        "max_concurrency": args.max_concurrency,
        "bridge_ports": bridge_ports,
        "base_backend_manifest_sha256": manifest["manifest_sha256"],
        "arm_backend_manifest_sha256": {p: m["manifest_sha256"] for p, m in arm_manifests.items()},
    }
    if holdout is not None:
        plan["holdout_freeze"] = holdout
    if validation is not None:
        plan["validation_freeze"] = validation
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_root or REPO / "results/harness-pilots" / f"{timestamp}-{os.getpid()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = snapshot_sources(output)
    write_json(output / "plan.json", plan)
    summaries = []
    data_by_task = {}
    manifest_paths = {}
    for profile, arm_manifest in arm_manifests.items():
        manifest_path = output / f"backend-{profile}.json"
        write_json(manifest_path, arm_manifest)
        manifest_paths[profile] = manifest_path
    if holdout is not None:
        write_json(output / "holdout-freeze.json", holdout)
    if validation is not None:
        write_json(output / "validation-freeze.json", validation)
    for item in execution_order:
        profile, task = item["profile"], item["task"]
        ids = selected[task]
        contract = profiles[profile]
        arm_manifest = arm_manifests[profile]
        manifest_path = manifest_paths[profile]
        changed = [name for name, sha in hashes.items()
                   if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != sha]
        if changed:
            raise RuntimeError(f"Source changed during pilot; refusing to mix implementations: {changed}")
        run_id = f"pilot-{profile}"
        command = [sys.executable, str(REPO / "scripts/run_task.py"), "--model-key", model_key,
                   "--task", task, "--query-ids", *ids, "--run-id", run_id,
                   "--results-root", str(output / "runs"), "--backend-manifest", str(manifest_path),
                   "--bridge-port", str(bridge_ports[profile][task]),
                   "--max-concurrency", str(args.max_concurrency)]
        for artifact in args.backend_artifact:
            command.extend(["--backend-artifact", str(artifact.resolve())])
        env = {**os.environ, "DCI_PROTOCOL_PROFILE": profile, "INSPECT_EVAL_LOG_MODEL_API": "1"}
        env.pop("DCI_PROTOCOL_CONTRACT_SHA256", None)
        # The pilot endpoint and the per-sample bridge are loopback services.
        for name in ("NO_PROXY", "no_proxy"):
            env[name] = ",".join(filter(None, [env.get(name), "127.0.0.1", "localhost"]))
        write_json(output / f"command-{profile}-{task}.json", {"argv": command, "profile": profile,
                   "base_contract_sha256": contract["sha256"], "log_model_api": True})
        print(f"[pilot] {profile} {task} {ids}", flush=True)
        with (output / f"eval-{profile}-{task}.log").open("x") as log:
            completed = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        run = output / "runs" / model_key / task / "runs" / run_id
        if completed.returncode not in (0, 1) or not (run / "_COMPLETE").is_file():
            raise RuntimeError(f"Pilot infrastructure failure ({completed.returncode}); inspect {output / f'eval-{profile}-{task}.log'}")
        result = json.loads((run / "task-result.json").read_text())
        if task in data_by_task and result["data"] != data_by_task[task]:
            raise RuntimeError(f"Data changed between profiles for {task}; paired comparison is invalid")
        data_by_task[task] = result["data"]
        row = {"profile": profile, "task": task, "run": str(run), "counts": result["counts"],
               "backend_manifest_sha256": arm_manifest["manifest_sha256"],
               "data_sha256": digest(result["data"]), "base_contract_sha256": contract["sha256"],
               "metrics": {group: {metric: result["metrics"][group][metric]["mean"]
                                   for metric in ("recall_at_10", "f1_at_10", "ndcg_at_10")}
                           for group in ("macro", "valid_only")},
               "total_model_tokens": result["usage"]["total_tokens"]["sum"]}
        summaries.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    write_json(output / "summary.json", {"selection": plan["selection"], "results": summaries})
    from dci_bench.results.protocol import write_json_atomic
    write_json_atomic(output / "_COMPLETE", {"schema_version": "dci-harness-pilot-complete-v1",
                      "rows": len(summaries), "summary_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()})
    print(f"[pilot] complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
