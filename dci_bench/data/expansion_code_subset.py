"""Host-only ARB subset whose positive file views are all untruncated.

Candidate snapshots are inherited unchanged from the frozen file-view release.
This selection does not establish full-source, line, or function equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from dci_bench.data.expansion_code import load_reference, _sha, _write
from dci_bench.protocol.code import CODE_LABELS_VERSION
from dci_bench.scoring.code import load_json

ARTIFACTS = {"queries.json", "labels.json", "selection_audit.json", "manifest.json", "build_source.py"}


def build_subset(source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or os.path.lexists(output):
        raise FileExistsError("Subset must have a new output directory outside its frozen source")
    complete = load_json((source / "_COMPLETE.json").read_text())
    if _sha(source / "manifest.json") != complete["manifest_sha256"]:
        raise ValueError("Source completion hash mismatch")
    manifest = load_json((source / "manifest.json").read_text())
    for name in ("host/labels.json", "queries.json"):
        if _sha(source / name) != manifest["artifact_hashes"][name]:
            raise ValueError("Source query/label hash mismatch")
    labels = load_json((source / "host/labels.json").read_text())["queries"]
    queries = load_json((source / "queries.json").read_text())
    if len(queries) != len(labels) or {q["query_id"] for q in queries} != set(labels):
        raise ValueError("Source query and label IDs differ")
    snapshots, checked, selected, exclusions = {}, set(), {}, []
    for qid, label in labels.items():
        key = label["snapshot_id"]
        if key not in snapshots:
            path = source / "snapshots" / key / "manifest.json"
            if _sha(path) != manifest["snapshot_manifest_sha256"][key]:
                raise ValueError("Source snapshot manifest hash mismatch")
            snapshots[key] = load_json(path.read_text())
        snapshot = snapshots[key]
        reference = load_reference(label, source / "snapshots")
        truncated = []
        for path in reference["gold_files"]:
            if (key, path) not in checked:
                content = (source / "snapshots" / key / "corpus" / path).read_bytes()
                if hashlib.sha256(content).hexdigest() != snapshot["files"][path]["sha256"]:
                    raise ValueError("Positive corpus bytes differ from frozen source")
                text = content.decode("utf-8")
                view = snapshot["upstream_file_view"][path]
                if text.endswith("...[truncated]") != view["truncation_marker"] or len(text) != view["characters"]:
                    raise ValueError("Positive file-view truncation metadata differs")
                checked.add((key, path))
            if snapshot["upstream_file_view"][path]["truncation_marker"]:
                truncated.append(path)
        if truncated:
            exclusions.append({"query_id": qid, "reason": "at_least_one_positive_file_view_truncated",
                               "truncated_gold_files": truncated})
        else:
            selected[qid] = reference
    if not selected:
        raise ValueError("No queries have untruncated positive file views")
    kept_queries = [q for q in queries if q["query_id"] in selected]
    counts = Counter(q["task"] for q in kept_queries)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        _write(stage / "queries.json", kept_queries)
        _write(stage / "labels.json", {"schema_version": CODE_LABELS_VERSION, "queries": selected})
        _write(stage / "selection_audit.json", {"source_queries": len(labels), "retained_queries": len(selected),
                                               "excluded_queries": exclusions, "policy": "exclude whole query; never trim gold files"})
        result = {"schema_version": "dci-arb-gold-untruncated-v1", "source_directory": str(source),
                  "source_completion_sha256": _sha(source / "_COMPLETE.json"), "samples": len(selected),
                  "tasks": dict(sorted(counts.items())),
                  "gold_file_occurrences": sum(len(r["gold_files"]) for r in selected.values()),
                  "snapshot_ids": sorted({q["snapshot_id"] for q in kept_queries}),
                  "candidate_policy": "inherit every file in the original query snapshot; no corpus pruning",
                  "corpus_path_template": str(source / "snapshots/{snapshot_id}/corpus"),
                  "source_complete": False, "line_function_annotations": "unavailable",
                  "host_only": sorted(ARTIFACTS), "model_calls": 0}
        _write(stage / "manifest.json", result)
        (stage / "build_source.py").write_bytes(Path(__file__).read_bytes())
        _write(stage / "_COMPLETE.json", {"artifact_sha256": {name: _sha(stage / name) for name in sorted(ARTIFACTS)}})
        if os.path.lexists(output):
            raise FileExistsError("Subset output appeared during build")
        os.rename(stage, output)
        return result
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_subset(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
