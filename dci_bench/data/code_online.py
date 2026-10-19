"""Host-only online code sample loading and snapshot preflight."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Sequence

from dci_bench.protocol.code import CODE_LABELS_VERSION
from dci_bench.protocol.code_runtime import code_runtime_contract
from dci_bench.scoring.code import file_metadata, load_json, snapshot_identity, validate_reference


@dataclass(frozen=True)
class CodeSample:
    query_id: str
    query: str
    workspace: Path
    reference: dict[str, Any]
    export_root: Path | None = None

    @property
    def directory_name(self) -> str:
        # Source IDs are opaque, including slashes; never use them as paths.
        return hashlib.sha256(self.query_id.encode("utf-8")).hexdigest()


def validate_workspace(workspace: Path, files: dict[str, Any]) -> None:
    """Check the exact nested corpus, hashes, token costs and LF line counts."""
    corpus = workspace / "corpus"
    if corpus.is_symlink() or not corpus.is_dir():
        raise ValueError(f"missing or linked code corpus: {corpus}")
    actual = set()
    expected_directories = {parent.as_posix() for name in files for parent in Path(name).parents if parent != Path(".")}
    limit = code_runtime_contract()["max_corpus_document_bytes"]
    for directory, children, names in os.walk(corpus, followlinks=False):
        for child in children:
            path = Path(directory) / child
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError(f"non-directory or linked candidate directory: {path}")
            if path.relative_to(corpus).as_posix() not in expected_directories:
                raise ValueError(f"unexpected candidate directory: {path}")
        for name in names:
            path = Path(directory) / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > limit:
                raise ValueError(f"candidate must be a regular unlinked file within byte limit: {path}")
            relative = path.relative_to(corpus).as_posix()
            if relative not in files:
                raise ValueError(f"unexpected candidate file: {relative}")
            expected = files[relative]
            observed = file_metadata(path.read_bytes().decode("utf-8"))
            if any(observed[key] != value for key, value in expected.items()):
                raise ValueError(f"candidate bytes/metadata changed: {relative}")
            actual.add(relative)
    if actual != set(files):
        raise ValueError("candidate manifest has missing files")


def load_code_samples(
    *, code_root: Path | None = None, queries_path: Path | None = None,
    labels_path: Path | None = None, query_ids: Sequence[str] | None = None,
    all_queries: bool = False, task_name: str | None = None,
) -> list[CodeSample]:
    if (query_ids is None) == (not all_queries):
        raise ValueError("select exactly one of query_ids or all_queries")
    if code_root is not None:
        if queries_path is not None or labels_path is not None:
            raise ValueError("code_root cannot be combined with queries/labels")
        from dci_bench.data.expansion_code import load_reference, verify_arb
        code_root = Path(code_root).resolve()
        # This checks completion, snapshot closure and bytes before any model call.
        verify_arb(code_root)
        rows = load_json((code_root / "queries.json").read_text(encoding="utf-8"))
        references = load_json((code_root / "host/labels.json").read_text(encoding="utf-8"))["queries"]
        workspaces = {row["query_id"]: code_root / "snapshots" / row["snapshot_id"] for row in rows}
    else:
        if queries_path is None or labels_path is None:
            raise ValueError("provide code_root or both queries_path and labels_path")
        rows = load_json(Path(queries_path).read_text(encoding="utf-8"))
        labels = load_json(Path(labels_path).read_text(encoding="utf-8"))
        if not isinstance(labels, dict) or set(labels) != {"schema_version", "queries"} or labels["schema_version"] != CODE_LABELS_VERSION:
            raise ValueError("online code labels must use dci-code-labels-v1")
        references = labels["queries"]
        if not isinstance(rows, list) or not isinstance(references, dict):
            raise ValueError("queries must be a list and label queries an object")
        workspaces = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"query_id", "query", "workspace"}:
                raise ValueError("each query requires exactly query_id, query and workspace")
            if not isinstance(row["workspace"], str) or not row["workspace"]:
                raise ValueError("workspace must be a nonempty path")
            if not isinstance(row["query_id"], str):
                raise ValueError("query_id must be a string")
            workspaces[row["query_id"]] = (Path(queries_path).resolve().parent / row["workspace"]).resolve()
    by_id = {}
    for row in rows:
        qid = row["query_id"]
        if not isinstance(qid, str) or not qid.strip() or qid in by_id:
            raise ValueError("query IDs must be nonempty and unique")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"empty query: {qid}")
        by_id[qid] = row
    if set(by_id) != set(references):
        raise ValueError("query/label identity sets differ")
    eligible = [qid for qid in by_id if task_name is None or references[qid]["task"] == task_name]
    selected = eligible if all_queries else list(query_ids or [])
    if not selected or len(set(selected)) != len(selected) or any(qid not in eligible for qid in selected):
        raise ValueError("selection is empty, duplicated, unknown or outside the requested task")
    samples, checked = [], set()
    for qid in selected:
        # Expand only selected compact references; a one-query preflight must
        # not retain a separate full file manifest for every query in ARB.
        reference = (load_reference(references[qid], code_root / "snapshots") if code_root is not None
                     else validate_reference(references[qid]))
        workspace = workspaces[qid]
        identity = (str(workspace), snapshot_identity(reference)["corpus_sha256"])
        if identity not in checked:
            validate_workspace(workspace, reference["files"])
            checked.add(identity)
        samples.append(CodeSample(qid, by_id[qid]["query"], workspace, reference, code_root))
    return samples
