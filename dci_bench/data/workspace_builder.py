"""Build Phase 1 corpus workspaces from local MTEB(LLM) parquet data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dci_bench.data.loader import (
    assert_qrels_reference_known_ids,
    load_corpus,
    load_qrels,
    load_queries,
    parquet_files,
    resolve_task_dir,
    sha256_files,
    stable_json_sha256,
)
from dci_bench.data.registry import (
    DEFAULT_DATA_ROOT,
    TASKS,
    TaskSpec,
    get_task,
    validate_task_manifest,
)
from dci_bench.protocol.contracts import BENCHMARK_CONTRACT_VERSION, TOP_K, protocol_contract


WORKSPACE_FORMAT_VERSION = "dci-corpus-markdown-v2"
CORPUS_MAX_LINE_CHARS = 200


@dataclass(frozen=True)
class WorkspaceBuildResult:
    task: str
    workspace_dir: Path
    metadata_dir: Path
    manifest_path: Path


def doc_id_to_filename(doc_id: str) -> str:
    return f"{quote(doc_id, safe='-_.~')}.md"


def wrap_corpus_text(text: str, *, width: int = CORPUS_MAX_LINE_CHARS) -> str:
    """Insert deterministic soft line breaks without changing document IDs.

    Existing newlines and every source character are retained.  For ordinary
    prose, a break is inserted immediately after whitespace near ``width``;
    an unbroken token longer than ``width`` is split as a last resort.  Keeping
    corpus lines bounded prevents a single grep match from returning an entire
    multi-kilobyte document.
    """

    if width < 1:
        raise ValueError("corpus line width must be positive")

    wrapped_lines: list[str] = []
    for source_line in text.split("\n"):
        remaining = source_line
        while len(remaining) > width:
            split_at = max(remaining.rfind(" ", 0, width), remaining.rfind("\t", 0, width))
            if split_at < 0:
                split_at = width
            else:
                # Retain the original whitespace before inserting the newline.
                split_at += 1
            wrapped_lines.append(remaining[:split_at])
            remaining = remaining[split_at:]
        wrapped_lines.append(remaining)
    return "\n".join(wrapped_lines)


def render_doc_markdown(title: str, text: str) -> str:
    parts: list[str] = []
    if title.strip():
        parts.append(f"# {title.strip()}\n")
    parts.append(wrap_corpus_text(text))
    return "\n\n".join(part for part in parts if part)


def load_repo_metadata(task_dir: Path) -> dict[str, object] | None:
    path = task_dir / ".hfd" / "repo_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_pinned_revision(task: TaskSpec, task_dir: Path) -> str:
    metadata = load_repo_metadata(task_dir)
    if metadata is None:
        raise FileNotFoundError(
            f"Missing revision metadata for {task.name}: "
            f"{task_dir / '.hfd/repo_metadata.json'}"
        )
    dataset_id = metadata.get("id")
    if dataset_id != task.hf_dataset:
        raise ValueError(
            f"{task.name} local dataset id {dataset_id!r} does not match {task.hf_dataset!r}"
        )
    sha = metadata.get("sha")
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"{task_dir / '.hfd/repo_metadata.json'} has no non-empty string sha")
    if not sha.startswith(task.revision):
        raise ValueError(
            f"{task.name} local data sha {sha} does not match pinned revision prefix {task.revision}"
        )
    return sha


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_workspace(
    task: TaskSpec,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = Path("data/workspaces"),
    metadata_root: Path = Path("data/metadata"),
    overwrite: bool = False,
    skip_existing: bool = False,
    refresh_existing: bool = False,
    sample_queries: int = 5,
    sample_seed: int = 0,
) -> WorkspaceBuildResult:
    if sum((overwrite, skip_existing, refresh_existing)) > 1:
        raise ValueError("overwrite, skip_existing, and refresh_existing are mutually exclusive")
    task_dir = resolve_task_dir(task, data_root)
    resolved_revision = assert_pinned_revision(task, task_dir)
    workspace_dir = output_root / task.name
    corpus_dir = workspace_dir / "corpus"
    metadata_dir = metadata_root / task.name

    if workspace_dir.exists() or metadata_dir.exists():
        if skip_existing or refresh_existing:
            required = (
                workspace_dir / "corpus",
                metadata_dir / "manifest.json",
                metadata_dir / "queries.jsonl",
                metadata_dir / "qrels.json",
                metadata_dir / "doc_id_files.json",
            )
            missing = [path for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "cannot skip incomplete existing workspace; missing: "
                    + ", ".join(str(path) for path in missing)
                )
            manifest = json.loads((metadata_dir / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"existing manifest is not an object: {metadata_dir / 'manifest.json'}")
            existing_revision = validate_task_manifest(task, manifest)
            if existing_revision != resolved_revision:
                raise ValueError(
                    f"existing workspace resolved revision {existing_revision!r} does not match "
                    f"local source revision {resolved_revision!r} for {task.name}"
                )
            workspace_format = manifest.get("workspace_format")
            expected_format = {
                "version": WORKSPACE_FORMAT_VERSION,
                "max_line_chars": CORPUS_MAX_LINE_CHARS,
            }
            if workspace_format == expected_format:
                return WorkspaceBuildResult(
                    task=task.name,
                    workspace_dir=workspace_dir,
                    metadata_dir=metadata_dir,
                    manifest_path=metadata_dir / "manifest.json",
                )
            if skip_existing:
                raise ValueError(
                    f"existing workspace format for {task.name} is {workspace_format!r}; "
                    "rebuild with --refresh-existing or --overwrite"
                )
            # refresh_existing is permitted only after the complete existing
            # workspace and pinned source identity have both been validated.
            shutil.rmtree(workspace_dir)
            shutil.rmtree(metadata_dir)
        elif overwrite:
            shutil.rmtree(workspace_dir, ignore_errors=True)
            shutil.rmtree(metadata_dir, ignore_errors=True)
        if not overwrite:
            if not refresh_existing:
                raise FileExistsError(
                    f"{workspace_dir} or {metadata_dir} already exists; pass overwrite=True to rebuild"
                )

    docs = load_corpus(task_dir)
    queries = load_queries(task_dir)
    qrels = load_qrels(task_dir)
    assert_qrels_reference_known_ids(
        qrels,
        query_ids={query["_id"] for query in queries},
        doc_ids={doc["_id"] for doc in docs},
    )

    corpus_dir.mkdir(parents=True, exist_ok=False)
    metadata_dir.mkdir(parents=True, exist_ok=False)

    doc_id_files: dict[str, str] = {}
    filenames: set[str] = set()
    for doc in sorted(docs, key=lambda row: row["_id"]):
        filename = doc_id_to_filename(doc["_id"])
        if filename in filenames:
            raise ValueError(f"Filename collision after percent-encoding: {filename}")
        filenames.add(filename)
        doc_id_files[doc["_id"]] = filename
        (corpus_dir / filename).write_text(
            render_doc_markdown(doc["title"], doc["text"]),
            encoding="utf-8",
        )

    readme = (
        "# DCI-Bench Corpus Workspace\n\n"
        f"Task: {task.name}\n\n"
        f"Dataset: {task.hf_dataset}\n\n"
        f"Revision: {task.revision}\n\n"
        f"Documents: {len(docs)}\n\n"
        "Each corpus file is named as the URL percent-encoded original document id "
        "with a `.md` suffix.\n"
    )
    (workspace_dir / "README.md").write_text(readme, encoding="utf-8")

    rng = random.Random(sample_seed)
    sampled_queries = rng.sample(queries, k=min(sample_queries, len(queries))) if queries else []

    qrels_canonical = {
        query_id: dict(sorted(doc_scores.items()))
        for query_id, doc_scores in sorted(qrels.items())
    }
    manifest = {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "task": task.name,
        "dataset": task.hf_dataset,
        "revision": task.revision,
        "resolved_revision": resolved_revision,
        "source_dir": str(task_dir),
        "num_queries": len(queries),
        "num_docs": len(docs),
        "num_qrels": sum(len(items) for items in qrels.values()),
        "top_k": TOP_K,
        "corpus_hash": stable_json_sha256(
            [{"_id": doc["_id"], "title": doc["title"], "text": doc["text"]} for doc in sorted(docs, key=lambda row: row["_id"])]
        ),
        "queries_hash": stable_json_sha256(sorted(queries, key=lambda row: row["_id"])),
        "qrels_hash": stable_json_sha256(qrels_canonical),
        "source_parquet_hash": sha256_files(
            parquet_files(task_dir, "corpus")
            + parquet_files(task_dir, "queries")
            + parquet_files(task_dir, "data")
        ),
        "workspace_dir": str(workspace_dir),
        "metadata_dir": str(metadata_dir),
        "workspace_format": {
            "version": WORKSPACE_FORMAT_VERSION,
            "max_line_chars": CORPUS_MAX_LINE_CHARS,
        },
        "task_instruction": task.task_instruction,
        "protocol": protocol_contract(
            task_name=task.name,
            task_instruction=task.task_instruction,
        ),
    }

    write_jsonl(metadata_dir / "queries.jsonl", sorted(queries, key=lambda row: row["_id"]))
    write_json(metadata_dir / "qrels.json", qrels_canonical)
    write_json(metadata_dir / "doc_id_files.json", dict(sorted(doc_id_files.items())))
    write_json(metadata_dir / "query_samples.json", sampled_queries)
    manifest_path = metadata_dir / "manifest.json"
    write_json(manifest_path, manifest)

    return WorkspaceBuildResult(
        task=task.name,
        workspace_dir=workspace_dir,
        metadata_dir=metadata_dir,
        manifest_path=manifest_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", choices=TASKS)
    selection.add_argument(
        "--all-tasks",
        action="store_true",
        help="build all six registry tasks in canonical order",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data/workspaces"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/metadata"))
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true")
    existing.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse only complete workspaces whose manifest matches the pinned source",
    )
    existing.add_argument(
        "--refresh-existing",
        action="store_true",
        help="reuse current-format workspaces and rebuild complete pinned older formats",
    )
    parser.add_argument("--sample-queries", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = tuple(TASKS) if args.all_tasks else (args.task or "LLMPublicHealthQA",)
    results = [
        build_workspace(
            get_task(task_name),
            data_root=args.data_root,
            output_root=args.output_root,
            metadata_root=args.metadata_root,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            refresh_existing=args.refresh_existing,
            sample_queries=args.sample_queries,
            sample_seed=args.sample_seed,
        )
        for task_name in task_names
    ]
    payloads = [
        {
            "task": result.task,
            "workspace_dir": str(result.workspace_dir),
            "metadata_dir": str(result.metadata_dir),
            "manifest_path": str(result.manifest_path),
        }
        for result in results
    ]
    payload: object = payloads[0] if len(payloads) == 1 else payloads
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
