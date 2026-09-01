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
from dci_bench.data.registry import DEFAULT_DATA_ROOT, TASKS, TaskSpec, get_task
from dci_bench.protocol.contracts import BENCHMARK_CONTRACT_VERSION, TOP_K, protocol_contract


@dataclass(frozen=True)
class WorkspaceBuildResult:
    task: str
    workspace_dir: Path
    metadata_dir: Path
    manifest_path: Path


def doc_id_to_filename(doc_id: str) -> str:
    return f"{quote(doc_id, safe='-_.~')}.md"


def render_doc_markdown(title: str, text: str) -> str:
    parts: list[str] = []
    if title.strip():
        parts.append(f"# {title.strip()}\n")
    parts.append(text)
    return "\n\n".join(part for part in parts if part)


def load_repo_metadata(task_dir: Path) -> dict[str, object] | None:
    path = task_dir / ".hfd" / "repo_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_pinned_revision(task: TaskSpec, task_dir: Path) -> str | None:
    metadata = load_repo_metadata(task_dir)
    if metadata is None:
        return None
    sha = metadata.get("sha")
    if not isinstance(sha, str):
        raise ValueError(f"{task_dir / '.hfd/repo_metadata.json'} has no string sha")
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
    sample_queries: int = 5,
    sample_seed: int = 0,
) -> WorkspaceBuildResult:
    task_dir = resolve_task_dir(task, data_root)
    resolved_revision = assert_pinned_revision(task, task_dir)
    workspace_dir = output_root / task.name
    corpus_dir = workspace_dir / "corpus"
    metadata_dir = metadata_root / task.name

    if workspace_dir.exists() or metadata_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{workspace_dir} or {metadata_dir} already exists; pass overwrite=True to rebuild"
            )
        shutil.rmtree(workspace_dir, ignore_errors=True)
        shutil.rmtree(metadata_dir, ignore_errors=True)

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
        "protocol": protocol_contract(),
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
    parser.add_argument("--task", default="LLMPublicHealthQA", choices=sorted(TASKS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data/workspaces"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/metadata"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-queries", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_workspace(
        get_task(args.task),
        data_root=args.data_root,
        output_root=args.output_root,
        metadata_root=args.metadata_root,
        overwrite=args.overwrite,
        sample_queries=args.sample_queries,
        sample_seed=args.sample_seed,
    )
    print(
        json.dumps(
            {
                "task": result.task,
                "workspace_dir": str(result.workspace_dir),
                "metadata_dir": str(result.metadata_dir),
                "manifest_path": str(result.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
