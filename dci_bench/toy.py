"""Phase 2 toy workspace helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def build_toy_workspace(root: Path = Path("data/toy/phase2")) -> dict[str, Path]:
    if root.exists():
        shutil.rmtree(root)
    workspace = root / "workspace"
    corpus = workspace / "corpus"
    scratch = root / "scratch"
    corpus.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    docs = {
        "doc_00": "A note about red apples and orchard irrigation.",
        "doc_01": "Meeting minutes for a budget review.",
        "doc_02": "The target document: polar bears rely on sea ice to hunt seals.",
        "doc_03": "A short recipe for tomato soup.",
        "doc_04": "Background on railway timetables.",
        "doc_05": "Notes about indoor air quality.",
        "doc_06": "A product warranty excerpt.",
        "doc_07": "Travel tips for mountain hikes.",
        "doc_08": "A sports schedule summary.",
        "doc_09": "An unrelated paragraph about database backups.",
    }
    for doc_id, text in docs.items():
        (corpus / f"{doc_id}.md").write_text(f"# {doc_id}\n\n{text}\n", encoding="utf-8")
    query = root / "query.txt"
    query.write_text("Which document explains how polar bears hunt seals?", encoding="utf-8")
    mock_script = root / "mock-script.json"
    mock_script.write_text(
        json.dumps(
            [
                {
                    "type": "tool_call",
                    "name": "bash",
                    "arguments": {
                        "command": "grep -Ril 'polar bears\\|hunt seals' /workspace/corpus",
                        "timeout_seconds": 10,
                    },
                    "id": "call_find",
                },
                {
                    "type": "tool_call",
                    "name": "read",
                    "arguments": {"path": "/workspace/corpus/doc_02.md"},
                    "id": "call_read",
                },
                {"type": "final", "text": "{\"ranked_doc_ids\":[\"doc_02\"]}"},
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"root": root, "workspace": workspace, "query": query, "mock_script": mock_script}
