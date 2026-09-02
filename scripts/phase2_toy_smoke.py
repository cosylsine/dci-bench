#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.agents.pi_agent import run_pi_dci
from dci_bench.scoring.retrieval import score_query
from dci_bench.toy import build_toy_workspace


def main() -> None:
    paths = build_toy_workspace()
    result = run_pi_dci(
        query=paths["query"].read_text(encoding="utf-8"),
        workspace=paths["workspace"],
        output_path=Path("results/phase2/toy/final.json"),
        trace_path=Path("results/phase2/toy/trace.json"),
        mock_script=paths["mock_script"],
    )
    score = score_query({"ranked_doc_ids": result.ranked_doc_ids}, {"doc_02": 1.0})
    print(
        {
            "valid_output": result.valid_output,
            "ranked_doc_ids": result.ranked_doc_ids,
            "agent_steps": result.agent_steps,
            "tool_calls": result.tool_calls,
            "score": score,
            "output_path": str(result.output_path),
            "trace_path": str(result.trace_path),
        }
    )
    if not result.valid_output or result.ranked_doc_ids[:1] != ["doc_02"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
