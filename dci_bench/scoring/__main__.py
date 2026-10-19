"""Score saved document or code retrieval predictions without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from dci_bench.protocol.code import code_protocol_contract
from dci_bench.protocol.contracts import protocol_contract
from dci_bench.scoring import code, document


def _read(path: Path) -> tuple[Any, str]:
    data = path.read_bytes()
    return code.load_json(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def _publish(path: Path, text: str) -> None:
    """Publish atomically without replacing any existing result, even in a race."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=("document", "code"))
    parser.add_argument("--predictions", type=Path, required=True, help="JSON query-id mapping")
    parser.add_argument("--labels", type=Path, required=True, help="Document qrels or dci-code-labels-v1 JSON")
    parser.add_argument("--metric-ks", type=int, nargs="+")
    parser.add_argument("--max-files", type=int, help="Code only; default 20")
    parser.add_argument("--token-budgets", type=int, nargs="+", help="Code only; ARB whole-file context packing")
    parser.add_argument("--output", type=Path, help="New report path; never overwrites. Default: stdout")
    args = parser.parse_args(argv)
    try:
        if args.track == "document" and (args.max_files is not None or args.token_budgets is not None):
            raise ValueError("--max-files and --token-budgets apply only to the code track")
        predictions, predictions_hash = _read(args.predictions)
        labels, labels_hash = _read(args.labels)
        kwargs: dict[str, Any] = {}
        if args.metric_ks is not None:
            kwargs["metric_ks"] = args.metric_ks
        if args.track == "code":
            if args.max_files is not None:
                kwargs["max_files"] = args.max_files
            if args.token_budgets is not None:
                kwargs["token_budgets"] = args.token_budgets
            result = code.score_run(predictions, labels, **kwargs)
            contract = code_protocol_contract()
            # Freeze effective settings, not just the defaults, in the report.
            contract["metric_ks"] = result["metric_ks"]
            contract["max_files"] = result["max_files"]
            contract["token_budgets"] = result["token_budgets"]
            contract["final_answer_schema"]["properties"]["ranked_files"]["maxItems"] = result["max_files"]
            version = code.CODE_SCORING_VERSION
        else:
            if not isinstance(predictions, dict) or not isinstance(labels, dict):
                raise ValueError("document predictions and qrels must be query-id objects")
            if "schema_version" in labels:
                raise ValueError("document labels must be a qrels mapping, not code labels")
            result = document.score_run(predictions, labels, **kwargs)
            contract = protocol_contract()
            contract["evaluation_metric_ks"] = list(document.normalize_metric_ks(args.metric_ks))
            version = document.SCORING_VERSION
        contract_bytes = json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")
        report = {
            "schema_version": "dci-offline-evaluation-v1",
            "evaluation_track": args.track,
            "scoring_version": version,
            "input_sha256": {"predictions": predictions_hash, "labels": labels_hash},
            "evaluation_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "evaluation_contract": contract,
            "result": result,
        }
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(text)
        else:
            _publish(args.output, text)
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
