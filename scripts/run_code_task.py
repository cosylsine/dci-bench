#!/usr/bin/env python3
"""Run an online code retrieval task, then score each sample on the host."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.data.code_online import load_code_samples
from dci_bench.protocol.code_runtime import code_runtime_contract
from dci_bench.scoring.code import load_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--code-root", type=Path, help="Completed ARB export root, never a staging directory")
    data.add_argument("--queries", type=Path, help="Custom host query list; requires --labels")
    parser.add_argument("--labels", type=Path, help="Custom dci-code-labels-v1 JSON")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--query-ids", nargs="+")
    selection.add_argument("--all-queries", action="store_true")
    parser.add_argument("--task", help="Optional task-name filter within the code suite")
    parser.add_argument("--backend-manifest", type=Path, help="Existing serving manifest; does not start the server")
    parser.add_argument("--model", help="Inspect model name, e.g. openai-api/vllm/Qwen3.8-27B")
    parser.add_argument("--base-url", help="Existing OpenAI-compatible /v1 endpoint")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--context-window", type=int)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--bridge-port", type=int, default=13131)
    parser.add_argument("--metric-ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--token-budgets", type=int, nargs="+", default=[4000, 8000, 16000, 32000])
    parser.add_argument("--output-dir", type=Path, help="New run directory; existing runs are never overwritten")
    parser.add_argument("--preflight-only", action="store_true", help="Validate data, ports and real Pi sandbox without model requests")
    parser.add_argument("--mock-script", type=Path, help="CPU acceptance only: Pi mock steps instead of a model service")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        from dci_bench.results.code_runner import code_preflight, execute_code_run
        from dci_bench.tasks.code_retrieval import code_retrieval
        backend_manifest = None
        if args.backend_manifest is not None:
            from dci_bench.backends.manifest import validate_backend_manifest
            backend_manifest = load_json(args.backend_manifest.read_text(encoding="utf-8"))
            validate_backend_manifest(backend_manifest, generation_contract=code_runtime_contract(args.max_files))
            backend, model = backend_manifest["backend"], backend_manifest["model"]
            expected_model = f"openai-api/{backend['openai_service']}/{model['served_model_name']}"
            expected_context = int(backend["context_length"])
            if ((args.model is not None and args.model != expected_model)
                    or (args.base_url is not None and args.base_url != backend["base_url"])
                    or (args.context_window is not None and args.context_window != expected_context)):
                raise ValueError("model/base-url/context-window overrides disagree with backend manifest")
            if backend["generation_config"] != code_runtime_contract(args.max_files)["generation_config"]:
                raise ValueError("backend generation config differs from code-v1; document pilot settings cannot be reused")
            args.model, args.base_url, args.context_window = expected_model, backend["base_url"], expected_context
        if args.mock_script is not None:
            if args.backend_manifest is not None or args.base_url is not None or args.model not in (None, "mockllm/model"):
                raise ValueError("mock-script cannot be combined with a real backend")
            args.model = "mockllm/model"
            args.mock_script = args.mock_script.resolve(strict=True)
        if not args.preflight_only and (not args.model or (args.mock_script is None and not args.base_url)):
            raise ValueError("online run needs backend-manifest or both model and base-url")
        context_window = args.context_window if args.context_window is not None else 65536
        samples = load_code_samples(code_root=args.code_root, queries_path=args.queries, labels_path=args.labels,
                                    query_ids=args.query_ids, all_queries=args.all_queries, task_name=args.task)
        run_dir = args.output_dir or Path("results/code") / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
        )
        if args.preflight_only:
            # Factory validation is pure; this creates neither result files nor requests.
            code_retrieval(samples, output_dir=run_dir / "samples", metric_ks=args.metric_ks,
                           max_files=args.max_files, token_budgets=args.token_budgets,
                           bridge_port=args.bridge_port, max_concurrency=args.max_concurrency,
                           model_context_window=context_window)
            code_preflight(samples, max_files=args.max_files, bridge_port=args.bridge_port,
                           max_concurrency=args.max_concurrency)
            print(json.dumps({"evaluation_track": "code", "status": "preflight_passed", "num_queries": len(samples)}))
            return 0
        report = execute_code_run(
            samples, run_dir=run_dir, model=args.model, base_url=args.base_url, api_key=args.api_key,
            metric_ks=args.metric_ks, max_files=args.max_files, token_budgets=args.token_budgets,
            bridge_port=args.bridge_port, max_concurrency=args.max_concurrency, context_window=context_window,
            mock_script=args.mock_script, backend_manifest=backend_manifest,
        )
        print(json.dumps({"run_dir": str(run_dir), "status": report["status"],
                          "num_queries": report["num_queries"], "valid_outputs": report["valid_outputs"],
                          "invalid_outputs": report["invalid_outputs"]}, ensure_ascii=False))
        if report["status"] == "infrastructure_error":
            return 2
        return 1 if report["invalid_outputs"] else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        message = str(exc).replace(args.api_key, "[redacted]") if args.api_key else str(exc)
        print(f"code evaluation error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
