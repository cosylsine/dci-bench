#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai import eval as inspect_eval

from dci_bench.data.registry import get_task
from dci_bench.data.workspace_builder import build_workspace
from dci_bench.tasks.mteb_llm_retrieval import mteb_llm_retrieval_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="LLMPublicHealthQA")
    parser.add_argument("--query-id", default="Q25")
    parser.add_argument("--model-path", default="/mnt/afs/share/Qwen3-4B")
    parser.add_argument("--served-model-name", default="Qwen3-4B")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--log-dir", default="results/inspect-logs")
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        build_workspace(get_task(args.task), overwrite=True)
    task = mteb_llm_retrieval_single(task_name=args.task, query_id=args.query_id)
    logs = inspect_eval(
        task,
        model=f"openai-api/vllm/{args.served_model_name}",
        model_base_url=args.vllm_base_url,
        model_args={"api_key": "EMPTY", "responses_api": False},
        log_dir=args.log_dir,
        display="plain",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_samples=1,
        fail_on_error=False,
        score_on_error=True,
    )
    print(logs)
    print(
        {
            "task": args.task,
            "query_id": args.query_id,
            "model_path": args.model_path,
            "served_model_name": args.served_model_name,
            "vllm_base_url": args.vllm_base_url,
            "log_dir": str(Path(args.log_dir)),
            "sandbox": "local",
        }
    )


if __name__ == "__main__":
    main()
