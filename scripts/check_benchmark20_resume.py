#!/usr/bin/env python3
"""Check a fixed-budget restart without rewriting its canonical artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.backends.manifest import load_backend_manifest
from scripts import run_benchmark_v2 as runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--token-budget', type=int, required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--serving-version')
    args = parser.parse_args(argv)
    backend_path = args.run.resolve() / 'backend.json'
    launch_path = args.run.resolve() / 'test/launch.json'
    if (args.run / 'test').exists() and not launch_path.is_file():
        raise ValueError('test directory has no launch.json; preserve it for inspection')
    if launch_path.is_file() and not backend_path.is_file():
        raise ValueError('resume requires the original backend.json')

    if backend_path.is_file():
        manifest = load_backend_manifest(backend_path)
        expected = {
            'kind': 'vllm', 'context_length': 65536, 'tensor_parallel_size': 4,
            'dtype': 'bfloat16', 'tool_call_parser': 'qwen3_coder',
            'reasoning_parser': 'qwen3', 'language_model_only': True,
            'max_num_seqs': args.concurrency,
            'gpu_memory_utilization': float(os.environ['VLLM_GPU_MEMORY_UTILIZATION']),
            'base_url': f"http://127.0.0.1:{os.environ['VLLM_PORT']}/v1",
        }
        if args.serving_version:
            expected['version'] = args.serving_version
        for key, value in expected.items():
            if manifest['backend'].get(key) != value:
                raise ValueError('resume serving configuration differs: ' + key)
        model_path = Path(os.environ['MODEL_PATH']).resolve()
        for key, value in {
            'model_key': 'Qwen3.8-27B', 'served_model_name': 'Qwen3.8-27B',
            'source_path': str(model_path),
            'config_sha256': runner.sha256_file(model_path / 'config.json'),
        }.items():
            if manifest['model'].get(key) != value:
                raise ValueError('resume model configuration differs: ' + key)

    # The second check validates the installed serving version just before load.
    if args.serving_version:
        return 0
    if not launch_path.is_file():
        print('No test launch yet; restart preparation in the same run directory.')
        return 0
    runtime = runner.verify_runtime_config(args.runtime_config)
    kwargs = dict(split='test', concurrency=args.concurrency,
                  token_budget=args.token_budget, backend_path=backend_path)
    if runtime is not None:
        kwargs['agent_runtime'] = runtime
    plan = runner.build_plan(args.suite.resolve(), **kwargs)
    plan.update(
        wall_time_limit_hours=None, price_quote=None, budget_cny=1000, preflight_only=False,
        budget_policy={
            'mode': 'user-fixed', 'token_budget': args.token_budget,
            'reason': f'User-requested fixed cumulative budget: {args.token_budget} tokens',
            'development_selection_applied': False, 'calibration_projection_applied': False,
        },
    )
    plan['fingerprint'] = runner.fingerprint(plan)
    if plan != json.loads(launch_path.read_text()):
        raise ValueError('resume launch fingerprint/configuration differs; preserve original code, data and settings')
    print('Resume launch identity matches; original backend.json retained.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
