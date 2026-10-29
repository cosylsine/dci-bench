#!/usr/bin/env python3
"""Bounded synthetic KaLM smoke: 1,001 documents, three queries, three pools."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.data.pipeline import batch
from dci_bench.data.pipeline.core import publish_bundle
from dci_bench.data.pipeline.retrieval import model_weight_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--skip-content-hashes', action='store_true')
    parser.add_argument('--parent-bundle', type=Path,
                        help='Use an existing sealed model identity instead of rehashing weights')
    parser.add_argument('--separate-processes', action='store_true',
                        help='Exercise actual export/rank/pools worker processes on one visible GPU')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    documents = [{'_id': f'doc-{i:04d}', 'title': f'Archive entry {i}',
                  'text': f'The experiment identified sample {i}. Its reference number is {10000 + i}.',
                  'language': 'en', 'metadata': {}} for i in range(1001)]
    queries = [{'_id': 'q-first', 'text': 'What is the reference number for sample 0?', 'language': 'en', 'metadata': {}},
               {'_id': 'q-last', 'text': 'What reference number belongs to sample 1000?', 'language': 'en', 'metadata': {}},
               {'_id': 'q-unicode', 'text': 'Find sample 500.\u2028Return its archive entry.', 'language': 'en', 'metadata': {}}]
    data = {'schema_version': 'dci-data-bundle-v1', 'input_kind': 'retrieval',
            'source': {'dataset': 'synthetic-kalm-pool-smoke', 'revision': '20261024'},
            'documents': documents, 'queries': queries,
            'qrels': {'q-first': {'doc-0000': 1}, 'q-last': {'doc-1000': 2}, 'q-unicode': {'doc-0500': 1}},
            'evidence': [], 'judgments': [], 'provenance': {'synthetic': True}}
    release = args.output / 'synthetic_release'
    publish_bundle(data, release, specification={'id': 'synthetic-smoke', 'input_kind': 'retrieval'})
    entry = batch.inspect_release({'source_id': 'synthetic_smoke', 'source_path': str(release),
                                   'completion_sha256': batch.sha(release / '_COMPLETE.json'),
                                   'documents': 1001, 'queries': 3, 'positive_pairs': 3,
                                   'domain': 'synthetic', 'language': 'en'})
    weight_identity = (batch.load_plan(args.parent_bundle)['parameters']['weight_identity']
                       if args.parent_bundle else model_weight_identity(batch.MODEL))
    plan = {'validation': {'verify_content_hashes': not args.skip_content_hashes},
            'parameters': {'model_path': batch.MODEL, 'trust_remote_code': True,
                          'weight_identity': weight_identity,
                          'query_prompt': batch.QUERY_PROMPT, 'document_prompt': '',
                          'max_length': 8192, 'batch_size': args.batch_size, 'cache_shard_size': 256,
                          'search_query_batch_size': 2, 'top_k': 1000,
                          'capacities': [200, 500, 1000], 'seed': 20261021}}
    batch.write(args.output / 'smoke_plan.json', {'entry': entry, **plan})
    phases = []
    if args.separate_processes:
        if args.device != 'cuda:0':
            raise ValueError('Separate-process smoke uses --device cuda:0 on one visible device')
        bundle = batch.ROOT / 'releases/data-pipeline/synthetic' / args.output.resolve().name
        bundle.mkdir(parents=True, exist_ok=False)
        plan = {**plan, 'schema_version': batch.SCHEMA, 'sources': [entry],
                'run_root': str(args.output / 'run'),
                'code_files': batch.snapshot_code(bundle / 'code')}
        batch.write(bundle / 'plan.json', plan)
        (bundle / 'launch.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
        batch.write(bundle / '_COMPLETE.json', {'schema_version': batch.SCHEMA,
                    'plan_sha256': batch.sha(bundle / 'plan.json'),
                    'launcher_sha256': batch.sha(bundle / 'launch.sh')})
        slot = batch.cuda_slots(1)[0]
        for phase in ('export', 'rank', 'pools', 'export', 'pools'):
            attempt = len(phases)
            env = batch._worker_env(args.output / 'run', entry, phase, slot if phase == 'rank' else None)
            command = [sys.executable, str(bundle / 'code/scripts/run_kalm_pool_batch.py'),
                       '_worker', '--bundle', str(bundle), '--phase', phase,
                       '--source', entry['source_id']]
            phase_start = time.monotonic()
            with (args.output / f'phase-{attempt}-{phase}.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
            row = {'phase': phase, 'pid': child.pid, 'returncode': code,
                   'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES'],
                   'elapsed_seconds': time.monotonic() - phase_start}
            phases.append(row)
            print(json.dumps(row), flush=True)
            if code:
                raise RuntimeError(f'Smoke {phase} failed; inspect phase-{attempt}-{phase}.log')
            if attempt == 2:
                result = batch.read(args.output / 'run/sources/synthetic_smoke/_COMPLETE.json')
        second = batch.read(args.output / 'run/sources/synthetic_smoke/_COMPLETE.json')
        assert result == second
        assert all(row['cuda_visible_devices'] == '' for row in phases if row['phase'] != 'rank')
    else:
        result = batch.execute_source(entry, plan, args.output / 'run', device=args.device)
        # Check that a second invocation validates and reuses successful stages.
        second = batch.execute_source(entry, plan, args.output / 'run', device=args.device)
        assert result['ranking'] == second['ranking']
    receipt = {'boundary': 'synthetic single-device integration smoke, not dataset retrieval quality or four-GPU throughput',
               'elapsed_seconds': time.monotonic() - started, 'device': args.device,
               'documents': 1001, 'queries': 3, 'pool_capacities': [200, 500, 1000],
               'resume_verified': True, 'separate_processes': phases, 'result': result}
    batch.write(args.output / '_COMPLETE.json', receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
