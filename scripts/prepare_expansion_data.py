#!/usr/bin/env python3
"""Build/verify the pinned 2026-10-14 expansion, without running any model."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.expansion import (
    EXPANSION_VERSION, build_document_task, sha256_file, verify_document_task,
    verify_hfd_source, write_json,
)
from dci_bench.data.expansion_documents import (
    browsecomp_tasks, enterprise_tasks, wixqa_tasks, worksurface_tasks,
)
from dci_bench.data.loader import stable_json_sha256
from dci_bench.data.loader import load_corpus, load_queries, load_qrels
from dci_bench.data.registry import TASKS

BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench')
SOURCES = {
    'wixqa': ('Wix/WixQA', 'd662dc42479c14e202eccd832f8c4b66a035c4cc', wixqa_tasks),
    'browsecomp_plus': ('Tevatron/browsecomp-plus-corpus', 'b27b02bc3e45511b8b82a13e6f90ce761df726f6', browsecomp_tasks),
    'enterprise_rag': ('onyx-dot-app/EnterpriseRAG-Bench', '69916e31c68aa5963c00248fd7f0bc12d04fd235', enterprise_tasks),
    'worksurface': ('lhpku20010120/WorkSurface-Bench', '3b3aef45f1e8f69416d6112c46e6e873cf522315', worksurface_tasks),
}
TOPICS_REVISION = '046949032b0328319cc9a02663a759ec601d9402'
SOURCE_TASKS = {
    'wixqa': {'wixqa_expertwritten', 'wixqa_simulated'},
    'browsecomp_plus': {'browsecomp_plus'},
    'enterprise_rag': {'enterprise_rag'},
    'worksurface': {f'worksurface_{persona}_rag' for persona in
                    ('backend_developer', 'logistics_manager', 'operations_manager')},
}


def verify_catalog(suite: Path) -> dict:
    complete = json.loads((suite / '_COMPLETE.json').read_text())
    if sha256_file(suite / 'catalog.json') != complete['catalog_sha256']:
        raise ValueError('Suite catalog hash differs from completion marker')
    for name, digest in complete.get('audit_hashes', {}).items():
        if name not in ('annotation_audit.json', 'build_provenance.json') or sha256_file(suite / name) != digest:
            raise ValueError('Suite audit artifact hash differs')
    catalog = json.loads((suite / 'catalog.json').read_text())
    tasks = {row['task']: row for row in catalog['tasks']}
    if (len(tasks) != len(catalog['tasks']) or len(tasks) != complete['num_tasks']
            or sum(t['num_queries'] for t in tasks.values()) != complete['num_queries']
            or set(tasks) != set(catalog['task_completion_hashes'])
            or set(tasks) != {p.name for p in (suite / 'metadata').iterdir()}):
        raise ValueError('Suite catalog task identities or counts differ')
    for name, row in tasks.items():
        if sha256_file(suite / 'metadata' / name / '_COMPLETE.json') != catalog['task_completion_hashes'][name]:
            raise ValueError(f'Task completion hash differs: {name}')
        manifest = json.loads((suite / 'metadata' / name / 'manifest.json').read_text())
        if any(manifest[k] != v for k, v in row.items()):
            raise ValueError(f'Catalog and task manifest differ: {name}')
    return catalog


def fetch_topics(topics: Path) -> dict:
    """Small official GitHub sidecars; all bulk datasets use the requested hfd."""
    files = {}
    topics.mkdir(parents=True, exist_ok=True)
    environment = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')}
    for name in ('qrel_evidence.txt', 'qrel_golds.txt'):
        url = f'https://raw.githubusercontent.com/texttron/BrowseComp-Plus/{TOPICS_REVISION}/topics-qrels/{name}'
        path = topics / name
        if not path.exists():
            data = subprocess.run(['curl', '-4', '-fsSL', '--connect-timeout', '10', '--max-time', '60',
                                   '--retry', '3', '--retry-delay', '2', url], env=environment, check=True, capture_output=True).stdout
            if not data:
                raise ValueError(f'Empty topic file: {url}')
            with path.open('xb') as out:
                out.write(data)
        files[name] = {'url': url, 'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    result = {'repository': 'texttron/BrowseComp-Plus', 'revision': TOPICS_REVISION, 'files': files}
    if (topics / 'source_manifest.json').exists():
        if json.loads((topics / 'source_manifest.json').read_text()) != result:
            raise ValueError('Official topics sidecar changed')
    else:
        write_json(topics / 'source_manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES), default=['wixqa', 'browsecomp_plus', 'enterprise_rag'])
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--fetch-topics-only', action='store_true')
    parser.add_argument('--baseline-only', action='store_true', help='Copy and verify the six existing pinned tasks into the new suite')
    parser.add_argument('--finalize', action='store_true', help='Verify and freeze a suite containing --sources and all six baseline tasks')
    parser.add_argument('--sources-only', action='store_true', help='With --finalize, freeze only the selected sources as a standalone supplement')
    parser.add_argument('--io-workers', type=int, default=16)
    args = parser.parse_args()
    if args.sources_only and not args.finalize:
        parser.error('--sources-only requires --finalize')
    suite = args.suite or args.base / 'eval_data/dci_expansion_20261014_v1'
    if args.fetch_topics_only:
        print(json.dumps(fetch_topics(args.base / 'eval_data/browsecomp_plus/topics'), indent=2))
        return
    if args.verify_only:
        if (suite / '_COMPLETE.json').exists():
            verify_catalog(suite)
        for metadata in sorted((suite / 'metadata').iterdir()):
            result = verify_document_task(suite, metadata.name, io_workers=args.io_workers)
            print(json.dumps({'verified': metadata.name, 'queries': result['num_queries'], 'documents': result['num_docs']}), flush=True)
        return
    if args.baseline_only:
        for spec in TASKS.values():
            raw = args.base / 'eval_data/mteb_llm_retrieval' / spec.local_dir_name
            revision = json.loads((raw / '.hfd/repo_metadata.json').read_text())['sha']
            if not revision.startswith(spec.revision):
                raise ValueError('Existing baseline raw revision differs from registry')
            source = verify_hfd_source(raw, spec.hf_dataset, revision)
            queries, qrels = load_queries(raw), load_qrels(raw)
            task = {'name': spec.name, 'dataset': spec.hf_dataset, 'revision': spec.revision, 'resolved_revision': revision,
                    'track': 'document', 'task_instruction': spec.task_instruction, 'corpus': load_corpus(raw),
                    'queries': queries, 'qrels': qrels, 'host_labels': qrels,
                    'provenance': {'candidate_scope': 'existing complete MTEB(LLM) corpus', 'exclusions': [],
                                   'original_split': 'existing pinned retrieval split', 'baseline': True}}
            result = build_document_task(task, suite, source, io_workers=args.io_workers)
            print(json.dumps({k: result[k] for k in ('task', 'num_docs', 'num_queries', 'num_qrels')}), flush=True)
        return
    catalog_path = suite / 'catalog.json'
    if catalog_path.exists():
        raise FileExistsError('Completed suite already exists; use --verify-only or a new --suite')
    for name in ([] if args.finalize else args.sources):
        dataset, revision, adapter = SOURCES[name]
        raw = args.base / 'eval_data' / name / 'raw'
        print(f'Verifying raw {name}', flush=True)
        source = verify_hfd_source(raw, dataset, revision)
        kwargs = {}
        if name == 'browsecomp_plus':
            topics = args.base / 'eval_data/browsecomp_plus/topics'
            source['topics'] = fetch_topics(topics)
            queries_raw = args.base / 'eval_data/browsecomp_plus_queries/raw'
            source['query_source'] = verify_hfd_source(queries_raw, 'Tevatron/browsecomp-plus', '144cff8e35b5eaef7e526346aa60774a9deb941f')
            source['source_fingerprint'] = stable_json_sha256({'files': source['files'], 'topics': source['topics'],
                                                             'query_source': source['query_source']})
            kwargs['topics'] = topics
            kwargs['queries_raw'] = queries_raw
        for task in adapter(raw, revision, **kwargs):
            print(f'Building {task["name"]}: {len(task["queries"])} annotated queries', flush=True)
            result = build_document_task(task, suite, source, io_workers=args.io_workers)
            print(json.dumps({k: result[k] for k in ('task', 'num_docs', 'num_queries', 'num_qrels')}), flush=True)
    # A partial source build is resumable but is not declared the complete suite.
    manifests = [json.loads(p.read_text()) for p in sorted((suite / 'metadata').glob('*/manifest.json'))]
    if args.finalize:
        expected_tasks = (set() if args.sources_only else set(TASKS)) | set().union(*(SOURCE_TASKS[name] for name in args.sources))
        if expected_tasks != {m['task'] for m in manifests}:
            raise ValueError('Selected tasks differ from completed tasks; suite is incomplete')
        for m in manifests:
            verify_document_task(suite, m['task'], io_workers=args.io_workers)
            print(f'Verified {m["task"]}: {m["num_docs"]} files', flush=True)
        catalog = {'schema_version': EXPANSION_VERSION, 'evaluation_track': 'document',
                   'suite_directory': str(suite), 'selected_sources': args.sources,
                   'task_completion_hashes': {m['task']: sha256_file(suite / 'metadata' / m['task'] / '_COMPLETE.json') for m in manifests},
                   'tasks': [{k: m[k] for k in ('task', 'dataset', 'revision', 'task_instruction', 'num_docs', 'num_queries', 'num_qrels')} for m in manifests]}
        write_json(catalog_path, catalog)
        write_json(suite / '_COMPLETE.json', {'catalog_sha256': sha256_file(catalog_path), 'num_tasks': len(manifests),
                                            'num_queries': sum(m['num_queries'] for m in manifests),
                                            'audit_hashes': {name: sha256_file(suite / name) for name in
                                                            ('annotation_audit.json', 'build_provenance.json')
                                                            if (suite / name).is_file()}})
        verify_catalog(suite)
        print(f'Complete document suite: {catalog_path}', flush=True)


if __name__ == '__main__':
    main()
