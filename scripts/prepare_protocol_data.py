#!/usr/bin/env python3
"""Publish a host-only index and online query views of existing frozen data.

This does not download, rewrite corpora, change labels, or invoke a model.
Normal verification checks metadata and protocol inputs. --verify-corpus also
rereads every source candidate and canonical document artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.expansion import (
    CANONICAL_FILES, EXPANSION_VERSION, METADATA_FILES, sha256_file, verify_document_task, write_json,
)
from dci_bench.data.expansion_code import load_reference, verify_arb
from dci_bench.data.expansion_code_subset import ARTIFACTS as SUBSET_ARTIFACTS
from dci_bench.data.loader import stable_json_sha256
from dci_bench.data.registry import TASKS, TaskSpec, validate_task_manifest
from dci_bench.protocol.code import CODE_LABELS_VERSION, code_protocol_contract
from dci_bench.scoring.code import load_json, score_query as score_code, snapshot_identity
from dci_bench.scoring.retrieval import score_query as score_document
from scripts.prepare_expansion_data import verify_catalog

BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench')
VERSION = 'dci-protocol-data-v1'
ARTIFACTS = {'manifest.json', 'document_tasks.json', 'code_views.json', 'commands.json',
             'audit.json', 'build_source.py', 'code/file_view/queries.json',
             'code/gold_untruncated/queries.json'}


class SourceReader:
    """Record exact bytes of every host dependency in the release hash chain."""

    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f'Source hash mismatch: {path}')
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f'Source changed during preparation: {path}')
        self.hashes[str(path)] = digest
        return data.decode('utf-8')

    def json(self, path, expected=None):
        return load_json(self.read(path, expected))


def document_inputs(suites, reader):
    from dci_bench.results.phase4_runner import data_audit
    from dci_bench.tasks.mteb_llm_retrieval import _make_samples

    tasks, audits, seen = [], [], set()
    for suite in suites:
        complete = reader.json(suite / '_COMPLETE.json')
        catalog = reader.json(suite / 'catalog.json', complete['catalog_sha256'])
        if (catalog.get('schema_version') != EXPANSION_VERSION or catalog.get('evaluation_track') != 'document'
                or not isinstance(complete.get('audit_hashes'), dict)):
            raise ValueError('Expected a completed document expansion catalog')
        verify_catalog(suite)
        for filename, digest in complete.get('audit_hashes', {}).items():
            reader.read(suite / filename, digest)
        for row in catalog['tasks']:
            name = row['task']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in seen:
                raise ValueError(f'Unsafe or duplicate document task: {name}')
            seen.add(name)
            md = suite / 'metadata' / name
            done = reader.json(md / '_COMPLETE.json', catalog['task_completion_hashes'][name])
            if (done.get('schema_version') != EXPANSION_VERSION or set(done['metadata_hashes']) != METADATA_FILES
                    or set(done['canonical_hashes']) != CANONICAL_FILES):
                raise ValueError('Incomplete document metadata hash catalog')
            values = {key: reader.read(md / key, digest) for key, digest in done['metadata_hashes'].items()}
            manifest = load_json(values['manifest.json'])
            if manifest.get('evaluation_track') != 'document' or manifest.get('expansion_version') != EXPANSION_VERSION:
                raise ValueError('Expected document task metadata')
            spec = TaskSpec(name, row['dataset'], row['revision'], name, row['task_instruction'])
            validate_task_manifest(spec, manifest)
            previous = TASKS.get(name)
            if previous and (previous.hf_dataset, previous.revision, previous.task_instruction) != (
                    spec.hf_dataset, spec.revision, spec.task_instruction):
                raise ValueError(f'Document task conflicts with current registry: {name}')
            queries = [load_json(line) for line in values['queries.jsonl'].splitlines()]
            qrels = load_json(values['qrels.json'])
            mapping = load_json(values['doc_id_files.json'])
            candidate_ids = set(mapping)
            if (len(queries) != len({q['_id'] for q in queries}) or not queries
                    or {q['_id'] for q in queries} != set(qrels)
                    or len(queries) != row['num_queries'] or len(mapping) != row['num_docs']
                    or sum(map(len, qrels.values())) != row['num_qrels']
                    or stable_json_sha256(queries) != manifest['queries_hash']
                    or stable_json_sha256(qrels) != manifest['qrels_hash']):
                raise ValueError(f'Document query/label/count mismatch: {name}')
            for query in queries:
                gold = qrels[query['_id']]
                if not isinstance(query['text'], str) or not query['text'].strip() or not gold or set(gold) - candidate_ids:
                    raise ValueError(f'Invalid query or unknown positive: {name}/{query["_id"]}')
                score = score_document(sorted(gold, key=lambda d: (-gold[d], d))[:10], gold, metric_ks=[10])
                if not score['valid_output'] or abs(score['recall_at_10'] - min(10, len(gold)) / len(gold)) > 1e-12:
                    raise ValueError('Document scorer oracle failed')
            TASKS[name] = spec
            try:
                data_audit(suite / 'metadata', name)
                samples = _make_samples(name, None, suite / 'metadata', suite / 'workspaces', all_queries=True)
                if [(s.id, s.input) for s in samples] != [(q['_id'], q['text']) for q in queries]:
                    raise ValueError('Document sample text or ID changed')
                if any(s.target or set(s.metadata) != {'task', 'query_id', 'query_index', 'workspace'} for s in samples):
                    raise ValueError('Unexpected host information in document sample')
            finally:
                if previous is None:
                    TASKS.pop(name)
                else:
                    TASKS[name] = previous
            tasks.append({**row, 'suite': str(suite), 'workspace': str(suite / 'workspaces' / name),
                          'metadata': str(md), 'canonical': str(suite / 'canonical' / name)})
            audit = load_json(values['audit.json'])
            audits.append({'task': name, 'audit_path': str(md / 'audit.json'),
                           'raw_num_docs': audit['raw_num_docs'],
                           'document_exclusions': audit['document_exclusions'],
                           'query_exclusions': audit['query_exclusions'],
                           'adapter_exclusions': audit['adapter_exclusions'],
                           'scorer_oracle_queries': len(queries), 'inspect_samples': len(samples)})
    return tasks, audits


def code_inputs(source, subset, labels_path, reader, output, *, io_workers=16):
    # Reuse the source adapter's identity, snapshot/tree and source-label checks.
    # This mode checks file attributes/closure but does not read candidate bytes.
    verify_arb(source, verify_corpus=False, io_workers=io_workers)
    complete = reader.json(source / '_COMPLETE.json')
    manifest = reader.json(source / 'manifest.json', complete['manifest_sha256'])
    if manifest['schema_version'] != 'dci-arb-file-view-v1':
        raise ValueError('Expected frozen ARB file-view export')
    required = {'queries.json', 'host/labels.json', 'host/raw_samples.json', 'protocol.json'}
    if set(manifest['artifact_hashes']) != required:
        raise ValueError('Incomplete ARB artifact hash catalog')
    artifacts = {name: reader.json(source / name, digest) for name, digest in manifest['artifact_hashes'].items()}
    if artifacts['protocol.json'] != code_protocol_contract():
        raise ValueError('Code scoring protocol changed')
    queries, compact = artifacts['queries.json'], artifacts['host/labels.json']['queries']
    labels = reader.json(labels_path)
    if set(labels) != {'schema_version', 'queries'} or labels['schema_version'] != CODE_LABELS_VERSION:
        raise ValueError('Expected expanded dci-code-labels-v1')
    by_id = {q['query_id']: q for q in queries}
    if (len(queries) != len(by_id) or set(by_id) != set(compact) or set(by_id) != set(labels['queries'])
            or len(queries) != manifest['samples'] or len(queries) != complete['samples']):
        raise ValueError('ARB query/label/count mismatch')
    snapshots = {}
    for key, digest in manifest['snapshot_manifest_sha256'].items():
        if not re.fullmatch('[0-9a-f]{64}', key):
            raise ValueError('Unsafe snapshot ID')
        snapshots[key] = reader.json(source / 'snapshots' / key / 'manifest.json', digest)
    untruncated = set()
    for qid, row in by_id.items():
        reference = load_reference(compact[qid], source / 'snapshots')
        if (row['snapshot_id'] != compact[qid]['snapshot_id'] or row['task'] != reference['task']
                or labels['queries'][qid] != reference or not isinstance(row['query'], str) or not row['query'].strip()):
            raise ValueError(f'Code query/reference mismatch: {qid}')
        gold = reference['gold_files']
        prediction = {**snapshot_identity(reference), 'output': {'ranked_files': gold[:20], 'abstain': False}}
        score = score_code(prediction, reference)
        if not score['valid_output'] or abs(score['metrics']['file_recall_at_20'] - min(20, len(gold)) / len(gold)) > 1e-12:
            raise ValueError(f'Code scorer oracle failed: {qid}')
        snapshot = snapshots[row['snapshot_id']]
        if all(not snapshot['upstream_file_view'][p]['truncation_marker'] for p in gold):
            untruncated.add(qid)
    subset_done = reader.json(subset / '_COMPLETE.json')
    if set(subset_done['artifact_sha256']) != SUBSET_ARTIFACTS:
        raise ValueError('Incomplete code subset hash catalog')
    small = {name: reader.read(subset / name, digest) for name, digest in subset_done['artifact_sha256'].items()}
    small_manifest = load_json(small['manifest.json'])
    small_rows = load_json(small['queries.json'])
    small_labels = load_json(small['labels.json'])
    if (small_manifest['source_completion_sha256'] != reader.hashes[str(source / '_COMPLETE.json')]
            or Path(small_manifest['source_directory']).resolve() != source
            or small_labels['schema_version'] != CODE_LABELS_VERSION
            or len(small_rows) != len(untruncated) or {q['query_id'] for q in small_rows} != untruncated
            or set(small_labels['queries']) != untruncated or small_manifest['samples'] != len(untruncated)):
        raise ValueError('Code subset differs from the complete untruncated-positive selection')
    for row in small_rows:
        qid = row['query_id']
        if row != by_id[qid] or small_labels['queries'][qid] != labels['queries'][qid]:
            raise ValueError(f'Subset changed source query, candidates, or gold: {qid}')
    views, normalized = [], {}
    for name, rows, label_file in [('file_view', queries, labels_path),
                                    ('gold_untruncated', small_rows, subset / 'labels.json')]:
        relative = f'code/{name}/queries.json'
        normalized[relative] = [{'query_id': q['query_id'], 'query': q['query'],
                                 'workspace': str(source / 'snapshots' / q['snapshot_id'])} for q in rows]
        views.append({'view': name, 'queries': str(output / relative), 'labels': str(label_file),
                      'code_root': str(source), 'num_queries': len(rows),
                      'num_qrels': sum(len(labels['queries'][q['query_id']]['gold_files']) for q in rows),
                      'tasks': dict(sorted(Counter(q['task'] for q in rows).items())),
                      'source_complete': False, 'line_function_annotations': 'unavailable',
                      'subset_of': 'file_view' if name == 'gold_untruncated' else None})
    return views, normalized, {'source_files': manifest['files'], 'source_snapshots': manifest['snapshots'],
                              'upstream_truncated_files': manifest['truncated_files'],
                              'queries_with_truncated_positive': len(queries) - len(untruncated),
                              'scorer_oracle_queries': len(queries),
                              'selection_audit': str(subset / 'selection_audit.json')}


def verify_release(output, *, verify_corpus=False, io_workers=16):
    complete = load_json((output / '_COMPLETE.json').read_text())
    if complete['schema_version'] != VERSION or set(complete['artifact_sha256']) != ARTIFACTS:
        raise ValueError('Incomplete protocol data release')
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != ARTIFACTS | {'_COMPLETE.json'}:
        raise ValueError('Unexpected/missing protocol data artifact')
    for name, digest in complete['artifact_sha256'].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f'Protocol data artifact changed: {name}')
    manifest = load_json((output / 'manifest.json').read_text())
    for name, digest in manifest['source_artifact_sha256'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    if verify_corpus:
        for row in load_json((output / 'document_tasks.json').read_text()):
            verify_document_task(Path(row['suite']), row['task'], io_workers=io_workers)
            print(f'Corpus verified: {row["task"]}', flush=True)
        verify_arb(Path(manifest['code_root']), io_workers=io_workers)
    return {'status': 'verified', **manifest['counts'], 'source_corpus_bytes_verified': verify_corpus,
            'model_calls': 0, 'gpu_runs': 0}


def build_release(output, suites, source, subset, labels_path, *, io_workers=16):
    output = output.resolve()
    protected = [*suites, source, subset]
    if os.path.lexists(output) or any(output.is_relative_to(p) for p in protected):
        raise FileExistsError('Use a new output directory outside the frozen source packages')
    reader = SourceReader()
    documents, document_audits = document_inputs(suites, reader)
    print(f'Document protocol inputs checked: {len(documents)} tasks', flush=True)
    views, normalized, code_audit = code_inputs(source, subset, labels_path, reader, output, io_workers=io_workers)
    commands = {}
    for row in documents:
        commands[f'document/{row["task"]}'] = [sys.executable, str(ROOT / 'scripts/run_expansion_task.py'),
            '--suite', row['suite'], '--task', row['task'], '--all-queries']
    for view in views:
        for task in view['tasks']:
            commands[f'code/{view["view"]}/{task}'] = [sys.executable, str(ROOT / 'scripts/run_code_task.py'),
                '--queries', view['queries'], '--labels', view['labels'], '--task', task, '--all-queries']
    counts = {'document_tasks': len(documents), 'document_queries': sum(t['num_queries'] for t in documents),
              'document_entries': sum(t['num_docs'] for t in documents),
              'document_positive_pairs': sum(t['num_qrels'] for t in documents),
              'code_queries': views[0]['num_queries'], 'code_positive_pairs': views[0]['num_qrels'],
              'code_gold_untruncated_queries': views[1]['num_queries']}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}.building-', dir=output.parent))
    manifest = {'schema_version': VERSION, 'counts': counts, 'document_suites': list(map(str, suites)),
                'code_root': str(source), 'code_subset': str(subset),
                'source_artifact_sha256': reader.hashes,
                'layout': 'host-only references to immutable source packages; candidates are not copied',
                'candidate_policy': 'preserve every candidate in each selected original task/snapshot',
                'subset_counts_overlap': True,
                'build_validation': 'metadata hashes, IDs/labels, host samples, scorer oracle; no model calls',
                'source_corpus_bytes_verified_in_build': False}
    for name, value in {'manifest.json': manifest, 'document_tasks.json': documents,
                        'code_views.json': views, 'commands.json': commands,
                        'audit.json': {'document': document_audits, 'code': code_audit,
                                       'model_calls': 0, 'gpu_runs': 0}, **normalized}.items():
        write_json(stage / name, value)
    (stage / 'build_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(stage / '_COMPLETE.json', {'schema_version': VERSION,
        'artifact_sha256': {name: sha256_file(stage / name) for name in sorted(ARTIFACTS)}})
    verify_release(stage)
    if os.path.lexists(output):
        raise FileExistsError('Output appeared during preparation; staging preserved')
    os.rename(stage, output)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--document-suites', nargs='+', type=Path, default=[
        BASE / 'eval_data/dci_expansion_20261014_v1', BASE / 'eval_data/worksurface/dci_v1_20261014'])
    code_base = BASE / 'eval_data_code/agent_retrieval_bench'
    parser.add_argument('--code-root', type=Path, default=code_base / 'dci_v1_20261014')
    parser.add_argument('--code-subset', type=Path, default=code_base / 'dci_gold_untruncated_v1_20261014')
    parser.add_argument('--code-labels', type=Path, default=code_base / 'host_labels_v1_20261014.json')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verify-corpus', action='store_true', help='With --verify-only, also read all candidate bytes')
    parser.add_argument('--io-workers', type=int, default=16)
    parser.add_argument('--print-command', help='Print a recorded command key; never executes it')
    args = parser.parse_args(argv)
    if args.verify_corpus and not args.verify_only:
        parser.error('--verify-corpus requires --verify-only')
    if args.print_command:
        verify_release(args.output)
        commands = load_json((args.output / 'commands.json').read_text())
        print(shlex.join(commands[args.print_command]))
    elif args.verify_only:
        print(json.dumps(verify_release(args.output, verify_corpus=args.verify_corpus,
                                        io_workers=args.io_workers), sort_keys=True))
    else:
        print(json.dumps(build_release(args.output, [p.resolve(strict=True) for p in args.document_suites],
            args.code_root.resolve(strict=True), args.code_subset.resolve(strict=True),
            args.code_labels.resolve(strict=True), io_workers=args.io_workers), sort_keys=True))


if __name__ == '__main__':
    main()
