"""Frozen, resumable per-source KaLM pool jobs; one process per allocated GPU."""
from __future__ import annotations

import argparse
import copy
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench/eval_data')
DEFAULT_MATRIX = BASE / 'pipeline_matrix_overview_20261023_v3/data_matrix.json'
MODEL = '/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5'
QUERY_PROMPT = 'Instruct: Given a query, retrieve documents that answer the query \n Query: '
SCHEMA = 'dci-kalm-pool-batch-v1'
PHASES = ('export', 'rank', 'pools')
CPU_THREAD_LIMIT = 4
DEFAULT_EXPORT_WORKERS = 2
DEFAULT_POOL_WORKERS = 2
DEFAULT_EXPORT_PREFETCH = 8


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _paths_overlap(left, right):
    left, right = Path(left).absolute().resolve(), Path(right).absolute().resolve()
    return left == right or left in right.parents or right in left.parents


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def content_hashes_enabled(plan):
    value = plan.get('validation', {}).get('verify_content_hashes', True)
    if type(value) is not bool:
        raise ValueError('verify_content_hashes must be a boolean')
    return value


def compatible_parameters(old, new):
    """Batch size is operational; all retrieval/pool semantics must still match."""
    return ({k: v for k, v in old.items() if k != 'batch_size'}
            == {k: v for k, v in new.items() if k != 'batch_size'})


def _ranking_receipt_valid(value, plan):
    path = Path(value['path'])
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        return False
    if 'bytes' in value and path.stat().st_size != value['bytes']:
        return False
    if content_hashes_enabled(plan):
        return bool(value.get('sha256')) and sha(path) == value['sha256']
    return bool(value.get('sha256')) or (value.get('content_hashes_verified') is False
                                        and isinstance(value.get('bytes'), int))


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    if not replace and path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    os.replace(temporary, path)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process owns {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def audit_scifact(path=BASE / 'pipeline_scifact_rankings_v1.json'):
    """Inspect the saved executed configuration, rather than today's defaults."""
    path = Path(path)
    if not path.is_file():
        return {'rerun_required': True, 'reason': 'saved ranking is missing', 'path': str(path)}
    data = read(path)
    configs = [data.get('model', {})] + [data.get('encoding_audit', {}).get(k, {})
                                       for k in ('queries', 'documents')]
    ok = all(c.get('trust_remote_code') is True and c.get('query_prompt') == QUERY_PROMPT
             and c.get('document_prompt') == '' for c in configs)
    return {'path': str(path), 'sha256': sha(path), 'rerun_required': not ok,
            'reason': 'saved model and encoding audits confirm instructed KaLM' if ok else
                      'saved prompt/trust configuration is missing or differs',
            'query_prompt': configs[0].get('query_prompt'),
            'document_prompt': configs[0].get('document_prompt'),
            'trust_remote_code': configs[0].get('trust_remote_code'),
            'weight_identity': configs[0].get('weight_identity')}


def inspect_release(row):
    """Cheap preparation gate; full corpus hash verification runs before export."""
    release = Path(row['source_path'])
    complete = release / '_COMPLETE.json'
    if sha(complete) != row['completion_sha256']:
        raise ValueError(f'Changed release completion: {release}')
    inventory = read(complete)['files']
    manifest = read(release / 'manifest.json')
    if manifest['state'] != 'labels_ready' or manifest.get('automatic_labels'):
        raise ValueError(f'Expected original, ready labels: {release}')
    for name, info in inventory.items():
        p = release / name
        if p.is_symlink() or not p.is_file() or p.stat().st_nlink != 1 or p.stat().st_size != info['bytes']:
            raise ValueError(f'Release file size/type mismatch: {p}')
    # Labels and queries are small enough to inspect now, before GPU submission.
    for name in ('manifest.json', 'qrels.json', 'queries.jsonl'):
        if sha(release / name) != inventory[name]['sha256']:
            raise ValueError(f'Release metadata hash mismatch: {release / name}')
    labels = read(release / 'qrels.json')
    with (release / 'queries.jsonl').open(encoding='utf-8') as f:
        queries = [json.loads(line)['_id'] for line in f if line.strip()]
    if len(queries) != len(set(queries)) or set(queries) != set(labels):
        raise ValueError(f'Query/label coverage mismatch: {release}')
    positives = [sum(float(v) > 0 for v in rel.values()) for rel in labels.values()]
    if not positives or min(positives) < 1 or max(positives) > 200:
        raise ValueError(f'All queries must retain 1..200 positives: {release}')
    expected = (row['documents'], row['queries'], row['positive_pairs'])
    actual = (manifest['num_documents'], len(queries), sum(positives))
    if expected != actual or row['documents'] < 1000:
        raise ValueError(f'Source counts/tier eligibility changed: {release}')
    return {'source_id': row['source_id'], 'release': str(release), 'task': row['source_id'],
            'domain': row['domain'], 'language': row['language'],
            'documents': actual[0], 'queries': actual[1], 'positive_pairs': actual[2],
            'max_positives_per_query': max(positives),
            'completion_sha256': row['completion_sha256'],
            'manifest_sha256': inventory['manifest.json']['sha256'],
            'bundle_hash': manifest['bundle_hash'],
            'documents_jsonl_bytes': inventory['documents.jsonl']['bytes'],
            'qrels_sha256': inventory['qrels.json']['sha256'],
            'queries_sha256': inventory['queries.jsonl']['sha256']}


def select_sources(matrix, *, scifact):
    rows = [s for s in matrix['sources'] if s.get('origin') == 'pipeline'
            and s.get('inventory_role') == 'three_tier_source'
            and s.get('release_state') == 'labels_ready'
            and s['source_id'] != 'beir_scifact_test']
    if len(rows) != 53 or len({r['source_id'] for r in rows}) != 53:
        raise ValueError('This launch contract requires exactly the 53 selected v3 source views')
    if scifact['rerun_required']:
        rows += [s for s in matrix['sources'] if s['source_id'] == 'beir_scifact_test']
        if len(rows) != 54:
            raise ValueError('SciFact needs a rerun but its release is missing')
    return sorted(rows, key=lambda r: r['source_id'])


def snapshot_code(destination):
    files = []
    for directory in ('dci_bench', 'schemas'):
        files += [p for p in (ROOT / directory).rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts]
    files += [ROOT / 'scripts/run_kalm_pool_batch.py', ROOT / 'scripts/data_pipeline.py',
              ROOT / 'pyproject.toml']
    hashes = {}
    for p in sorted(files):
        relative = p.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        hashes[str(relative)] = sha(target)
    return hashes


def prepare(bundle, matrix=DEFAULT_MATRIX, *, run_root=None, batch_size=None, resume_bundle=None,
            verify_content_hashes=True):
    from .retrieval import model_weight_identity
    bundle, matrix = Path(bundle).absolute(), Path(matrix).absolute()
    if {'results', 'eval_data', 'raw_data', 'eval_data_code'} & set(bundle.resolve().parts):
        raise ValueError('Executable bundles belong under the code repository releases/ directory')
    if bundle.exists():
        raise FileExistsError(f'Choose a new bundle directory: {bundle}')
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError('batch size must be positive')
    if type(verify_content_hashes) is not bool:
        raise ValueError('verify_content_hashes must be a boolean')
    old_plan = None
    reuse = None
    if resume_bundle is not None:
        old_bundle = Path(resume_bundle).absolute()
        if _paths_overlap(bundle, old_bundle):
            raise ValueError('A resume bundle requires a bundle directory disjoint from the parent bundle')
        old_plan = load_plan(old_bundle)
        # Revalidate the frozen parent before copying its exact source and model
        # identities.  This checks source metadata and the current weight identity.
        preflight(old_bundle, gpu_count=0, verify_model=True,
                  verify_content_hashes=verify_content_hashes)
        entries = copy.deepcopy(old_plan['sources'])
        parameters = copy.deepcopy(old_plan['parameters'])
        if batch_size is not None:
            parameters['batch_size'] = batch_size
        matrix = Path(old_plan['matrix']).absolute()
        matrix_sha256 = old_plan['matrix_sha256']
        if sha(matrix) != matrix_sha256:
            raise ValueError('Resume matrix changed since the parent plan was sealed')
        scifact = copy.deepcopy(old_plan.get('scifact', {}))
        summary = copy.deepcopy(old_plan['summary'])
        old_root = Path(old_plan['run_root']).absolute()
        if run_root is None:
            run_root = (BASE / 'pipeline_pool_runs' / bundle.name).absolute()
        if _paths_overlap(run_root, old_root):
            raise ValueError('A resume bundle requires a disjoint run root; refusing to reuse the old run tree')
        reuse = {'bundle': str(old_bundle), 'plan_sha256': sha(old_bundle / 'plan.json'),
                 'run_root': old_plan['run_root']}
    else:
        original = read(matrix)
        scifact = audit_scifact()
        entries = [inspect_release(row) for row in select_sources(original, scifact=scifact)]
        parameters = {'model_path': MODEL, 'weight_identity': model_weight_identity(MODEL),
                      'trust_remote_code': True, 'query_prompt': QUERY_PROMPT,
                      'document_prompt': '', 'max_length': 8192, 'batch_size': batch_size or 32,
                      'cache_shard_size': 2048, 'search_query_batch_size': 32,
                      'top_k': 1000, 'capacities': [200, 500, 1000], 'seed': 20261021}
        matrix_sha256 = sha(matrix)
        summary = {'source_views': len(entries), 'document_entries': sum(s['documents'] for s in entries),
                   'queries': sum(s['queries'] for s in entries),
                   'candidate_slots_all_tiers': sum(s['queries'] for s in entries) * 1700,
                   'documents_jsonl_bytes': sum(s['documents_jsonl_bytes'] for s in entries)}
        run_root = Path(run_root or BASE / 'pipeline_pool_runs' / bundle.name).absolute()

    model_config = read(Path(parameters['model_path']) / 'config_sentence_transformers.json')
    if model_config.get('prompts') != {'query': parameters['query_prompt'],
                                       'document': parameters['document_prompt']}:
        raise ValueError('Local KaLM prompt configuration changed; inspect before sealing')
    stage = bundle.with_name('.' + bundle.name + '.staging-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    code = snapshot_code(stage / 'code')
    plan = {'schema_version': SCHEMA, 'created_at': now(),
            'matrix': str(matrix), 'matrix_sha256': matrix_sha256, 'sources': entries,
            'scifact': scifact, 'code_files': code,
            'run_root': str(Path(run_root).absolute()), 'parameters': parameters,
            'validation': {'verify_content_hashes': verify_content_hashes,
                           'boundary': ('content hashes verified' if verify_content_hashes else
                                        'structural checks only; declared content digests reused without byte verification')},
            'scheduling': {'export_workers': DEFAULT_EXPORT_WORKERS,
                           'pool_workers': DEFAULT_POOL_WORKERS,
                           'export_prefetch': DEFAULT_EXPORT_PREFETCH,
                           'cpu_thread_limit': CPU_THREAD_LIMIT},
            'summary': summary,
            'validation_boundary': 'metadata preflight; no full corpus scan, GPU ranking, or pools executed'}
    if reuse is not None:
        plan['reuse'] = reuse
    write(stage / 'plan.json', plan)
    # The launcher executes this code snapshot, not the mutable working checkout.
    launcher = '''#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EMBED_PY="${EMBED_PY:-/mnt/afs/250010100/miniconda/envs/retriever/bin/python}"
EMBED_ENV_LIB="$(dirname -- "$(dirname -- "$EMBED_PY")")/lib"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="$EMBED_ENV_LIB:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
exec "$EMBED_PY" "$BUNDLE_DIR/code/scripts/run_kalm_pool_batch.py" run --bundle "$BUNDLE_DIR" --gpus 4 "$@"
'''
    (stage / 'launch.sh').write_text(launcher, encoding='utf-8')
    (stage / 'launch.sh').chmod(0o755)
    write(stage / '_COMPLETE.json', {'schema_version': SCHEMA, 'plan_sha256': sha(stage / 'plan.json'),
                                   'launcher_sha256': sha(stage / 'launch.sh')})
    stage.rename(bundle)
    return {'bundle': str(bundle), 'command': 'bash ' + str(bundle / 'launch.sh'),
            'summary': plan['summary'], 'scifact': scifact}


def load_plan(bundle):
    bundle = Path(bundle).absolute()
    seal = read(bundle / '_COMPLETE.json')
    if seal.get('schema_version') != SCHEMA or sha(bundle / 'plan.json') != seal['plan_sha256']:
        raise ValueError('Batch plan seal mismatch')
    if sha(bundle / 'launch.sh') != seal['launcher_sha256']:
        raise ValueError('Batch launcher changed')
    plan = read(bundle / 'plan.json')
    for name, expected in plan['code_files'].items():
        if sha(bundle / 'code' / name) != expected:
            raise ValueError(f'Snapshot code changed: {name}')
    if plan['parameters']['trust_remote_code'] is not True or not plan['parameters']['query_prompt'].strip():
        raise ValueError('Instructed trusted KaLM is required')
    return plan


def preflight(bundle, *, gpu_count=0, verify_model=True, verify_content_hashes=None):
    plan = load_plan(bundle)
    verify_content_hashes = (content_hashes_enabled(plan) if verify_content_hashes is None
                             else verify_content_hashes)
    versions = {k: importlib.metadata.version(k)
                for k in ('torch', 'transformers', 'sentence-transformers', 'numpy', 'pyarrow', 'jsonschema')}
    if int(versions['transformers'].split('.')[0]) != 4:
        raise RuntimeError('This pinned KaLM remote code requires Transformers 4; use the retriever environment')
    # Version metadata alone cannot detect incompatible native shared libraries.
    from sentence_transformers import SentenceTransformer  # noqa: F401
    from .retrieval import model_weight_identity
    if verify_model and verify_content_hashes and model_weight_identity(plan['parameters']['model_path']) != plan['parameters']['weight_identity']:
        raise ValueError('KaLM weights/config/tokenizer identity changed')
    if not Path(plan['parameters']['model_path']).is_dir():
        raise ValueError('KaLM model directory is missing')
    for entry in plan['sources']:
        release = Path(entry['release'])
        if sha(release / '_COMPLETE.json') != entry['completion_sha256']:
            raise ValueError(f'Changed source release: {release}')
        for name, key in [('manifest.json', 'manifest_sha256'), ('qrels.json', 'qrels_sha256'),
                          ('queries.jsonl', 'queries_sha256')]:
            if not (release / name).is_file():
                raise ValueError(f'Missing source metadata: {release / name}')
            if verify_content_hashes and sha(release / name) != entry[key]:
                raise ValueError(f'Changed source metadata: {release / name}')
    devices = []
    if gpu_count:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < gpu_count:
            raise RuntimeError(f'Requires {gpu_count} visible GPUs; found {torch.cuda.device_count()}')
        for index in range(gpu_count):
            prop = torch.cuda.get_device_properties(index)
            # An actual tiny kernel checks CUDA usability; this is not model inference.
            value = (torch.ones(2, device=f'cuda:{index}') + 1).sum().item()
            if value != 4:
                raise RuntimeError(f'CUDA kernel failed on GPU {index}')
            devices.append({'logical_index': index, 'name': prop.name, 'total_memory': prop.total_memory})
    disk_path = Path(plan['run_root'])
    while not disk_path.exists():
        disk_path = disk_path.parent
    return {'at': now(), 'sources': len(plan['sources']), 'versions': versions, 'devices': devices,
            'free_disk_bytes': shutil.disk_usage(disk_path).free,
            'content_hashes_verified': verify_content_hashes,
            'boundary': ('metadata/model identity verified; full release hashes checked per source before encoding'
                         if verify_content_hashes else
                         'structural checks only; source/model content hashes not recomputed')}


def cuda_slots(count, visible=None):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES') if visible is None else visible
    slots = visible.split(',') if visible and visible.strip() else [str(i) for i in range(count)]
    slots = [s.strip() for s in slots]
    if len(slots) < count or any(not s or s == '-1' for s in slots) or len(set(slots)) != len(slots):
        raise ValueError('CUDA_VISIBLE_DEVICES does not provide distinct requested GPU slots')
    return slots[:count]


def terminate_children(active):
    """Best-effort cleanup must visit every child even when another exits first."""
    for process, _, _, _ in list(active.values()):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process, _, _, log in list(active.values()):
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            log.close()


def _source_status(root, stage, **extra):
    write(root / 'status.json', {'at': now(), 'stage': stage, **extra}, replace=True)
    print(json.dumps({'source': root.name, 'stage': stage, **extra}, ensure_ascii=False), flush=True)


def _assert_export_retention(suite, entry):
    md = Path(suite) / 'metadata' / entry['task']
    manifest, audit = read(md / 'manifest.json'), read(md / 'audit.json')
    source = read(md / 'source_manifest.json')
    if (manifest.get('source_fingerprint') != entry['bundle_hash']
            or source.get('source_fingerprint') != entry['bundle_hash']
            or source.get('pipeline_manifest', {}).get('bundle_hash') != entry['bundle_hash']):
        raise ValueError('Export belongs to different source content')
    if (manifest['num_docs'], manifest['num_queries'], manifest['num_qrels']) != (
            entry['documents'], entry['queries'], entry['positive_pairs']):
        raise ValueError('DCI export changed retained counts; inspect exclusions before encoding')
    if audit['document_exclusions'] or audit['query_exclusions']:
        raise ValueError('DCI export introduced new exclusions; refusing partial-source pools')
    if read(md / 'original_qrels.json') != read(Path(entry['release']) / 'qrels.json'):
        raise ValueError('Export changed original query/document labels or grades')


def _phase_file(root, name):
    return root / 'phases' / (name + '.json')


def _validate_rank(ranked, suite, entry, params):
    from dci_bench.data.document_pools import build_source_identity, load_saved_rankings
    data = read(ranked)
    for key in ('trust_remote_code', 'query_prompt', 'document_prompt', 'max_length', 'weight_identity'):
        if data.get('model', {}).get(key) != params[key]:
            raise ValueError(f'Executed rank configuration mismatch: {key}')
    del data
    md = Path(suite) / 'metadata' / entry['task']
    mapping = read(md / 'source_doc_ids.json')
    labels = read(md / 'original_qrels.json')
    with (md / 'queries.jsonl').open(encoding='utf-8') as f:
        queries = [json.loads(line) for line in f if line.strip()]
    with (md / 'corpus_files.jsonl').open(encoding='utf-8') as f:
        hashes = {row['doc_id']: row['sha256'] for row in (json.loads(line) for line in f if line.strip())}
    ids = sorted(mapping.values())
    qids = [q['_id'] for q in queries]
    identity = build_source_identity(task=entry['task'], corpus_ids=ids, query_ids=qids,
                                     source_doc_ids=mapping, qrels=labels,
                                     query_records=queries, corpus_file_hashes=hashes)
    load_saved_rankings(ranked, task=entry['task'], source_identity=identity, corpus_ids=ids,
                        query_ids=qids, capacity=1000, qrels=labels)


def _resume_module(*, required=False):
    """Load resume helpers, failing closed for a plan that requires reuse."""
    try:
        from . import batch_resume
    except ModuleNotFoundError as exc:
        if exc.name in {f'{__package__}.batch_resume', 'dci_bench.data.pipeline.batch_resume'}:
            if required:
                raise RuntimeError('This reuse plan requires dci_bench.data.pipeline.batch_resume') from exc
            return None
        raise
    return batch_resume


def _import_source_receipts(entry, plan, source_root):
    """Import old phase receipts while the per-source lock is held, if enabled."""
    module = _resume_module(required=bool(plan.get('reuse')))
    if module is None:
        return None
    importer = getattr(module, 'import_source_receipts', None)
    if importer is None:
        raise AttributeError('batch_resume.import_source_receipts is required')
    return importer(entry, plan, source_root)


def _pool_target(entry, plan, source_root, capacity):
    """Choose a new pool destination or an immutable completed old destination."""
    module = _resume_module(required=bool(plan.get('reuse')))
    if module is not None:
        chooser = getattr(module, 'pool_target', None)
        if chooser is None:
            raise AttributeError('batch_resume.pool_target is required')
        target = chooser(entry, plan, source_root, capacity)
        if target is None:
            raise ValueError(f'batch_resume.pool_target returned no target for capacity {capacity}')
        return Path(target)
    return Path(source_root) / 'pools' / str(capacity)


def _load_export(entry, source_root, *, verify_document_task, plan=None):
    receipt = _phase_file(source_root, 'export')
    if not receipt.exists():
        raise ValueError('Export phase is incomplete; run export before this phase')
    export = read(receipt)
    try:
        suite = Path(export['suite'])
        completion = export['completion_sha256']
    except (KeyError, TypeError) as exc:
        raise ValueError('Export receipt is incomplete') from exc
    if sha(suite / 'metadata' / entry['task'] / '_COMPLETE.json') != completion:
        raise ValueError('Export completion changed')
    options = {} if content_hashes_enabled(plan or {}) else {'verify_content_hashes': False}
    verify_document_task(suite, entry['task'], io_workers=int(os.environ.get('DCI_KALM_IO_WORKERS', 4)),
                         **options)
    _assert_export_retention(suite, entry)
    return suite, export


def _load_rank(entry, plan, source_root, suite):
    receipt = _phase_file(source_root, 'rank')
    if not receipt.exists():
        raise ValueError('Rank phase is incomplete; run rank after export')
    ranking = read(receipt)
    try:
        ranked = Path(ranking['path'])
        expected_sha = ranking['sha256']
    except (KeyError, TypeError) as exc:
        raise ValueError('Rank receipt is incomplete') from exc
    if not _ranking_receipt_valid(ranking, plan):
        raise ValueError('Completed ranking changed; refusing to replace it')
    _validate_rank(ranked, suite, entry, plan['parameters'])
    return ranked, ranking


def _pool_manifest_check(entry, plan, suite, ranked, target, *, capacity=None):
    """Check the small manifest binding after a pool has been fully verified."""
    target = Path(target)
    task, params = entry['task'], plan['parameters']
    if not target.exists():
        raise ValueError(f'Pool target does not exist: {target}')
    md = target / 'metadata' / task
    manifest = read(md / 'manifest.json')
    expected = {
        'capacity': capacity,
        'seed': params['seed'],
        'num_queries': entry['queries'],
        'corpus_size': entry['documents'],
        'rankings_path': str(ranked),
        'source_manifest_sha256': sha(suite / 'metadata' / task / 'manifest.json'),
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError('Resumed pool source/ranking/parameter identity mismatch')
    return manifest, expected


def _validate_pool(entry, plan, suite, ranked, target, *, verify_document_pool_suite,
                   capacity=None):
    target = Path(target)
    options = {} if content_hashes_enabled(plan) else {'verify_content_hashes': False}
    verify_document_pool_suite(target, task=entry['task'], **options)
    return _pool_manifest_check(entry, plan, suite, ranked, target, capacity=capacity)


def _load_pool_receipt(entry, plan, source_root, suite, ranked, *, verify_document_pool_suite):
    receipt = _phase_file(source_root, 'pools')
    if not receipt.exists():
        return None
    value = read(receipt)
    binding = {'entry_hash': digest(entry), 'parameters_hash': digest(plan['parameters'])}
    if any(value.get(key) != expected for key, expected in binding.items()):
        raise ValueError('Completed pool receipt identity changed')
    rows = value.get('pools')
    if not isinstance(rows, list) or len(rows) != len(plan['parameters']['capacities']):
        raise ValueError('Completed pool receipt is incomplete')
    checked = []
    for capacity, row in zip(plan['parameters']['capacities'], rows):
        if row.get('capacity') != capacity:
            raise ValueError('Completed pool receipt capacity changed')
        target = Path(row['path'])
        manifest, expected = _validate_pool(entry, plan, suite, ranked, target,
                                            verify_document_pool_suite=verify_document_pool_suite,
                                            capacity=capacity)
        completion = target / 'metadata' / entry['task'] / '_COMPLETE.json'
        if sha(completion) != row.get('completion_sha256'):
            raise ValueError('Completed pool changed')
        checked.append({'capacity': capacity, 'path': str(target),
                        'completion_sha256': sha(completion)})
    return checked


def _complete_source(entry, plan, source_root, binding, ranking, pool_receipts):
    result = {'schema_version': SCHEMA, **binding, 'at': now(), 'source': entry['source_id'],
              'documents': entry['documents'], 'queries': entry['queries'],
              'ranking': ranking, 'pools': pool_receipts,
              'content_hashes_verified': content_hashes_enabled(plan)}
    completion = source_root / '_COMPLETE.json'
    if not completion.exists():
        write(completion, result)
    else:
        old = read(completion)
        if old.get('entry_hash') != binding['entry_hash'] or old.get('parameters_hash') != binding['parameters_hash']:
            raise ValueError('Completed source identity changed')
        if old.get('pools') != pool_receipts or old.get('ranking') != ranking:
            raise ValueError('Completed source receipt differs from verified phases')
        if old.get('content_hashes_verified', True) != content_hashes_enabled(plan):
            raise ValueError('Completed source validation policy differs; use a new run directory')
        result = old
    _source_status(source_root, 'complete', queries=entry['queries'])
    return result


def execute_source(entry, plan, run_root, *, device='cuda:0', phase='all'):
    """Execute one phase, or all phases for backwards-compatible callers.

    A phase receipt is written only after its complete artifact has passed the
    same identity checks used on resume.  Attempts and failures are retained.
    """
    if phase not in ('all', *PHASES):
        raise ValueError(f'Unknown source phase: {phase!r}')
    from .core import export_dci
    from .stages import rank_source
    from dci_bench.data.expansion import verify_document_task
    from dci_bench.data.document_pools import prepare_document_pools, verify_document_pool_suite
    source_root = Path(run_root) / 'sources' / entry['source_id']
    source_root.mkdir(parents=True, exist_ok=True)
    task, params = entry['task'], plan['parameters']
    verify_content_hashes = content_hashes_enabled(plan)
    validation_options = {} if verify_content_hashes else {'verify_content_hashes': False}
    binding = {'entry_hash': digest(entry), 'parameters_hash': digest(params)}
    with lock(source_root / '.lock'):
        bound = source_root / 'binding.json'
        if bound.exists():
            if read(bound) != binding:
                raise ValueError('Source resume identity changed; use a new run directory')
        else:
            write(bound, binding)
        try:
            # Export is the only phase that imports a reuse snapshot.  Other
            # direct phase entry points still fail closed when the pinned
            # helper is absent, rather than silently recomputing a phase from
            # incomplete receipts.
            if plan.get('reuse') and phase not in ('all', 'export'):
                _resume_module(required=True)
            if phase in ('all', 'export'):
                _source_status(source_root, 'export')
                _import_source_receipts(entry, plan, source_root)
                receipt = _phase_file(source_root, 'export')
                if receipt.exists():
                    suite, export = _load_export(entry, source_root, plan=plan,
                                                 verify_document_task=verify_document_task)
                else:
                    suite = source_root / 'attempts' / ('export-' + uuid.uuid4().hex) / 'suite'
                    export_options = dict(validation_options)
                    io_workers = int(os.environ.get('DCI_KALM_IO_WORKERS', 4))
                    if io_workers != 4:
                        export_options['io_workers'] = io_workers
                    export_dci(Path(entry['release']), suite, task, **export_options)
                    _assert_export_retention(suite, entry)
                    export = {'at': now(), 'suite': str(suite),
                              'completion_sha256': sha(suite / 'metadata' / task / '_COMPLETE.json')}
                    write(receipt, export)
                _assert_export_retention(suite, entry)
                if phase == 'export':
                    return {'schema_version': SCHEMA, **binding, 'source': entry['source_id'],
                            'phase': 'export', 'export': export}
            else:
                suite, export = _load_export(entry, source_root, plan=plan,
                                             verify_document_task=verify_document_task)

            if phase in ('all', 'rank'):
                _source_status(source_root, 'rank')
                receipt = _phase_file(source_root, 'rank')
                newly_ranked = not receipt.exists()
                if newly_ranked:
                    ranked = source_root / 'attempts' / ('rank-' + uuid.uuid4().hex) / 'rankings.json'
                    rank_source(ranked, suite=suite, task=task, cache_dir=source_root / 'embedding_cache',
                                model_path=params['model_path'], device=device, batch_size=params['batch_size'],
                                max_length=params['max_length'], top_k=params['top_k'],
                                query_prompt=params['query_prompt'], document_prompt=params['document_prompt'],
                                cache_shard_size=params['cache_shard_size'], search_device=device,
                                search_query_batch_size=params['search_query_batch_size'],
                                **({} if verify_content_hashes else {
                                    'verify_content_hashes': False,
                                    'weight_identity': params['weight_identity']}))
                    # Verify the actual executed settings before publishing receipt.
                    _validate_rank(ranked, suite, entry, params)
                    ranking = {'at': now(), 'path': str(ranked),
                               'sha256': sha(ranked) if verify_content_hashes else None,
                               'bytes': ranked.stat().st_size,
                               'content_hashes_verified': verify_content_hashes}
                    write(receipt, ranking)
                else:
                    ranked, ranking = _load_rank(entry, plan, source_root, suite)
                if phase == 'rank':
                    return {'schema_version': SCHEMA, **binding, 'source': entry['source_id'],
                            'phase': 'rank', 'export': export, 'ranking': ranking}
            else:
                ranked, ranking = _load_rank(entry, plan, source_root, suite)

            _source_status(source_root, 'pools')
            pool_receipts = _load_pool_receipt(entry, plan, source_root, suite, ranked,
                                               verify_document_pool_suite=verify_document_pool_suite)
            if pool_receipts is None:
                pool_receipts = []
                for capacity in params['capacities']:
                    _source_status(source_root, 'pools', capacity=capacity)
                    target = _pool_target(entry, plan, source_root, capacity)
                    if target.exists():
                        manifest, expected = _validate_pool(
                            entry, plan, suite, ranked, target,
                            verify_document_pool_suite=verify_document_pool_suite,
                            capacity=capacity)
                    else:
                        prepare_document_pools(suite, target, task, capacity=capacity,
                                               seed=params['seed'], rankings_path=ranked,
                                               **validation_options)
                        # prepare already verifies all pool files before its atomic publish.
                        # The builder already performed the expensive full suite
                        # verification; only inspect the published manifest here.
                        manifest, expected = _pool_manifest_check(
                            entry, plan, suite, ranked, target, capacity=capacity)
                    completion = target / 'metadata' / task / '_COMPLETE.json'
                    if not target.exists() or not completion.is_file():
                        raise ValueError('Pool preparation did not publish a complete target')
                    pool_receipts.append({'capacity': capacity, 'path': str(target),
                                          'completion_sha256': sha(completion)})
                write(_phase_file(source_root, 'pools'),
                      {'at': now(), **binding, 'source': entry['source_id'],
                       'pools': pool_receipts})
            result = _complete_source(entry, plan, source_root, binding, ranking, pool_receipts)
            return {'schema_version': SCHEMA, **binding, 'source': entry['source_id'],
                    'phase': 'pools' if phase == 'pools' else 'all', **result}
        except BaseException as exc:
            failure = {'at': now(), 'phase': phase, 'error': type(exc).__name__, 'message': str(exc),
                       'traceback': traceback.format_exc()}
            write(source_root / 'failures' / (uuid.uuid4().hex + '.json'), failure)
            _source_status(source_root, 'failed', phase=phase, error=str(exc))
            raise


def _source_complete_valid(entry, plan, root):
    marker = Path(root) / 'sources' / entry['source_id'] / '_COMPLETE.json'
    if not marker.is_file():
        return False
    try:
        value = read(marker)
        if (value.get('entry_hash') != digest(entry)
                or value.get('parameters_hash') != digest(plan['parameters'])):
            return False
        capacities = plan.get('parameters', {}).get('capacities')
        if not isinstance(capacities, list) or not capacities:
            return False
        if value.get('schema_version') != SCHEMA:
            return False
        if value.get('content_hashes_verified', True) != content_hashes_enabled(plan):
            return False
        ranking = value.get('ranking')
        if not isinstance(ranking, dict) or not ranking.get('path'):
            return False
        ranked = Path(ranking['path'])
        if not _ranking_receipt_valid(ranking, plan):
            return False
        pools = value.get('pools')
        if not isinstance(pools, list) or len(pools) != len(capacities):
            return False
        for capacity, pool in zip(capacities, pools):
            if not isinstance(pool, dict) or pool.get('capacity') != capacity:
                return False
            target = Path(pool.get('path', ''))
            completion = target / 'metadata' / entry['task'] / '_COMPLETE.json'
            if not completion.is_file() or sha(completion) != pool.get('completion_sha256'):
                return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _source_next_phase(entry, plan, root):
    """Return the first phase that can make progress for a source.

    Receipts are completion boundaries, so a valid rank receipt never causes
    another GPU process to start.  Any malformed receipt is handed to the
    corresponding phase for strict validation and preserved failure reporting.
    """
    source_root = Path(root) / 'sources' / entry['source_id']
    # A prior source marker proves the ranking/pools closure, but restart still
    # queues a CPU pools worker once so the current run validates every pool
    # receipt before it publishes its root completion marker.
    if _source_complete_valid(entry, plan, root):
        return 'pools'
    if _phase_file(source_root, 'pools').is_file():
        return 'pools'
    if _phase_file(source_root, 'rank').is_file():
        return 'pools'
    # A reuse import is pinned before receipts are copied.  If an earlier
    # process stopped after export was copied, replay the CPU import so a
    # completed old rank is never accidentally recomputed on a GPU.
    if plan.get('reuse') and (source_root / 'reuse.json').is_file():
        try:
            reuse_snapshot = read(source_root / 'reuse.json')
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            reuse_snapshot = {}
        if 'rank' in reuse_snapshot.get('phases', {}):
            return 'export'
    if _phase_file(source_root, 'export').is_file():
        return 'rank'
    return 'export'


def _worker_env(root, entry, phase, slot, *, cpu_threads=CPU_THREAD_LIMIT, io_workers=4):
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false')
    env['DCI_KALM_IO_WORKERS'] = str(io_workers)
    if phase == 'rank':
        if not slot:
            raise ValueError('GPU rank worker requires an assigned slot')
        env['CUDA_VISIBLE_DEVICES'] = slot
    else:
        # CPU stages must never inherit the node-wide visible GPU list.
        env['CUDA_VISIBLE_DEVICES'] = ''
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            try:
                inherited = int(env.get(name, cpu_threads))
            except (TypeError, ValueError):
                inherited = cpu_threads
            env[name] = str(max(1, min(inherited, cpu_threads)))
    # Prevent optional shared dynamic-module writes by assigning each worker a cache.
    env['HF_MODULES_CACHE'] = str(Path(root) / 'runtime_cache' / entry['source_id'] / 'modules')
    return env


def _phase_success(entry, plan, root, phase):
    source_root = Path(root) / 'sources' / entry['source_id']
    try:
        if phase == 'export':
            value = read(_phase_file(source_root, 'export'))
            marker = Path(value['suite']) / 'metadata' / entry['task'] / '_COMPLETE.json'
            return marker.is_file() and sha(marker) == value['completion_sha256']
        if phase == 'rank':
            value = read(_phase_file(source_root, 'rank'))
            path = Path(value['path'])
            return _ranking_receipt_valid(value, plan)
        # A pools receipt is published before the source marker, so it cannot
        # alone advance the parent to final completion.
        return _source_complete_valid(entry, plan, root)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _phase_queue_key(item):
    # Preserve the existing largest-source-first scheduling policy within each queue.
    return (item.get('documents', 0), item.get('queries', 0), item['source_id'])


def run(bundle, *, gpu_count=4, export_workers=DEFAULT_EXPORT_WORKERS,
        pool_workers=DEFAULT_POOL_WORKERS, export_prefetch=DEFAULT_EXPORT_PREFETCH,
        cpu_threads=CPU_THREAD_LIMIT, io_workers=4):
    """Run independent CPU export, four-GPU rank, and CPU pool queues."""
    bundle = Path(bundle).absolute()
    plan = load_plan(bundle)
    if gpu_count != 4:
        raise ValueError('This batch is contracted for one node with four visible GPUs')
    if export_workers < 1 or pool_workers < 1 or export_prefetch < 1:
        raise ValueError('stage worker counts and export prefetch must be positive')
    if export_prefetch < export_workers:
        raise ValueError('export prefetch must be at least export workers')
    if type(cpu_threads) is not int or cpu_threads < 1 or type(io_workers) is not int or not 1 <= io_workers <= 256:
        raise ValueError('cpu threads must be positive and io workers must be 1..256')
    for key in ('RANK', 'SLURM_PROCID'):
        if os.environ.get(key, '0') != '0':
            raise RuntimeError('Launch once per node; do not wrap this launcher in torchrun or four srun tasks')
    root = Path(plan['run_root'])
    reuse = plan.get('reuse')
    if reuse and _paths_overlap(root, Path(reuse['run_root'])):
        raise ValueError('Reuse run root overlaps the new run root; refusing to touch the old run tree')
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / '.run.lock'):
        identity = {'plan_sha256': sha(bundle / 'plan.json'), 'bundle': str(bundle)}
        if (root / 'run.json').exists():
            if read(root / 'run.json') != identity:
                raise ValueError('Run directory belongs to another frozen plan')
        else:
            write(root / 'run.json', identity)
        checked = preflight(bundle, gpu_count=gpu_count)
        checked = {**checked, 'scheduling': {
            'gpu_count': gpu_count,
            'export_workers': export_workers,
            'pool_workers': pool_workers,
            'export_prefetch': export_prefetch,
            'cpu_thread_limit': cpu_threads,
            'io_workers': io_workers,
        }}
        write(root / 'preflight' / (uuid.uuid4().hex + '.json'), checked)
        print(json.dumps({'preflight': checked, 'run_root': str(root)}, ensure_ascii=False), flush=True)

        ordered = sorted(plan['sources'], key=_phase_queue_key, reverse=True)
        states = {}
        ready = {phase: deque() for phase in PHASES}
        export_backlog = deque()
        export_waiting = deque()
        for entry in ordered:
            phase = _source_next_phase(entry, plan, root)
            states[entry['source_id']] = {'entry': entry, 'phase': phase, 'failed': False}
            if phase == 'export':
                export_backlog.append(entry)
            elif phase is not None:
                ready[phase].append(entry)
        while export_backlog and len(export_waiting) + len(ready['rank']) < export_prefetch:
            export_waiting.append(export_backlog.popleft())

        free = deque(cuda_slots(gpu_count))
        active, active_phase, outcomes = {}, {}, {}
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)

        def refill_export_waiting():
            # Keep a bounded CPU prefetch window so a slow exporter cannot stage the
            # complete corpus set before rank and pool queues make progress.
            active_exports = sum(1 for phase in active_phase.values() if phase == 'export')
            while export_backlog and (len(export_waiting) + active_exports + len(ready['rank'])
                                      < export_prefetch):
                export_waiting.append(export_backlog.popleft())

        def mark_failed(entry, phase, code, log_name, error=None):
            source_id = entry['source_id']
            states[source_id]['failed'] = True
            states[source_id]['phase'] = None
            value = {'exit_code': code, 'log': log_name}
            if error:
                value['error'] = str(error)
            outcomes.setdefault(source_id, {})[phase] = value

        def launch(entry, phase, slot=None):
            log = (logs / (entry['source_id'] + '-' + phase + '-' + uuid.uuid4().hex + '.log')).open('x')
            env = _worker_env(root, entry, phase, slot, cpu_threads=cpu_threads, io_workers=io_workers)
            command = [sys.executable, str(bundle / 'code/scripts/run_kalm_pool_batch.py'),
                       '_worker', '--bundle', str(bundle), '--phase', phase,
                       '--source', entry['source_id']]
            try:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except BaseException as exc:
                log.close()
                mark_failed(entry, phase, 127, log.name, exc)
                return False
            active[process.pid] = (process, slot, entry, log)
            active_phase[process.pid] = phase
            print(json.dumps({'started': entry['source_id'], 'phase': phase,
                              'gpu': slot, 'pid': process.pid, 'log': log.name}), flush=True)
            return True

        previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        stop_requested = False

        def interrupted(_signum, _frame):
            nonlocal stop_requested
            stop_requested = True

        for s in previous_handlers:
            signal.signal(s, interrupted)
        try:
            while active or export_backlog or export_waiting or any(ready.values()):
                if stop_requested:
                    raise KeyboardInterrupt('Interrupted; preserving partial work')
                refill_export_waiting()
                while (export_waiting
                       and sum(1 for phase in active_phase.values() if phase == 'export') < export_workers
                       and (sum(1 for phase in active_phase.values() if phase == 'export')
                            + len(ready['rank']) < export_prefetch)):
                    entry = export_waiting.popleft()
                    launch(entry, 'export')
                while ready['rank'] and free:
                    entry = ready['rank'].popleft()
                    slot = free.popleft()
                    if not launch(entry, 'rank', slot):
                        free.appendleft(slot)
                while ready['pools'] and sum(1 for phase in active_phase.values() if phase == 'pools') < pool_workers:
                    entry = ready['pools'].popleft()
                    launch(entry, 'pools')

                for pid, (process, slot, entry, log) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    phase = active_phase.pop(pid)
                    log.close()
                    del active[pid]
                    if slot is not None:
                        free.append(slot)
                    outcomes.setdefault(entry['source_id'], {})[phase] = {
                        'exit_code': code, 'log': log.name}
                    if code == 0 and _phase_success(entry, plan, root, phase):
                        if phase == 'export':
                            # Resume import may have brought in a completed rank;
                            # inspect receipts again before allocating a GPU.
                            next_phase = _source_next_phase(entry, plan, root)
                        else:
                            next_phase = {'rank': 'pools', 'pools': None}[phase]
                        states[entry['source_id']]['phase'] = next_phase
                        if next_phase == 'rank':
                            ready['rank'].append(entry)
                        elif next_phase == 'pools':
                            ready['pools'].append(entry)
                    else:
                        states[entry['source_id']]['failed'] = True
                        states[entry['source_id']]['phase'] = None
                        if code == 0:
                            outcomes[entry['source_id']][phase]['error'] = 'phase receipt missing'
                    write(root / 'summary.json', {'at': now(), 'total': len(plan['sources']),
                                                  'finished': outcomes, 'active': len(active)}, replace=True)
                    print(json.dumps({'finished': entry['source_id'], 'phase': phase,
                                      'exit_code': code}), flush=True)
                if active:
                    time.sleep(1)
        except BaseException:
            terminate_children(active)
            raise
        finally:
            for s, handler in previous_handlers.items():
                signal.signal(s, handler)

        failures = sorted(source_id for source_id, state in states.items() if state['failed'])
        if failures:
            raise RuntimeError(f'{len(failures)} source(s) failed; preserved outputs; rerun the same command: {failures}')
        sources, pool_count = {}, 0
        for entry in plan['sources']:
            marker = root / 'sources' / entry['source_id'] / '_COMPLETE.json'
            value = read(marker)
            if not _source_complete_valid(entry, plan, root):
                raise ValueError('Final source identity mismatch')
            sources[entry['source_id']] = sha(marker)
            pool_count += len(value.get('pools', []))
        complete = {'schema_version': SCHEMA, **identity, 'sources': sources,
                    'source_views': len(sources), 'pools': pool_count,
                    'content_hashes_verified': content_hashes_enabled(plan),
                    'boundary': 'KaLM candidate pools completed; no retrieval-agent benchmark implied'}
        if not (root / '_COMPLETE.json').exists():
            write(root / '_COMPLETE.json', complete)
        elif read(root / '_COMPLETE.json') != complete:
            raise ValueError('Batch completion identity changed')
        return complete


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('prepare')
    c.add_argument('--bundle', type=Path, required=True)
    c.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    c.add_argument('--run-root', type=Path)
    c.add_argument('--batch-size', type=int, default=None)
    c.add_argument('--resume-bundle', type=Path)
    c.add_argument('--skip-content-hashes', action='store_true',
                   help='Reuse declared content digests; do not rehash source/model/result bytes at runtime')
    for name in ('preflight', 'run', '_worker'):
        c = commands.add_parser(name)
        c.add_argument('--bundle', type=Path, required=True)
        if name == 'preflight':
            c.add_argument('--gpus', type=int, default=0)
        elif name == 'run':
            c.add_argument('--gpus', type=int, default=4)
            c.add_argument('--export-workers', type=int, default=DEFAULT_EXPORT_WORKERS)
            c.add_argument('--pool-workers', type=int, default=DEFAULT_POOL_WORKERS)
            c.add_argument('--export-prefetch', type=int, default=DEFAULT_EXPORT_PREFETCH)
            c.add_argument('--cpu-threads', type=int, default=CPU_THREAD_LIMIT)
            c.add_argument('--io-workers', type=int, default=4)
        else:
            c.add_argument('--source', required=True)
            c.add_argument('--phase', choices=('all', *PHASES), default='all')
    args = p.parse_args(argv)
    if args.command == 'prepare':
        if args.batch_size is not None and args.batch_size < 1:
            raise ValueError('batch size must be positive')
        result = prepare(args.bundle, args.matrix, run_root=args.run_root, batch_size=args.batch_size,
                         resume_bundle=args.resume_bundle,
                         verify_content_hashes=not args.skip_content_hashes)
    elif args.command == 'preflight':
        result = preflight(args.bundle, gpu_count=args.gpus)
    elif args.command == 'run':
        result = run(args.bundle, gpu_count=args.gpus, export_workers=args.export_workers,
                     pool_workers=args.pool_workers, export_prefetch=args.export_prefetch,
                     cpu_threads=args.cpu_threads, io_workers=args.io_workers)
    else:
        plan = load_plan(args.bundle)
        entry = next(e for e in plan['sources'] if e['source_id'] == args.source)
        if sha(Path(entry['release']) / '_COMPLETE.json') != entry['completion_sha256']:
            raise ValueError('Source completion changed before worker launch')
        device = 'cuda:0' if args.phase in ('all', 'rank') else 'cpu'
        result = execute_source(entry, plan, plan['run_root'], device=device, phase=args.phase)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
