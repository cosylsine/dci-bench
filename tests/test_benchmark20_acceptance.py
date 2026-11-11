"""CPU contracts for appending BCP without changing accepted selections."""
import copy
from unittest.mock import patch

import pytest

from scripts.accept_benchmark20_20261103 import assert_selection_stable, validate_pools


@pytest.mark.parametrize('field', [None, 'source_id', 'query_ids', 'query_groups'])
def test_old_19_selection_must_be_identical(field):
    tasks = {f'task_{i}': {'status': 'ready', 'source_id': str(i), 'source_task': str(i),
                         'query_ids': {'dev': [f'd{i}'], 'test': [f't{i}'], 'smoke': [f'd{i}']},
                         'query_groups': {f'd{i}': f'dg{i}', f't{i}': f'tg{i}'}}
             for i in range(19)}
    before = {'tasks': tasks}
    after = copy.deepcopy(before)
    after['tasks']['web_en'] = {'status': 'ready'}
    if field:
        after['tasks']['task_0'][field] = 'changed'
        with pytest.raises(ValueError, match='selection changed'):
            assert_selection_stable(before, after)
    else:
        result = assert_selection_stable(before, after)
        assert result['stable_old_task_count'] == 19
        assert result['changed'] == {}


@pytest.mark.parametrize('failure', [None, 'not_nested', 'missing_positive', 'group_leak'])
@pytest.mark.parametrize('task_id', ['web_en', 'argumentation_en'])
def test_final_pool_invariants(tmp_path, failure, task_id):
    ids = [f'q{i}' for i in range(60)]
    mapping = {f'd{i}': f'original{i}' for i in range(1001)}
    task = {'query_ids': {'dev': ids[:10], 'test': ids[10:], 'smoke': ids[:1]},
            'query_groups': {qid: qid for qid in ids},
            'pools': {tier: {'suite': f'tasks/{task_id}/{tier}'} for tier in ('lite', 'full')}}
    values = {}
    for tier, capacity in (('lite', 200), ('full', 1000)):
        pool = tmp_path / task['pools'][tier]['suite']
        (pool / 'workspaces' / task_id).mkdir(parents=True)
        if task_id != 'web_en':
            for base in (tmp_path, tmp_path / 'old'):
                metadata = base / task['pools'][tier]['suite'] / 'metadata' / task_id
                metadata.mkdir(parents=True)
                for name in ('queries.jsonl', 'qrels.json', 'original_qrels.json', 'candidate_pools.json',
                             'source_doc_ids.json', 'doc_id_files.json', 'file_inventory.json',
                             'source_identity.json', '_COMPLETE.json'):
                    (metadata / name).write_text('{}\n')
        candidates = [f'd{i}' for i in range(capacity)]
        if tier == 'lite' and failure == 'not_nested':
            candidates[-1] = 'd1000'
        if tier == 'lite' and failure == 'missing_positive':
            candidates[0] = 'd1000'
        values[tier] = {'candidate_pools': {'pools': {qid: {'candidate_doc_ids': candidates} for qid in ids}},
                        'queries': [{'_id': qid, 'text': qid} for qid in ids],
                        'qrels': {qid: {'d0': 1} for qid in ids},
                        'original_qrels': {qid: {'original0': 1} for qid in ids},
                        'source_doc_ids': mapping}
    if failure == 'group_leak':
        task['query_groups'][ids[-1]] = ids[0]
    with patch('scripts.accept_benchmark20_20261103._load_pool_metadata',
               side_effect=lambda path, _task: values[path.name]):
        if failure:
            with pytest.raises(ValueError):
                validate_pools(tmp_path, {'tasks': {task_id: task}}, tmp_path / 'old')
        else:
            result = validate_pools(tmp_path, {'tasks': {task_id: task}}, tmp_path / 'old')
            assert result['pool_records_checked'] == 120
            assert result['global_group_leakage'] is False
            assert result['stable_old_metadata_files'] == (18 if task_id != 'web_en' else 0)
