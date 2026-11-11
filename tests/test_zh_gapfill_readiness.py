"""Source row counts must survive the benchmark's actual query deduplication."""
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import prepare_zh_gapfill_20261102 as readiness


def test_duplicate_query_count_cannot_pass_source_readiness():
    bundle = {
        'provenance': {'translation_status': 'native'},
        'documents': [{'text': f'独立文档 {index}'} for index in range(1000)],
        'queries': [
            {'_id': str(index), 'text': f'查询 {max(1, index)}',
             'metadata': {'group_id': f'group-{index}'}}
            for index in range(60)
        ],
        'qrels': {str(index): {'doc1': 1} for index in range(60)},
    }
    manifest = {'state': 'labels_ready', 'automatic_labels': False,
                'num_documents': 1000, 'num_queries': 60}
    with (
        patch.object(readiness, 'SOURCES', {'software_zh': 'fixture'}),
        patch.object(readiness, 'verify_release', return_value=manifest),
        patch.object(readiness, 'load_bundle', return_value=bundle),
        patch.object(readiness, 'export_dci', side_effect=AssertionError('export before admission')),
        pytest.raises(ValueError, match='fewer than 60 queries after benchmark eligibility'),
    ):
        readiness.export_sources(Path('/unused-fixture'))
