import json
import pytest
from scripts.prepare_gapfill_20261031 import filter_query_self_matches


def test_arguana_self_exclusion_keeps_1000_candidates_and_unranked_positive_scores(tmp_path):
    raw, output, mapping = (tmp_path / name for name in ('raw.json', 'filtered.json', 'ids.json'))
    ranking = ['q1'] + [f'd{i}' for i in range(1000)]
    payload = {'rankings': {'q1': ranking},
               'scores': {'q1': {**dict.fromkeys(ranking, 0.5), 'unranked-gold': 0.1}},
               'source_identity': {'queries_hash': 'retained'}}
    raw.write_text(json.dumps(payload))
    mapping.write_text(json.dumps({'self': 'q1'}))
    receipt = filter_query_self_matches(raw, output, mapping)
    filtered = json.loads(output.read_text())
    assert filtered['rankings']['q1'] == ranking[1:]
    assert 'q1' not in filtered['scores']['q1']
    assert filtered['scores']['q1']['unranked-gold'] == 0.1
    assert filtered['source_identity'] == payload['source_identity']
    assert receipt['removed'] == {'q1': 'q1'}
    assert json.loads(raw.read_text()) == payload


def test_annotation_self_filter_never_removes_a_positive(tmp_path):
    raw, output, mapping = (tmp_path / name for name in ('raw.json', 'filtered.json', 'ids.json'))
    ranking = ['canonical-claim1'] + [f'd{i}' for i in range(1000)]
    raw.write_text(json.dumps({'rankings': {'query-claim1': ranking},
                              'scores': {'query-claim1': dict.fromkeys(ranking, 0.5)}}))
    mapping.write_text(json.dumps({'self': 'canonical-claim1'}))
    kwargs = {'self_documents': {'query-claim1': 'canonical-claim1'}}
    with pytest.raises(ValueError, match='annotated positive'):
        filter_query_self_matches(raw, output, mapping, **kwargs, qrels={'query-claim1': {'self': 1}})
    assert not output.exists()
    filter_query_self_matches(raw, output, mapping, **kwargs, qrels={'query-claim1': {'d1': 1}})
    assert json.loads(output.read_text())['rankings']['query-claim1'] == ranking[1:]
