import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('rawqa_round4', Path(__file__).resolve().parents[1] / 'scripts/acquire_rawqa_round4.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RawQAConversionTest(unittest.TestCase):
    def source(self, kind='hc3_human'):
        return {'kind': kind, 'filename': 'source.jsonl', 'language': 'zh',
                'family': 'test', 'split': 'unsplit', 'label_basis': 'published source association',
                'limitations': []}

    def test_missing_answers_and_synthetic_answers_do_not_create_labels(self):
        docs, queries, qrels, audit = module.convert([
            {'question': 'q1', 'human_answers': ['same', None], 'chatgpt_answers': ['synthetic']},
            {'question': 'q2', 'human_answers': ['same'], 'chatgpt_answers': ['synthetic2']},
            {'question': 'q3', 'human_answers': [None], 'chatgpt_answers': ['synthetic3']},
        ], self.source())
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(docs[0]['metadata']['source_aliases']), 2)
        self.assertEqual(qrels[queries[0]['_id']], qrels[queries[1]['_id']])
        self.assertEqual(audit['counts']['queries_without_positive'], 1)
        self.assertEqual(audit['counts']['invalid_answer_slots'], 2)
        self.assertEqual(audit['counts']['chatgpt_answer_slots_outside_human_only_scope'], 3)

    def test_whitespace_and_unicode_are_preserved(self):
        docs, queries, qrels, audit = module.convert([
            {'question': 'q', 'human_answers': ['答案', '答案 ', '答\u2028案'], 'chatgpt_answers': []},
        ], self.source())
        self.assertEqual({d['text'] for d in docs}, {'答案', '答案 ', '答\u2028案'})
        self.assertEqual(len(qrels[queries[0]['_id']]), 3)

    def test_contradictory_reranking_labels_fail_closed(self):
        with self.assertRaises(module.SourceLabelConflict) as caught:
            module.convert([{'query': 'q', 'positive': ['same'], 'negative': ['same']}], self.source('reranking_union'))
        self.assertEqual(len(caught.exception.conflicts), 1)


if __name__ == '__main__':
    unittest.main()
