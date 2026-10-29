import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/derive_quora_exact_aliases_20261023.py'
spec = importlib.util.spec_from_file_location('quora_aliases', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExactAliasTests(unittest.TestCase):
    def docs(self):
        return [{'_id': 'b', 'title': '', 'text': 'same', 'language': 'en'},
                {'_id': 'a', 'title': '', 'text': 'same', 'language': 'en'},
                {'_id': 'c', 'title': 'different', 'text': 'same', 'language': 'en'}]

    def test_all_positive_grades_survive_and_titles_are_not_ignored(self):
        docs, qrels, aliases, merged = module.exact_alias_view(self.docs(), {'q1': {'b': 2}, 'q2': {'c': 3, 'a': 1}})
        self.assertEqual(len(docs), 2)
        self.assertEqual(qrels, {'q1': {'a': 2}, 'q2': {'c': 3, 'a': 1}})
        self.assertEqual(aliases, {'a': 'a', 'b': 'a', 'c': 'c'})
        self.assertEqual(merged, [{'canonical_id': 'a', 'original_document_ids': ['a', 'b']}])

    def test_conflicting_native_grades_are_not_aggregated(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting grades'):
            module.exact_alias_view(self.docs(), {'q1': {'a': 1, 'b': 2}})

    def test_whitespace_differences_are_not_merged(self):
        docs = self.docs()[:2]
        docs[0]['text'] += ' '
        kept, _, aliases, _ = module.exact_alias_view(docs, {'q1': {'a': 1, 'b': 1}})
        self.assertEqual(len(kept), 2)
        self.assertNotEqual(aliases['a'], aliases['b'])


if __name__ == '__main__':
    unittest.main()
