# STARD native Chinese legal source

`prepare_stard_pos10.py` prepares the STARD source at the pinned repository
commit `bae10e1f358685446c3af4b15e7b94724f83bc9e`:

- Repository: <https://github.com/oneal2000/STARD/tree/bae10e1f358685446c3af4b15e7b94724f83bc9e>
- Corpus: <https://raw.githubusercontent.com/oneal2000/STARD/bae10e1f358685446c3af4b15e7b94724f83bc9e/data/corpus.jsonl>
- Queries and original `match_id` qrels: <https://raw.githubusercontent.com/oneal2000/STARD/bae10e1f358685446c3af4b15e7b94724f83bc9e/data/queries.json>
- License: <https://raw.githubusercontent.com/oneal2000/STARD/bae10e1f358685446c3af4b15e7b94724f83bc9e/LICENSE>
- Data statement: <https://raw.githubusercontent.com/oneal2000/STARD/bae10e1f358685446c3af4b15e7b94724f83bc9e/Data%20Statement.md>

The raw snapshot is stored below `raw/stard/` by the existing pinned connector.
Its `source_manifest.json` records each downloaded file's byte count, SHA-256,
and actual pinned URL. The cleaned release is
`releases/gapfill_stard_legal_zh`; the DCI export is
`run/source` under the selected data root. Existing snapshots, releases, and
exports are verified and reused; a changed or incomplete destination raises
instead of being overwritten.

Run on the preparation node with:

```bash
PYTHONPATH=/mnt/afs/250010100/zxp/dci-bench \
python /mnt/afs/250010100/zxp/dci-bench/scripts/prepare_stard_pos10.py \
  --data-root /mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_legal_pos10_20261116_v1
```

After publication, a read-only integrity check is available:

```bash
python /mnt/afs/250010100/zxp/dci-bench/scripts/prepare_stard_pos10.py \
  --data-root /mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_legal_pos10_20261116_v1 \
  --check-only
```

`--check-only` requires the existing raw manifest, release, export, and
receipt; it does not download or write them.

The pinned files contain 55,348 statute-article documents and 1,543 native
Chinese legal consultation queries with 2,717 original positive links. Query
651 has 23 original positives and is excluded as a whole. The remaining
1,542 queries retain all 2,694 original positives; no positive is clipped or
generated. The parser fails if these pinned source counts or corpus IDs do not
match. The release policy also rejects any accidental positive count above 10,
checks exact document-body duplicates, and applies the existing whole-query
exclusion for an unlabelled exact-text equivalent. Near duplicates are kept.

Each retained query keeps its source integer ID as a string, has binary qrels
with score `1`, and uses the namespaced original query ID (`stard-query:<id>`)
as `group_id`; the unmodified source ID is retained only as `raw_query_id`.
This leaves
equivalent-group isolation conservative and permits the standard deterministic
10-query dev / 50-query test sampler without connecting queries through shared
statute articles. The query metadata preserves original `match_name` and
`相关法规` keys for audit; these are host-side and are not model-visible query
text.

STARD's task is native Chinese legal document retrieval: retrieve the Chinese
statute or judicial-interpretation articles matched to a real legal-consultation
question. The repository describes human legal annotation of the original
links, and the source is not translated or model-labelled. MIT covers the
STARD repository; the repository's underlying statute texts retain their own
source status.

This script performs CPU acquisition, parsing, quality publication, and
`export_dci` verification only. It does not build retrieval pools or run a
model benchmark.
