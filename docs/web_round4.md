# Web round 4: bounded multilingual mMARCO

This round seals twelve new non-English web cells from the official MTEB
`MMarcoRetrievalMultilingual` dev release at dataset revision
`44480423f56a42b948edd899c707e640b8e46f19`.  The MTEB task code is pinned at
`2ad77827480c853ee9518eb207a244f13f8a0966` and identifies a bounded
100,000-candidate sample.  The pinned language parquet files contain 106,813
corpus rows after positive closure, 6,980 queries, and 7,437 positive qrel
pairs per language.  This is a bounded official benchmark view and does not
claim the full parent mMARCO corpus.

The selected languages are `ar`, `de`, `es`, `fr`, `hi`, `id`, `it`, `ja`,
`nl`, `pt`, `ru`, and `vi`.  The source family is machine-translated MS MARCO
(`translation_status=translated`); no translation was performed locally.  The
source license is Apache-2.0.  MIRACL is excluded because it is an
encyclopedia/Wikipedia source owned by the separate encyclopedia worker.
JaCWIR remains a documented candidate; Japanese mMARCO already supplies this
round's bounded Japanese web cell.

Each source's raw directory is
`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/r4_web_mmarco_<lang>/`.
It contains the three original parquet downloads, `source_manifest.json` with
the dataset/file revisions, byte counts, URLs, and SHA-256 values,
`canonical_input_ascii_v1/`, and `conversion_audit.json`.  The audit compares
every parquet row one-for-one with canonical JSONL, checks all qrel IDs and
grades against actual documents, and compares the released IDs/text/metadata
and qrels back to canonical input.

The immutable official v1 releases are under
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/r4_web_mmarco_<lang>/v1/`.
Seven are directly `labels_ready` (`ar`, `fr`, `it`, `nl`, `pt`, `ru`, `vi`).
Five official v1 releases retain exact duplicate rows that trigger the
pipeline review state (`de`, `es`, `hi`, `id`, `ja`); none was discarded.

For those five languages, the script also publishes an explicit child release
under `r4_web_mmarco_<lang>_exact_alias/v1/`.  Only exact
`(title,text,language)` duplicates are aliased.  The child keeps every query
and rechecks every parent qrel grade through the canonical ID; conflicting
mapped grades fail closed.  Alias inventories and audits are stored as
`derived_aliases.jsonl` and `derived_alias_audit.json` beside each raw source.
The official v1 parent remains immutable and is the full source-ID/text/grade
comparison surface.  The five derived views are `labels_ready` with,
respectively, 28, 34, 27, 23, and 44 aliased duplicate rows; each retains
6,980 queries and 7,437 positive pairs.

The stable combined receipt is
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round4/web/research_final.json`.
It records one selected `labels_ready` view per language, both parent and
derived paths where applicable, all raw/release audit paths and hashes, the
translation and bounded-scope declarations, and the preserved per-run receipt
history.  The older failure receipt at `research.json` and all timestamped
successful receipts are retained.

Run acquisition or reuse with:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_web_round4.py \
  --config configs/data_pipeline/web_round4.json
```

The script refuses to overwrite immutable artifacts, reuses matching pinned
downloads/releases, preserves failed staging artifacts, and applies
`max_positive_per_query=null`, `drop_near_duplicates=false`, and
`exclude_review_required=false`.  To seal the combined receipt from complete
per-source receipts after a bounded rerun:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_web_round4.py \
  --config configs/data_pipeline/web_round4.json --finalize
```

Verification performed for the final receipt: Python compilation, pinned raw
SHA/byte checks, raw-to-canonical full comparisons, `verify_release` on all
twelve official parents and all five alias children, qrel resolution, and
full selected-view checks.  No model, GPU, cascade, or benchmark execution
was performed.
