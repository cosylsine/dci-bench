# Encyclopedia round four acquisition

This round pins the official MTEB `MIRACLRetrievalHardNegatives.v2` development release for the nine delegated absent languages: Bengali (`bn`), Persian (`fa`), Finnish (`fi`), French (`fr`), Indonesian (`id`), Korean (`ko`), Swahili (`sw`), Telugu (`te`), and Yoruba (`yo`). The MTEB dataset revision is `d7d94fa4b946cec4a27c84653aa0cf6b33f74a3c`; the MTEB task-code tree is pinned to `2ad77827480c853ee9518eb207a244f13f8a0966`. The task uses the native same-language MIRACL Wikipedia-passage retrieval prompt and `dev` split. Official receipts are the [MTEB task code](https://github.com/embeddings-benchmark/mteb/blob/2ad77827480c853ee9518eb207a244f13f8a0966/mteb/tasks/retrieval/multilingual/miracl_retrieval.py), [MTEB dataset card](https://huggingface.co/datasets/mteb/MIRACLRetrievalHardNegatives), and [MIRACL project](https://project-miracl.github.io/).

The three pinned Parquet inputs for every language are retained under `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/r4_encyclopedia_<lang>/source_inputs/`. Their byte sizes and SHA-256 values are recorded in `source_manifest.json`, `source_counts.json`, and `research.json`. The raw qrels Parquet and `qrels_raw.jsonl` preserve every source grade, including score zero. Canonical qrels retain every positive source grade without capping positives or changing grades.

The pipeline release is an explicitly named derived view. It removes only an unlabelled document when `(title, text, language)` is byte-identical to another source document, keeps every document referenced by a positive qrel, and records every removed ID in `derived_aliases.jsonl` with its canonical ID and text digest. Near duplicates are never merged. The full source corpus remains available in the raw Parquet and the original YO release is preserved at `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/r4_encyclopedia_yo/v1_full_review_required/`; the active derived releases are under `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/r4_encyclopedia_<lang>/v1/`.

| language | raw D | derived D | Q | positive pairs | raw qrel rows | zero rows | missing zero IDs | exact aliases | review items | release state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bn | 297,265 | 296,878 | 411 | 863 | 4,206 | 3,343 | 0 | 387 | 3 | review_required |
| fa | 133,596 | 133,272 | 632 | 1,314 | 6,571 | 5,257 | 1,261 | 324 | 11 | review_required |
| fi | 194,415 | 194,191 | 1,000 | 1,918 | 9,436 | 7,518 | 1,211 | 224 | 2 | review_required |
| fr | 75,357 | 74,809 | 343 | 731 | 3,429 | 2,698 | 442 | 548 | 1 | review_required |
| id | 168,173 | 167,582 | 960 | 3,088 | 9,668 | 6,580 | 685 | 591 | 13 | review_required |
| ko | 43,293 | 43,236 | 213 | 547 | 3,057 | 2,510 | 362 | 57 | 1 | review_required |
| sw | 131,924 | 131,815 | 482 | 910 | 5,092 | 4,182 | 0 | 109 | 21 | review_required |
| te | 101,961 | 101,809 | 828 | 854 | 1,606 | 752 | 143 | 152 | 1 | review_required |
| yo | 49,043 | 48,441 | 119 | 144 | 1,188 | 1,044 | 0 | 602 | 1 | review_required |

All nine releases are valid sealed bundles, but none is marked ready. Remaining review items concern exact body matches with different titles. A parent audit of the original qrels distinguishes absent/unjudged counterparts from explicitly judged score-zero counterparts; an absent entry is not a negative label. Both cases remain pending, and no differing-grade conflict was resolved by inventing a precedence rule. See `eval_data/source_research_20261023_round4/integration/miracl_review_label_audit.json`. For YO the representative review is query `70580#0`, positive `3295#45`, and unlabelled body-equivalent `3544#44`; the titles differ, so the explicit title/text/language alias rule does not merge them. The 602-document difference between YO raw and derived views is therefore retained as a documented derived-view change, not a query or positive-label drop.

For FA, FI, FR, ID, KO, and TE, the official source qrels also contain score-zero IDs outside the bounded MTEB pooled corpus. Those rows remain in the raw Parquet and `qrels_raw.jsonl`; all positive qrel IDs resolve. `research.json` reports this distinction as `all_positive_qrel_ids_resolve: true`, `all_qrel_ids_resolve: false`, and `raw_zero_qrel_ids_resolve: false`. This is an explicit source boundary rather than a hidden exclusion. The aggregate checks also report `all_releases_sealed: true`, `full_source_counts_compare: true`, `all_exact_aliases_explicit: true`, `no_query_drops: true`, and `no_hidden_exclusions: true`; `sealed_release_state_labels_ready` is false.

The reproducible acquisition command is:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  /mnt/afs/250010100/zxp/dci-bench/scripts/acquire_encyclopedia_round4.py \
  --config /mnt/afs/250010100/zxp/dci-bench/configs/data_pipeline/encyclopedia_round4.json
```

The completed verification command was the same script with `--verify-only`. No GPU, cascade, translation, query generation, or model evaluation was run. Earlier failed/intermediate manifests, canonical files, and research receipts remain beside the active raw snapshots with `failed`, `prior`, or `review_required` names.
