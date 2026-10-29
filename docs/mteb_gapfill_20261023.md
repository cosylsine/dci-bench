# MTEB round3 legal gap-fill

This run restores the MTEB acquisition path and adds two complete legal
retrieval sources.  The three round2 releases are treated as immutable,
read-only inputs.  New source bytes come from pinned ModelScope mirrors; the
official MTEB dataset revision and task implementation remain the benchmark
identity.  ModelScope file hashes are checked against the ModelScope tree, and
no Hugging Face byte equivalence is claimed.

The reproducible command is:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_mteb_gapfill_20261023.py \
  --config configs/data_pipeline/mteb_gapfill_20261023.json
```

The script uses `max_positive_per_query=null`,
`exclude_review_required=false`, and `drop_near_duplicates=false`.  It retains
all source-positive pairs and all source grades.  Canonical input JSONL is
written with `ensure_ascii=true` so decoded source text cannot introduce JSONL
line separators.

## Sealed releases

| source | language/domain | D | Q | positive pairs | state | release |
|---|---:|---:|---:|---:|---|---|
| CmedqaRetrieval | zh / medical | 100001 | 3999 | 7449 | labels_ready | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_mteb_cmedqa_retrieval/v1` |
| DuRetrieval | zh / web | 100001 | 2000 | 9839 | labels_ready | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_mteb_duretrieval/v1` |
| GermanDPR | de / sealed web; report correction encyclopedia | 2876 | 1025 | 1025 | labels_ready | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_mteb_germandpr/v1` |
| LeCaRDv2 | zh / legal | 3795 | 159 | 3896 | labels_ready | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_legal_lecardv2/v1` |
| GerDaLIRSmall | de / legal | 9969 | 12234 | 14320 | labels_ready | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_legal_gerdalirsmall/v1` |

For each new source the complete raw definition is retained under
`raw_data/gapfill_legal_*`, with the ASCII canonical inputs in
`canonical_input_ascii_v1`.  The final receipt records each raw file URL,
mirror revision, expected and observed byte count, SHA-256, canonical hashes,
release hash, and the official MTEB/source links.

LeCaRDv2 uses ModelScope revision `8abd8019ed605a8c01467a1c65d446862312f2f5`
and totals 157,717,617 source bytes.  GerDaLIRSmall uses revision
`ffd75874a55b29236c220ab1307449c9e5532414` and totals 228,713,699 source
bytes.  The combined 386,431,316 bytes are within the 350,000,000-byte
per-source and 800,000,000-byte aggregate budgets.  No multi-gigabyte source
was downloaded.

The raw qrel audits retain every source row.  LeCaRDv2 has 3,896 rows, all
grade 1, across 159 query IDs; GerDaLIRSmall has 14,320 rows, all grade 1,
across 12,234 query IDs.  Neither source has duplicate pairs, non-positive
grades, or qrels pointing to unknown query/document IDs.  The earlier
research metadata used in round2 reported 22,203 GerDaLIRSmall qrels. Parent
integration checked the current official Hugging Face viewer, which shows
14.3k test rows, consistent with the 14,320 rows in the pinned raw file.
22,203 equals documents plus queries (9,969 + 12,234); this suggests a previous
count-category mistake, not missing qrels. The exact origin of that old
estimate is not established. No rows were silently dropped. The
positive-only pipeline contract is explicit: all 14,320 source grade-1 rows
are retained with `max_positive_per_query=null`.

Official source references:

- [LeCaRDv2 MTEB card](https://huggingface.co/datasets/mteb/LeCaRDv2), [MTEB task code](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/zho/le_ca_r_dv2_retrieval.py), [upstream LeCaRDv2](https://github.com/THUIR/LeCaRDv2), [paper](https://arxiv.org/abs/2310.17609)
- [GerDaLIRSmall MTEB card](https://huggingface.co/datasets/mteb/GerDaLIRSmall), [MTEB task code](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/deu/ger_da_lir_small_retrieval.py), [upstream GerDaLIR](https://github.com/lavis-nlp/GerDaLIR), [paper](https://aclanthology.org/2021.nllp-1.13/)

GermanDPR remains sealed with its round2 source domain value `web`; the
official card describes German Wikipedia contexts, so the report exposes
`encyclopedia` as a correction without modifying the old release.  The
translation status is `native` for all five MTEB sources; no translation or
synthetic labels were introduced. LLeQA is tracked in a separate follow-up
acquisition scope and is not part of these five source releases.

The final research receipt is
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round3/legal/research.json`.
Older preflight, failure, and per-source receipts in that directory are
preserved and are not release inputs.
