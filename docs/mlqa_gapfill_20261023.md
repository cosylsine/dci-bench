# MLQA multilingual gap-fill (2026-10-23)

This release uses only the official `MLQA_V1.zip` archive from the
[facebookresearch/MLQA repository](https://github.com/facebookresearch/MLQA):

- Archive: https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip
- README: https://github.com/facebookresearch/MLQA/blob/main/README.md
- Paper: https://arxiv.org/abs/1910.07475 and https://arxiv.org/html/1910.07475v3
- Archive revision: `MLQA_V1.zip#sha256:246e8089933d13007fe80684d5c5c0713d6834cf8b3b4a0ec7c66f0a0d2baac8`
- Archive size: `75,719,050` bytes; HTTP `Last-Modified: 2019-10-16T21:33:54Z`
- License: CC-BY-SA 3.0 (the archive is derived from Wikipedia paragraphs)

The paper describes English questions crowd-sourced on aligned English
paragraphs, followed by target-language question translation by professional
translators and target-context answer-span annotation. Therefore the releases
mark English questions `non_translated` and Arabic/German/Spanish/Hindi/
Vietnamese/Simplified Chinese questions `human_translated`. Contexts are
same-language Wikipedia paragraphs; article histories are not audited, so the
releases do not blanket-claim every context is independently native. All seven
views belong to one source family (`MLQA`) and use an explicit parallel query
family keyed by the official question ID.

## Published views

Each view has the complete same-language **dev + test paragraph corpus** (the
corpus unit is a paragraph, not a long document), but only official
same-language **test questions**. The qrel is a score-1 association from each
question to its source paragraph, derived from the official extractive QA
answer association. This is a bounded QA-derived retrieval label and is not an
exhaustive IR judgment over the corpus.

| view | corpus paragraphs | test queries | qrel pairs | valid evidence spans | duplicate context rows merged | release state |
|---|---:|---:|---:|---:|---:|---|
| `en` | 10,894 | 11,590 | 11,590 | 11,590 | 0 | `labels_ready` |
| `ar` | 5,085 | 5,335 | 5,335 | 5,335 | 0 | `labels_ready` |
| `de` | 4,509 | 4,517 | 4,517 | 4,517 | 0 | `labels_ready` |
| `es` | 5,215 | 5,253 | 5,253 | 5,253 | 0 | `labels_ready` |
| `hi` | 4,524 | 4,918 | 4,918 | 4,918 | 0 | `labels_ready` |
| `vi` | 5,246 | 5,495 | 5,495 | 5,495 | 0 | `labels_ready` |
| `zh` | 4,989 | 5,137 | 5,137 | 5,137 | 0 | `labels_ready` |

The README table reports 5,254 Spanish test instances, while the direct
same-language archive member contains 5,253 QAs. The release keeps the direct
archive count; the other six same-language counts match the README table.

All views have `max_positive_per_query=1`; every official test query has a qrel.
The source answer offsets were all valid in this archive (`0` malformed or
missing spans). The release audit reported `pass`, zero issues, zero warnings,
zero review rows, zero full duplicate groups, and zero equivalent-unlabelled
document groups for each view.

The archive has no paragraph IDs. The adapter therefore assigns stable IDs as
`sha256(language + NUL + exact_context)` (first 32 hex characters) and stores
every original split/article/paragraph position in `metadata.source_aliases`.
Exact duplicate contexts would be merged losslessly; none occurred across the
same-language dev+test collections in this archive.

## Paths and reproducibility

The original archive is preserved at:

`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/gapfill_mlqa_archive/source_inputs/MLQA_V1.zip`

Per-language extracted source files, canonical input, conversion counts, and
span audits are under:

`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/gapfill_mlqa_<lang>/`

Standard immutable releases are:

`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_mlqa_<lang>/v1/`

The source/release inventory and official research evidence are recorded in:

`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round2/mlqa/research.json`

The acquisition script is
[`scripts/acquire_mlqa_gapfill_20261023.py`](../scripts/acquire_mlqa_gapfill_20261023.py).
It uses the pipeline's direct opener (`ProxyHandler({})`), timeout 30 seconds,
and at most one retry. A failed download remains as `.partial`; no other
dataset or auxiliary MLQA translate package is fetched.

```bash
PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
cd /mnt/afs/250010100/zxp/dci-bench
$PY scripts/acquire_mlqa_gapfill_20261023.py --languages en ar de es hi vi zh
for lang in en ar de es hi vi zh; do
  $PY scripts/data_pipeline.py verify \
    /mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_mlqa_${lang}/v1
done
```

No model, GPU, full-corpus evaluation, or dependency installation is part of
this acquisition boundary.
