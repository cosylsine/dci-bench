# Multilingual expansion research and releases (2026-10-23)

Final integration selects `multilingual_korquad_ko/expansion-v2-fullqrels`:
10,645 documents and all 5,774 dev queries/qrels are retained with
`exclude_review_required=false`; 55 equivalent-text cases remain pending review.
The earlier 5,719-query release below is preserved as history and is not selected
in the final matrix. The final receipt is
`source_research_20261023/korquad_fullqrels/research.json`.

This bounded acquisition uses the direct-opener path in
`scripts/acquire_multilingual_20261023.py`. It does not use the configured
`127.0.0.1:27897` proxy, model inference, GPU, or shared caches. Raw inputs
are immutable under
`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/multilingual_*` and
clean releases are under
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/multilingual_*/expansion-v1`.

The completed native retrieval release is ViHealthQA (Vietnamese):

- Source: [official BEIR archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/vihealthqa.zip), last-modified 2022-06-30; archive SHA-256 is
  `65329d2c4f1154f4e6314c2c9f3eee0fe0682526aefbb4b36855bb9683181bf2`.
- Raw archive: `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/multilingual_vihealthqa_vi/source_inputs/vihealthqa.zip` (2,236,748 bytes).
- Release: `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/multilingual_vihealthqa_vi/expansion-v1`.
- The archive contains 9,896 Vietnamese answer passages, 7,009 queries, and train/dev/test qrels. The selected `test` qrels have 2,013 official query IDs and 5,019 score-1 pairs. The pipeline audit found 83 qrels whose document IDs are absent from the 9,896-row corpus, so the verified release retains 1,930 queries and 4,806 valid positive pairs. The raw archive preserves every original ID and score; no missing document or negative was fabricated.
- Documents are answer passages from the BEIR `corpus.jsonl`, not full articles. Questions and passages are Vietnamese/native; this is an original retrieval source, not a translation or QA-derived conversion. The archive/paper do not state a redistribution license, so the research record leaves the license as unspecified.

The configuration records research candidates such as GermanQuAD, DuRetrieval,
MLDR, MIRACL, and Mr.TyDi. An earlier handoff attempt also downloaded and
verified CMRC2018 under `multilingual_cmrc2018_zh`; those immutable artifacts
are preserved for the parent task, while the active configuration no longer
selects that source to avoid duplicating the parent-owned `qa_cmrc2018_zh`
release.
Candidates above the 150 MB per-source limit or blocked by the direct network
remain `researched_not_downloaded`; they are not silently truncated to 1,001
documents. CMRC2018 is marked native and `qa-derived` when consumed through
the existing `squad` adapter; its answer spans become score-1 seed qrels and
unjudged passages are not negatives.

The completed non-Chinese GitHub source is KorQuAD v1.0 (native Korean,
`qa-derived`):

- Pinned source revision: `korquad/korquad.github.io@61d7d6682948375dd8f06ef27e788fd19982af38`.
  The [official KorQuAD 1.0 page](https://korquad.github.io/category/1.0_KOR.html)
  reports 10,645 paragraphs, 60,407 train QA rows, 5,774 dev QA rows, the
  unreleased test split, and CC BY-ND 2.0 KR.
  The official GitHub Pages files are [train JSON](https://korquad.github.io/dataset/KorQuAD_v1.0_train.json)
  (38,527,475 bytes, SHA-256
  `40d5115879a701751781df721d901abfa736d8db5f89000f2619433f39bf2dd2`) and
  [public dev JSON](https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json)
  (3,881,058 bytes, SHA-256
  `25ffeb51e6c51ec02c071b60a10188e10005c144110f0d876b26079d80a35bdf`).
  The official page states CC BY-ND 2.0 KR.
- Raw originals: `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/multilingual_korquad_ko/source_inputs/`.
  They contain 9,681 train paragraphs/60,407 train QA rows and 964 dev
  paragraphs/5,774 dev QA rows. The canonical input preserves all 10,645
  paragraphs, all 5,774 public-dev query IDs, and all 5,774 dev qrels; raw
  originals preserve the train QA IDs as well.
- Release: `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/multilingual_korquad_ko/expansion-v1`.
  It contains 10,645 Korean Wikipedia paragraphs, 5,719 retained public-dev
  queries, and 5,719 score-1 source-paragraph pairs. `verify_release` passes;
  the release state is `review_required` because the generic quality policy
  excluded 55 public-dev queries whose positive paragraph has an exact
  duplicate elsewhere. Raw and canonical inputs retain those rows for review.
- Answer spans were checked against their original Unicode-codepoint offsets;
  `canonical_input/span_audit.json` records zero malformed spans. Evidence is
  retained only for valid spans; qrels come from the question-to-source-
  paragraph relation and do not invent negative labels. Documents are
  paragraphs, not whole Wikipedia articles. The public hidden test JSON is not
  included by the official repository, so the release uses the official public
  dev split as held-out evaluation queries and records that limitation.

Reproduce or verify the ViHealthQA release with:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_multilingual_20261023.py \
  --config configs/data_pipeline/multilingual_20261023.json \
  --source multilingual_vihealthqa_vi

/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/data_pipeline.py verify \
  /mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/multilingual_vihealthqa_vi/expansion-v1

/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/data_pipeline.py verify \
  /mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/multilingual_korquad_ko/expansion-v1

/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_multilingual_20261023.py \
  --config configs/data_pipeline/multilingual_20261023.json \
  --source multilingual_vihealthqa_vi multilingual_korquad_ko \
  --research-output /mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023/multilingual/research_final.json
```

The machine-readable source matrix and immutable receipts are in
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023/multilingual/research.json`.
The final handoff summary, with `status=fetched_verified` for the two
standard-verified releases and `researched_not_downloaded` for blocked or
over-budget candidates, is in
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023/multilingual/research_final.json`.
