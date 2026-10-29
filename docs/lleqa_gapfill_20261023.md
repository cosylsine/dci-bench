# LLeQA French legal gap-fill probe

This bounded probe targets the official LLeQA source from Maastricht Law &
Technology.  The planned source is native French Belgian legal QA/IR data:
the complete statutory article corpus, train/dev/test expert questions, and
the source `article_ids` associations.  The planned DCI evaluation split is
`test`; train and dev remain raw source material.  Associations would be
retained as grade 1 with `association_kind=evidence_qa`,
`max_positive_per_query=null`, and `exclude_review_required=false`.

Run it with:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_lleqa_gapfill_20261023.py \
  --config configs/data_pipeline/lleqa_gapfill_20261023.json
```

The official dataset card reports 27,941 articles and 1,868 questions
(train 1,472, dev 201, test 195).  These are expected card counts only; the
probe does not treat them as observed release counts.

The official GitHub repository was fixed at commit
`01ae7a96c58c1e34a9786bbecf3524b268e1b019`.  Its codeload archive is 944,383
bytes with SHA-256
`0093853bc1a78066c1f5c1ecbae72570160dcc545c684d320eb839d9ffe72dc9`; archive
inspection found code and documentation but no corpus/question data files.
The HF dataset card was fixed at revision
`f24d30f33e51354a1acbc2f5d58fa3353f845242`, 29,441 bytes, SHA-256
`ccdbf8ce08efdc20eaf9e034e5ba6c2753f1dd945f12660aa8e15e52667eef0a`.

The corpus and all three official question URLs returned HTTP 401 with the
message that access is restricted and authentication is required.  The card
requires a signed data-use agreement.  ModelScope candidates for the
original namespace and MTEB mirrors returned HTTP 404.  Therefore no raw
corpus, queries, qrels, synthetic associations, or `labels_ready` release is
claimed.  Exact URLs, response status, body hashes/previews, fixed metadata
hashes, and the absent release path are in:

`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round3/lleqa/research.json`

and the raw attempt manifest under
`raw_data/gapfill_legal_lleqa_fr/source_attempt_manifest.json`.  The receipt
is intentionally blocked and can be rerun after authorized HF access without
changing the source identity or label policy.

Official references:

- [LLeQA HF dataset card](https://huggingface.co/datasets/maastrichtlawtech/lleqa)
- [LLeQA official repository](https://github.com/maastrichtlawtech/lleqa)
- [LLeQA paper](https://arxiv.org/abs/2309.17050)
