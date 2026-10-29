# BEIR and software gap-fill (2026-10-23 round 3)

This acquisition fills two matrix cells with complete bounded sources:

| source | domain | language | source scope | expected D/Q/pairs | release |
| --- | --- | --- | ---: | ---: | --- |
| `gapfill_beir_cqadupstack_android` v1 | software | en | complete Android Stack Exchange title-only sentence view from the pinned MTEB ModelScope mirror | 22,998 / 699 / 1,696 | `eval_data/pipeline_v1/gapfill_beir_cqadupstack_android/v1` |
| `gapfill_beir_cqadupstack_android` v2-fulltext | software | en | complete official BEIR Android corpus with title+body text, native test queries, and test qrels | 22,998 / 699 / 1,696 | `eval_data/pipeline_v1/gapfill_beir_cqadupstack_android/v2-fulltext` |
| `gapfill_beir_quora` | general_qa | en | complete BEIR Quora corpus with test queries and test qrels | 522,931 / 10,000 / 15,675 | `eval_data/pipeline_v1/gapfill_beir_quora/v1` |

The v1 row is materialized by [scripts/acquire_beir_gapfill_20261023.py](../scripts/acquire_beir_gapfill_20261023.py) from the fixed ModelScope revision `c48cee56fcdafa78803dfb8674c53b3a9752cef9`, file `data/android-00000-of-00001.json.gz`, 555,190 bytes, SHA-256 `b4e7db4c1c3879939404677876a363af2f14e0c94e21ad4455b49f790cd6bddf`. The verbatim gzip is kept under a new timestamped directory in `raw_data/gapfill_beir_cqadupstack_android/`; the previous zero-byte `.partial` files remain untouched. The gzip contains aligned `corpus`, `queries`, and `relevant_docs` arrays. Every source ID, sentence, query, and relevant document relation is retained, but its sentence view contains the question title only; v1 must therefore be interpreted as a title-only view rather than complete original BEIR body text. The canonical qrels map every source `relevant_docs` relation to the official BEIR grade `1`; no labels are invented and no positives are capped.

The v2-fulltext row is acquired by the same script with `--android-fulltext --release-name v2-fulltext`. The official 5,343,728,040-byte ZIP is not downloaded as a whole. Its 6,209-byte central directory is fetched by Range, then the complete Android `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv` members are fetched independently (97,987,241 compressed bytes; 280,652,465 uncompressed bytes). Each member is checked against the pinned central-directory size/CRC and SHA-256 before release publication. The resulting corpus retains the original title and body text (`corpus_text_chars=12,450,637`; title characters=1,203,321); the official query text field and all qrels are preserved.

The MTEB v1 mirror exposes a sentence-only view, so its Android canonical records retain an empty `title` and record `title_field=not_available_in_mteb_mirror_sentence_view` in metadata. This is a documented field limitation. The v2-fulltext member extraction is the complete text-bearing Android release; its provenance records the official archive URL, central-directory range, member offsets, CRCs, byte counts, and hashes. Both variants record `translation_status=native`, `source_family`, `source_scope`, and `task_origin` in their canonical records, derivation manifests, source specifications, and round3 research receipts.

The Android `relevant_docs` array was checked as an aligned list of document-ID lists before derivation. It contains 1,696 unique query/document pairs; every pair maps to grade `1`, every document ID exists in the 22,998-document corpus, and no positive cap was applied. The largest row is the source query `19971` with 262 positives, which is retained in full. The official Hugging Face qrels preview has the same first rows and the same beginning of the `19971` sequence; the BEIR `ir_datasets` catalog independently reports 1,696 Android qrels, all with relevance `1`. This is a source property, not a parser expansion or overflow workaround.

The fulltext release remains `review_required` under the existing audit because review-required equivalent-document cases are retained by policy; the source qrels and positive grades are unchanged. The title-only v1 release remains separately sealed and is not rewritten when v2-fulltext is created.

The Quora row reuses the sealed round2 raw snapshot and v1 release only after running `verify_release` and comparing every raw query ID/text, corpus ID/title/text, and qrel pair/grade to the release. The old release state is preserved as reported by its manifest (`review_required` because the configured policy keeps review-required equivalent-document cases). The policy for both cells is `max_positive_per_query=null` and `exclude_review_required=false`; all native source qrels and positive grades remain present.

## Reproduction and evidence

Run the bounded CPU-only acquisition from the repository root:

```bash
python scripts/acquire_beir_gapfill_20261023.py \
  --config configs/data_pipeline/beir_gapfill_20261023.json \
  --research-root /mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round3/beir
```

To reproduce the fulltext Android variant, use the bounded member path explicitly:

```bash
python scripts/acquire_beir_gapfill_20261023.py \
  --config configs/data_pipeline/beir_gapfill_20261023.json \
  --research-root /mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round3/beir \
  --source gapfill_beir_cqadupstack_android \
  --android-fulltext --release-name v2-fulltext
```

The sealed evidence is under `eval_data/source_research_20261023_round3/beir/`:

- `research.json` records the exact source URLs, fixed revisions, bytes and hashes, acquisition/reuse status, preserved partials, counts, release state, bundle hash, and raw-to-release exactness checks.
- `gapfill_beir_cqadupstack_android.receipt.json` records the ModelScope raw stage, derivation manifest, raw snapshot, and release.
- `gapfill_beir_cqadupstack_android.v2-fulltext.research.json` and `gapfill_beir_cqadupstack_android.v2-fulltext.receipt.json` record the bounded official ZIP member acquisition, CRC/hash checks, fulltext raw snapshot, and v2-fulltext release.
- `gapfill_beir_quora.receipt.json` records the reused raw snapshot/release and the full raw-to-release comparison.
- `raw_data/gapfill_beir_cqadupstack_android/<acquisition-timestamp>/source_inputs/android-00000-of-00001.json.gz` is the verbatim ModelScope input; `data/derivation_manifest.json` records the derived-file hashes and counts.

The official references are the [BEIR catalog](https://github.com/beir-cellar/beir/blob/main/README.md), [BEIR paper](https://arxiv.org/abs/2104.08663), [BEIR CQADupStack dataset card](https://huggingface.co/datasets/BeIR/cqadupstack), [official Android qrels](https://huggingface.co/datasets/BeIR/cqadupstack-qrels/blob/main/android/test.tsv), [official CQADupStack archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip), [ModelScope MTEB mirror](https://www.modelscope.cn/datasets/MTEB/cqadupstack-retrieval), and the [official Quora archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip).

The full CQADupStack archive is deliberately excluded because the official archive is 5,343,728,040 bytes, above the 300 MB per-source and 600 MB total limits. Direct Hugging Face Android attempts from round2 are preserved as failed receipts and are not retried. No GPU, model, cascade, or package installation is part of this data-only acquisition.
