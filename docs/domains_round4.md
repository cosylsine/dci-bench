# Domains round 4 acquisition

This round adds 11 ready domain cells and three sealed cells that remain explicitly review-required across the requested scientific/medical/finance/argument/legal scope. The ready set is within the requested 8–12 cell target and uses only official MTEB task definitions with pinned source revisions. No model, GPU, ranking, candidate pool, or cascade work was run.

The research receipt is [research.json](/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round4/domains/research.json). Each acquired source has a raw directory under `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/r4_domains_*`; each published release is under `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/r4_domains_*/v1`.

## Ready releases

`D` is the verified release document count, `Q` is the verified test-query count, and `pairs` is the positive-qrel count. The MultilingualNanoSciFact shards retain their published 50-query test split; no queries were added to meet a corpus preference.

| Cell | Source family | D | Q | pairs | Translation / license | State |
| --- | --- | ---: | ---: | ---: | --- | --- |
| scientific / nl | BEIR-NL SciFact | 5,183 | 300 | 339 | translated; CC-BY-4.0 | labels_ready |
| scientific / pl | BEIR-PL SciFact | 5,183 | 300 | 339 | translated; CC-BY-SA-4.0 | labels_ready |
| scientific / vi | VN-MTEB SciFact | 5,183 | 134 | 155 | translated and LM-verified; CC-BY-SA-4.0 | labels_ready |
| scientific / de | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |
| scientific / es | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |
| scientific / fr | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |
| scientific / it | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |
| scientific / ja | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |
| scientific / ko | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |
| scientific / no | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |
| scientific / sv | MultilingualNanoSciFact | 2,919 | 50 | 56 | translated; CC-BY-4.0 | labels_ready |

The official task code and dataset card URLs, source revisions, mirror revisions, raw/release paths, completion hashes, translation metadata, and per-source counts are recorded in `research.json` and `configs/data_pipeline/domains_round4.json`.

## Sealed review releases

These releases satisfy the bounded source and release checks but have exact-text alias risks retained under the required `exclude_review_required=false` policy.

| Cell | Source family | D | Q | pairs | Alias audit | State |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| medical / nl | BEIR-NL NFCorpus | 3,633 | 323 | 12,334 | 215 positive/unlabelled alias rows; 40 exact-text groups | review_required |
| medical / pl | BEIR-PL NFCorpus | 3,633 | 323 | 12,334 | 215 positive/unlabelled alias rows; 40 exact-text groups | review_required |
| legal / nl | MTEB-NL LegalQA | 30,803 | 102 | 157 | 7 positive/unlabelled alias rows; 631 exact-text groups | review_required |

Documents and qrels are unchanged in these releases. `source_inputs/alias_audit.json` records byte-exact body aliases, and the pipeline audit records each review-required query. No near-duplicate merge or grade invention was performed.

## Held candidates and failures

The raw source snapshot and failure evidence are retained for every failed candidate.

* NL FiQA and PL FiQA each contain 38 empty source document bodies. Document `117276` is a positive for query `5206`; both are held pending review in `source_inputs/empty_document_audit.json`. The raw body, IDs, and qrel grade remain preserved.
* NL ArguAna, PL ArguAna, and VN ArguAna each have five positive test qrels whose document IDs are absent from the pinned corpus: `test-education-ufsdfkhbwu-con03b`, `test-free-speech-debate-yfsdfkhbwu-con03b`, `test-politics-dhwem-pro06b`, `test-science-sghwbdgmo-con03b`, and `test-society-asfhwapg-con04b`. The source was left pending; no document was synthesized or remapped.
* VN FiQA has duplicate query ID rows in the published query parquet (`6612`, `5155`, `10628`, `5906`, and `3594`, each byte-identical). The raw parquet is retained and the release is held rather than dropping query rows.

Failure records are in the `failures` array of `research.json`; source trees/manifests and partial retry artifacts remain in each corresponding `raw_data/r4_domains_*` directory.

## Raw/release and provenance checks

The acquisition script retains the complete pinned source files, including source train/dev qrels where published, while the canonical retrieval release uses the official test qrel split. Every acquired source has:

* `mirror_tree.json` from the pinned source tree and `source_manifest.json` with per-file byte count and SHA-256;
* `source_inputs/raw_test_qrels.jsonl`, preserving every test qrel row and original grade, including non-positive rows;
* `source_inputs/raw_release_compare.json`, comparing raw and canonical IDs, text digest, query counts, qrel counts, and explicit split exclusions;
* `source_inputs/alias_audit.json` and `source_inputs/empty_document_audit.json`; and
* a verified `_COMPLETE.json` inventory in the release directory.

The pipeline policy is fixed to `max_positive_per_query=null`, `drop_near_duplicates=false`, and `exclude_review_required=false`. Positive qrels in published retrieval bundles are source grades greater than zero; raw non-positive rows remain in the audit and are never turned into positives. Titles, document bodies, query text, original IDs, and source qrel grades are retained in the raw snapshot; the canonical input preserves the complete selected test-qrel records and the comparison audit records any split-only query exclusions. Exact duplicates are surfaced as reversible aliases; near duplicates are not merged.

## Reproduction and verification

From this isolated worktree:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python -m py_compile scripts/acquire_domains_round4.py
python -m json.tool configs/data_pipeline/domains_round4.json >/dev/null
ARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_domains_round4.py \
  --config configs/data_pipeline/domains_round4.json
```

The full command returns status 2 because the six held candidates are intentionally reported as failures; it still seals and verifies every successful source. A targeted retry accepts source IDs after `--source`; the research writer merges that receipt with prior source entries and preserves the previous research snapshot with a `.previous-*` suffix before publishing the new receipt.

The final successful run verified 14 source entries: 11 `labels_ready` and three `review_required`; six candidates remain held with explicit reasons above. All release manifests report `validation_boundary: data prepared and audited; no model benchmark implied`.
