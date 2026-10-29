# Domain source releases (2026-10-23)

Final integration also completed FiQA after a bounded range download and the
JSONL Unicode-separator reader fix: `domain_fiqa_test/expansion-v3-jsonl`,
57,600 documents / 647 queries / 1,700 positive pairs, `labels_ready`.
The earlier failure records below are preserved. See
[the final expansion record](data_pipeline_expansion_20261023.md) and
`source_research_20261023/fiqa_repaired/research.json` for the selected version.

This expansion adds three independently acquired domain releases through the existing BEIR/QASPER connectors. The raw snapshots and derived bundles are immutable outputs under the paths below. The qrels remain source judgments; a qrel row is not an exhaustive statement that every other corpus document is negative.

| source | official input and revision | release | verified counts | state |
| --- | --- | --- | --- | --- |
| NFCorpus | [BEIR archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip), `beir-nfcorpus-20210115`, CC-BY-SA-4.0 ([dataset card](https://huggingface.co/datasets/BeIR/nfcorpus)) | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/domain_nfcorpus_test/expansion-v1-fullqrels-noequiv` | 3,633 documents; 323 test queries; 12,334 positive pairs | `review_required` |
| ArguAna | [BEIR archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip), `beir-arguana-20210123`, original [ArguAna home](https://argumentation.bplaced.net/arguana/) terms need review | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/domain_arguana_test/expansion-v1-fullqrels-noequiv` | 8,674 documents; 1,401 valid test queries/qrels; 1,401 positive pairs | `review_required` |
| QASPER full corpus view | official [train-dev v0.3 archive](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz), `v0.3-train-dev`, CC-BY-4.0 as declared by the existing source catalog | `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/domain_qasper_dev_fullcorpus/expansion-v1` | 1,169 documents (888 train + 281 dev); 465 retained dev queries; 465 qrels; 694 evidence spans | `labels_ready` |

The NFCorpus archive is 2,448,432 bytes, MD5 `a89dba18a62ef92f7d323ec890a0d38d`, SHA-256 `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`. Its raw snapshot is `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/domain_nfcorpus_test/d6e2ea633d25526573c386535b04ef9f8e8837d9dd2c810b77ca9cddf8f187bc`, and the sealed bundle hash is `1da81daccde694089f458cd971721274d776a1ad443e0ab1b6a39f1de676126b`.

NFCorpus was intentionally published with all 12,334 test qrel pairs. The audit reports 194 queries with more than 10 positive documents, 13 with more than 200, and a maximum of 475; no qrels were clipped. The release stays `review_required` because 112 equivalent-unlabelled-document groups are recorded. Earlier `expansion-v1` (120 queries/502 pairs) and `expansion-v1-fullqrels` (211 queries/3,277 pairs) are preserved diagnostics: the former used the default positive limit and review exclusion, while the latter removed the positive limit but still excluded review-marked queries. They were not overwritten or deleted.

The ArguAna archive is 3,773,617 bytes, MD5 `8ad3e3c2a5867cdced806d6503f29b99`, SHA-256 `cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b`. Its raw snapshot is `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/domain_arguana_test/48d1f09de2dfddd49b5368432babaa7e160936329db60b02960106c66f3c9c8e`, and the full-qrels bundle hash is `fe72ebfafe7175341f82afbc59b52bdd4870eb25ca3ee46c3716d36235857764`. The raw test qrels contain 1,406 rows, but five rows refer to corpus IDs absent from the official corpus. The release keeps the 1,401 valid qrels and records the five unresolved rows in the raw snapshot/audit. The other 218 equivalent-unlabelled cases remain review-marked rather than silently changing labels. ArguAna is argument retrieval from online debate text and only legal-adjacent; it does not fill the legal corpus gap.

The earlier ArguAna `expansion-v1` release (1,183 queries/1,183 pairs) is also preserved as a diagnostic from the default equivalent-document policy. The `expansion-v1-fullqrels-noequiv` release is the one to integrate for the untrimmed valid qrels view.

For QASPER, the existing raw snapshot `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/qasper_dev/2afb05053bce7c7aa61add963277d9222c5507806da98a8bdd275efaf6c889d4` contains disjoint train (888) and dev (281) paper IDs. The archive SHA-256 is `a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a` and the archive size is 10,835,856 bytes. The domain-scoped derived input is `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/domain_qasper_dev_fullcorpus/input/qasper-full-v0.3.json` with SHA-256 `d30e470275b7a2929eb74c90d962394f5a60dbd1040009a3733e533bfc13ea3e`; its derivation manifest is next to it. The preparation copies all paper text/sections, removes only the train `qas` fields, and keeps dev annotations unchanged. Thus train documents expand the corpus without inventing train labels, while the original dev query IDs, qrels, and evidence policy remain intact. The QASPER raw source and the derived input are both retained.

Two additional official BEIR candidates were researched but not released in this run. FiQA is the finance candidate: [FiQA home](https://sites.google.com/view/fiqa/home), archive size 17,948,027 bytes, MD5 `17918ed23cd04fb15047f73e6c3bd9d9`, expected 57,638 documents and 648 test queries. The direct official stream stopped at 4,194,304 bytes; that partial file is preserved at `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/domain_fiqa_test/c18bf489f10cdc39674ee9c39a94c004d8f2bd730cd743d7e66d110556e44176.partial/beir/archive.zip.partial`, with no release. SciDocs is the scientific citation candidate: [AllenAI repository](https://github.com/allenai/scidocs), CC-BY-4.0, archive size 142,471,588 bytes and expected 25,657 documents/1,000 test queries. Its official archive was registered but deferred because the same slow stream would exceed the bounded acquisition window. No third-party mirror was substituted. The software domain and a proper legal corpus remain open matrix gaps.

The catalog is [configs/data_pipeline/domains_20261023.json](../configs/data_pipeline/domains_20261023.json). The acquisition wrapper is [scripts/acquire_domains_20261023.py](../scripts/acquire_domains_20261023.py); it uses the existing pipeline downloader and records commands, hashes, raw manifests, successful release verification, and preserved failure traces under `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023/domains/`. The consolidated record is `research.json` in that directory. The wrapper derives the QASPER full-corpus input only when the domain source is selected, and refuses to overwrite a mismatched existing derived file.

Reproduction/verification uses the configured environment and does not require models, GPUs, or package installation:

```bash
PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
$PY scripts/acquire_domains_20261023.py --source domain_qasper_dev_fullcorpus --release-name expansion-v1
$PY scripts/data_pipeline.py verify /mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/domain_nfcorpus_test/expansion-v1-fullqrels-noequiv
$PY scripts/data_pipeline.py verify /mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/domain_arguana_test/expansion-v1-fullqrels-noequiv
$PY scripts/data_pipeline.py verify /mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/domain_qasper_dev_fullcorpus/expansion-v1
```

No files under `dci_bench/data/pipeline/connectors.py` or `dci_bench/data/pipeline/core.py` were changed for this expansion.
