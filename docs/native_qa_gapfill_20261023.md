# Native QA gapfill (2026-10-23)

This release adds three monolingual encyclopedia QA-derived retrieval views:
Russian SberQuAD, ThaiQA, and Turkish TQuAD. They are separate source
families and are not combined into one multilingual task.

The conversion keeps every source `train` and `dev` context. Identical
Unicode context text is merged only when it is exact, and every original
source location is retained in `metadata.source_aliases`. The official `dev`
questions are the only emitted retrieval queries. Training questions are
counted in `conversion_manifest.json` but are never used as held-out queries.
Each qrel is the source QA question-to-context association with score `1.0`.
Answer spans are optional evidence; malformed offsets remain in
`span_audit.json` and do not remove their query or qrel.

The source contexts and questions remain in their native language. No
translation, synthetic question, synthetic positive, or cross-language corpus
padding is performed. ThaiQA's literal `<doc ...>...</doc>` context is kept
byte-for-byte and is treated as one source QA context retrieval paragraph; it
is not stripped or resegmented.

## Sources and evidence

| language | source | pinned revision / bytes | source evidence | license recorded |
| --- | --- | --- | --- | --- |
| `ru` | SberQuADClean | `sber_squad_clean-v1.1.tar.gz`, 22,766,184 bytes, SHA-256 `5f5ba3e1a9344e72a60b701f8c30b52536df860810654a899dcf441714d0d40a` | [archive](https://files.deeppavlov.ai/datasets/sber_squad_clean-v1.1.tar.gz), [DeepPavlov config](https://github.com/deeppavlov/DeepPavlov/blob/5f9fbed0c7191466bc7621e604b810f66f254c03/deeppavlov/configs/squad/squad_ru_bert.json), [SberQuAD paper](https://arxiv.org/abs/1912.09723) | unknown; the accessed archive/config does not state a dataset license |
| `th` | ThaiQA | `PyThaiNLP/thaiqa_squad` commit `61a4c4f36babcad721bb6bf9561fff4416a775ae`, `data.zip` 10,003,354 bytes, SHA-256 `6aa58c8a9b52862de166618b9b682d7094f56a1587f9681f3dd617bfd78f95c8` | [pinned repository](https://github.com/PyThaiNLP/thaiqa_squad/tree/61a4c4f36babcad721bb6bf9561fff4416a775ae), [dataset README](https://raw.githubusercontent.com/PyThaiNLP/thaiqa_squad/61a4c4f36babcad721bb6bf9561fff4416a775ae/README.md), [NECTEC source](https://aiforthai.in.th/corpus.php) | CC-BY-NC-SA-3.0 |
| `tr` | TQuAD | `TQuad/turkish-nlp-qa-dataset` commit `3bece60f62569182476fe00683793745ba92c32e`; train 3,078,146 bytes, dev 458,494 bytes | [pinned repository](https://github.com/TQuad/turkish-nlp-qa-dataset/tree/3bece60f62569182476fe00683793745ba92c32e), [README](https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/3bece60f62569182476fe00683793745ba92c32e/README.md) | unknown; no explicit data license was found in the pinned repository |

The ThaiQA README attributes native Thai contexts to Wikipedia and question / answer annotations to NECTEC. The TQuAD README describes Turkish
questions and contexts for the Teknofest 2018 Turkish and Islamic science
history task. The SberQuAD archive is distributed by the DeepPavlov SberQuAD
configuration; its dataset license remains explicitly unknown here.

## Counts observed before release

| language | source train+dev contexts | deduplicated documents | official dev queries | positive pairs | valid optional spans |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ru` | 50,364 | 9,078 | 5,036 | 5,036 | 5,036 |
| `th` | 4,074 QA rows / source contexts | 2,326 | 74 | 74 | 74 |
| `tr` | 2,507 | 2,182 | 892 | 892 | 892 |

The exact release counts, including deduplication and audit records, are
authoritative in each raw `conversion_manifest.json` and release `manifest.json`;
no source rows are dropped from the source audit even when exact contexts merge.
TQuAD's dev file repeats each answer annotation three times for these queries;
the 1,784 duplicate span records remain in `span_audit.json`, while the sealed
evidence file keeps one exact row per query.

## Artifacts

Raw source inputs and conversion audits:

* `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/gapfill_nativeqa_ru/`
* `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/gapfill_nativeqa_th/`
* `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/gapfill_nativeqa_tr/`

Clean immutable releases:

* `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_nativeqa_ru/v1/`
* `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_nativeqa_th/v1/`
* `/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_nativeqa_tr/v1/`

Research evidence and the complete source matrix are in
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round3/native_qa/research.json`.
The small `progress.json` is written first so a parent run can inspect
candidate progress while acquisition continues.

## Reproduction and validation

Run the bounded acquisition from the repository root:

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/acquire_native_qa_gapfill_20261023.py \
  --languages ru th tr
```

Then verify each sealed release:

```bash
for lang in ru th tr; do
  /mnt/afs/250010100/miniconda/envs/dci-bench/bin/python scripts/data_pipeline.py verify \
    "/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_nativeqa_${lang}/v1"
done
```

The acquisition performs source byte/hash checks, raw-to-canonical conversion,
query ID/text and qrel source-context mapping checks, exact duplicate alias
retention, optional span audits, and `verify_release`. It does not install
dependencies, run GPU inference, run cascade retrieval, or claim benchmark
metrics.

## Blocked or deferred candidates

* FQuAD (`fr`): the two official legacy storage URLs returned HTTP 403 during
  a bounded probe. It is recorded as blocked; no unverified mirror or partial
  source is used.
* PIAF (`fr`): official project and paper references were found, but no stable,
  size-bounded official train+dev source file was identified in this pass.

See `research.json` for the URLs, status, license notes, and reasons in
machine-readable form.
