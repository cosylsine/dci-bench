# PrivacyQA native English legal source

`gapfill_privacyqa_legal_en` is the native English PrivacyQA retrieval source
used for the legal gap fill.  It uses the frozen upstream commit
`5178ff3eaf61cbdaea03d85ff23b676b3fea451f` from
[`AbhilashaRavichander/PrivacyQA_EMNLP`](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP).
The test CSV contains the original crowdsourced questions and expert
`Any_Relevant`/`Ann1`--`Ann6` sentence annotations.  The train CSV contributes
policy sentences to the corpus only; train questions are never published.

The upstream files have already been downloaded for this revision. For a new,
empty raw directory, fetch them from these pinned raw URLs:

```bash
set -e
REV=5178ff3eaf61cbdaea03d85ff23b676b3fea451f
RAW=/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_legal_pos10_20261116_v1/raw/privacyqa
mkdir -p "$RAW"
test ! -e "$RAW/policy_train_data.csv"
test ! -e "$RAW/policy_test_data.csv"
curl --fail --location --output "$RAW/policy_train_data.csv" \
  "https://raw.githubusercontent.com/AbhilashaRavichander/PrivacyQA_EMNLP/$REV/data/policy_train_data.csv"
curl --fail --location --output "$RAW/policy_test_data.csv" \
  "https://raw.githubusercontent.com/AbhilashaRavichander/PrivacyQA_EMNLP/$REV/data/policy_test_data.csv"
sha256sum "$RAW/policy_train_data.csv" "$RAW/policy_test_data.csv"
```

The expected raw sizes and digests are:

| file | bytes | SHA256 |
| --- | ---: | --- |
| `policy_train_data.csv` | 62,602,240 | `81aa674452fc822a49d7d96aead91fc85199ec99b6a56c7a862bb9deb02f004a` |
| `policy_test_data.csv` | 22,409,457 | `3822409246bfc733cdb17bb7bea85a92d906d76dae48763350c3306ee24a924a` |

The repository identifies its code license as MIT.  The paper and citation
record are [PrivacyQA: A Question Answering Dataset for Privacy Policies](https://aclanthology.org/D19-1500/),
and the pinned repository license is
[LICENSE](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP/blob/5178ff3eaf61cbdaea03d85ff23b676b3fea451f/LICENSE).

## Conversion contract

Run the adapter from the repository checkout:

```bash
python scripts/prepare_privacyqa_pos10.py \
  --data-root /mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_legal_pos10_20261116_v1
```

The adapter reads `raw/privacyqa`, writes the immutable release to
`releases/gapfill_privacyqa_legal_en`, and writes the DCI document task to
`run/source`.  It refuses to overwrite either an existing release or an
existing task.  Re-running after a complete conversion only verifies and
reuses those artifacts.  To verify an existing release and export without
writing, use:

```bash
python scripts/prepare_privacyqa_pos10.py \
  --data-root /mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_legal_pos10_20261116_v1 \
  --check-only
```

The source conversion is deliberately mechanical.  Every unique exact
`(DocID, Segment)` pair across both CSVs becomes one document with a stable
`privacyqa-doc-<sha256-prefix>` ID.  The body is:

```text
Privacy policy: <product title>

<exact original Segment>
```

The product title is the `DocID` prefix before its trailing ` _N` policy
instance.  The exact `DocID`, original sentence, all source `SentID` values,
split membership, and source hashes remain in document metadata.  The title
header distinguishes identical sentence text belonging to different policy
products while keeping the evidence sentence unchanged.

Each published query is an original test question with the deterministic
prefix `Regarding the <product title> privacy policy: `; the untouched raw
question is retained in query metadata.  Its `group_id` is
`privacyqa-query:<raw QueryID>`, so grouping follows the source query identity
and does not merge an entire policy.  Positive qrels are the union of all
`Any_Relevant=Relevant` rows after sentence deduplication.  A complete query
is retained only when it has 1--10 unique positive canonical documents; no
positive row is truncated.  Every retained source annotation row is preserved
in `judgments`, and one exact quote/offset evidence row is emitted per
query/document positive pair.

The verified source inventory is 185,200 train rows, 62,150 test rows, 4,657
canonical documents, 400 test query groups, and 167 retained queries with 990
positive pairs.  The adapter keeps the full 4,657-document corpus and uses
the 167 retained test questions as the query set.  The built-in acceptance
check also runs the benchmark's policy-prefixed query-text eligibility and
deduplication check; the expected result is at least 60 eligible queries.

The release provenance records the pinned raw URLs, raw SHA256 values, MIT
license, paper URL, human annotation basis, native English status, all 400
query filter decisions, and the canonical `(DocID, Segment)` mapping.  A
`labels_ready` release here means source human labels passed the data quality
contract; it does not claim model retrieval or benchmark completion.
