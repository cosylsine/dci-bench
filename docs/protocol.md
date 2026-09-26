# DCI-Bench MVP Protocol

The canonical contract is `dci_bench/protocol/contract.json` at version
`dci-mvp-v2`. Python and TypeScript load that same file; prompts, tools and
budgets must not be duplicated in either runtime.

The base system prompt, sandbox rules, evidence requirement, budgets, and final
JSON schema are task-neutral. Each `TaskSpec` may add one trusted
`task_instruction` containing retrieval strategy only. At runtime that block is
inserted into the query prompt, recorded in the resolved benchmark contract and
Inspect/Pi artifacts, and therefore participates in the immutable run
fingerprint. An empty instruction preserves generic behavior for new tasks.
Task instructions must not contain relevance labels, document ids, qrels, or
exceptions to corpus inspection and output validation.

The task registry is authoritative. New metadata manifests include an audit
copy of the instruction. Pre-v2 manifests without the field remain compatible,
but a manifest that contains a different or non-string instruction is rejected.
Unknown task names still fail closed; only a registered task with an empty
instruction uses the generic prompt. Any instruction change produces a new run
fingerprint and cannot be resumed into an older incomplete run.

## Dataset To Workspace

- Source data is the official `mteb/llm-eval-*` held-out retrieval data at the
  pinned revisions in `dci_bench.data.registry`.
- The agent-visible workspace contains only:
  - `README.md`
  - `corpus/<url-percent-encoded-doc-id>.md`
- Each document markdown file contains the optional title and original text.
- Query records, qrels, sample metadata, and manifests are host-side metadata.
- The builder does not create embeddings, BM25 indexes, summaries, keywords, or
  generated document metadata.

## Pi To Ranked JSON

- Final answer schema: `{"ranked_doc_ids": ["..."]}`.
- `ranked_doc_ids` is a non-empty, duplicate-free string array of at most 10 ids.
- The id is the original corpus `_id`; if a filename is percent-encoded, remove
  `.md` and percent-decode it.
- Tool allowlist: `read`, `bash`.
- Fixed limits:
  - `top_k`: 10
  - `max_agent_steps`: 1200
  - `sample_timeout_seconds`: 600
  - `tool_result_max_chars`: 20000
  - `max_total_model_tokens`: 128000
  - `max_final_output_tokens`: 1024
- Context compaction is disabled for the MVP.
- MiniCPM5-2B/SGLang and Qwen3.8-27B/vLLM launchers both default to a
  65536-token context window per request (input plus output), separate from
  the 128000-token cumulative budget per query. The backend manifest records
  the configured window and passes it to the agent's context preflight.
- A sample succeeds on the first assistant message that parses as the final
  answer schema after the agent has inspected corpus document content. Filename
  listings alone are not evidence. Timeout, max steps, max tokens, unrecovered
  tool failures, or a model stop without valid JSON are failures.
- Every model turn must return usage. Pi accumulates input plus output tokens
  across normal and repair turns. A total at or above 128000 is `token_limit`,
  including when the crossing turn contains otherwise valid final JSON.
- Disabled capabilities: web, retriever, MCP, skills, sub-agents, memory,
  interactive clarification, and provider-specific agent enhancements.
- Inspect `sandbox="local"` is transport plumbing for the model bridge, not the
  security boundary. Every agent `bash` process runs in a Bubblewrap namespace
  with no network and no host data mounts beyond the current read-only corpus;
  `read` performs an independent canonical-path corpus check. Missing or broken
  Bubblewrap is a hard failure and never falls back to host execution.
- Unknown document ids are schema-valid but receive zero relevance. Duplicate
  ids, empty arrays, more than 10 ids, non-string ids, and extra JSON keys are
  invalid outputs.

## Inspect To Metrics And Trace

Phase 4 records use these versioned JSON Schemas:

- `dci-sample-result-v1`
- `dci-task-result-v1`
- `dci-run-manifest-v1`
- `dci-trajectory-v1`
- `dci-task-index-v1`

Each sample result records:

- task, query id, ranked ids, output validity, failure reason
- Recall/F1/nDCG at every configured cutoff
- agent steps, tool/model calls, repairs, routed-model token usage, wall and
  working time
- relative final/trace/Inspect artifact paths and SHA-256 hashes
- explicit provenance and `local`/`bubblewrap` security attestations

Inspect `EvalSample.model_usage`, `total_time`, and `working_time` are the
normative usage/timing values. Pi's per-turn trace remains the cross-audit
source. Missing successful-sample usage is `usage_unavailable`, not zero.

Task-level macro metrics include every planned query, with failed or invalid
samples contributing zero. `valid_only` is an additional view. Population
standard deviation and R-7 p50/p95 are used. Sample wall-time sums and task
elapsed time are distinct, especially under concurrency.

For a valid answer, the evaluated window at cutoff `k` is the returned prefix
of length `min(k, len(ranked_doc_ids))`. Precision divides relevant hits by
this actual window length; recall divides those hits by the total number of
relevant documents. F1 is the harmonic mean of these two values. An empty
window scores zero. nDCG retains its existing cutoff-based definition.
Completed run artifacts are immutable and keep their original metric values;
comparisons under a revised scoring rule must be recomputed from their saved
rankings and the host-side qrels.

Runs are stored under `results/<model_key>/<task>/runs/<run_id>`. Sample files
are first completed in `.staging`, then atomically promoted. `_COMPLETE` is
created only after schema and artifact hash verification; a completed run is
immutable. Resume requires a matching configuration fingerprint and skips only
terminal samples with intact `result.json`, `final.json`, and `trace.json`.

Scoring loads host-side qrels only after generation. Agent-visible workspaces
must not contain the qrels parquet or generated qrels JSON, and qrels must not
be placed in Inspect `Sample.target` or model-visible task state.
