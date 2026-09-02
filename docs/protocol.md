# DCI-Bench MVP Protocol

This document freezes Phase 0. Runtime code should import
`dci_bench.protocol.contracts` so Pi, Inspect, and scoring use the same contract.

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
- A sample succeeds on the first assistant message that parses as the final
  answer schema after the agent has inspected corpus document content. Filename
  listings alone are not evidence. Timeout, max steps, max tokens, unrecovered
  tool failures, or a model stop without valid JSON are failures.
- Disabled capabilities: web, retriever, MCP, skills, sub-agents, memory,
  interactive clarification, and provider-specific agent enhancements.
- Unknown document ids are schema-valid but receive zero relevance. Duplicate
  ids, empty arrays, more than 10 ids, non-string ids, and extra JSON keys are
  invalid outputs.

## Inspect To Metrics And Trace

Each sample result must record at least:

- task, query id, ranked ids, output validity, failure reason
- nDCG@10 and Recall@10
- agent steps, tool calls, model input/output tokens, wall time
- trace path

Scoring uses host-side qrels only. Agent-visible workspaces must not contain the
qrels parquet or generated qrels JSON.
