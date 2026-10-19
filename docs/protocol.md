# DCI-Bench MVP Protocol

This document describes the **document track** and its existing online runtime.
The independent **code evaluation track** has its own output schema and metrics
in [`code_evaluation.md`](code_evaluation.md) and `dci_bench/protocol/code.py`.
Its online entry is `scripts/run_code_task.py`, with runtime settings in
`dci_bench/protocol/code_runtime.py`. Both tracks score on the host after each
Pi solver finishes. Code's file/span/abstention rules do not alter the document
contract below or the existing document launchers.

The canonical contract is `dci_bench/protocol/contract.json` at version
`dci-mvp-v3.3`. Python and TypeScript load that same file; prompts, tools and
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

## Experimental budget-aware profiles

`DCI_PROTOCOL_PROFILE` is selected before importing the Python task stack and
is inherited by Pi. The default `v3` is the unbudgeted control arm. All arms use
the current exact corpus-ID validation and read metadata. The
opt-in profiles in `dci_bench/protocol/pilot_profiles.json` use contract version
`dci-mvp-v3.3-pilot`: `v3-temp1` changes only temperature to 1; `budgeted` adds
budget control at temperature 0; `budgeted-temp1` combines both changes.
The separately opt-in `budgeted-calibrated` profile inherits the `budgeted`
prompts and gates, overriding only the policy version and reserve-forecast
parameters. The original `budgeted` resolved contract hash is preserved.
Each profile must run in a fresh process with a matching generation manifest.

Budget control is task-neutral and never reads qrels. After three consecutive
identical tool calls it switches to a read-only phase if no body has been read,
or switches to finalization if body evidence exists. Six search turns without a
body read also force the read-only phase, including when commands keep changing.
Only `read` is advertised and executable during this phase, with a named read
tool choice. At most two responses may obtain nonempty body evidence; empty,
failed or ignored reads end as `tool_failure`. A successful required read resumes
search when the token reserve permits; the trace records
`resume_search`. This restores both tools and resets the repeated-call counter.
A read receipt proves inspection, not relevance or sufficient retrieval coverage.
Low remaining tokens still cause immediate finalization after the required read.
Policy `budgeted-finalize-v5`
reserves at least 32000 tokens. Before allowing search to continue, it accounts
for the next search/read request, final submission and one repair, including one
bounded next tool result replayed in both later inputs. This applies even when
body evidence already exists: another search can otherwise consume the repair
reserve before finalization begins.
The estimate uses the most recent observed input/output usage plus the UTF-8
byte count of subsequent tool text, with output and safety margins per request.
Wall time is only the existing 600-second hard timeout. It never triggers an
early read-only phase or final submission; a slow first request does not change
the search policy. Token limits and repeated-call controls drive finalization.
Late responses remain timeout failures. Finalization disables tools for
submission and repairs, and uses a JSON-only repair prompt referring to evidence
already inspected. A backend returning tool calls cannot execute those calls.

All control prompts, parameters and generation changes participate in the run
fingerprint. Pi final and trace files record `protocol_profile`, the task-neutral
`base_contract_sha256`, and control events; the full run fingerprint additionally
includes task instruction, data, model and execution settings. Parent and Pi
reject mismatched base hashes. Final and repair requests share the original
128000-token limit and are rejected before dispatch when the conservative
estimate cannot fit. Actual returned usage remains the acceptance authority;
budget estimates do not relax the read gate, JSON validation or hard limits.
The forecast assumes one bounded next tool result; it is not a guarantee for
multiple returned tool results or repeated invalid finals. Updated prompts and
policies produce new contract fingerprints; prior results remain immutable.

Policy `budgeted-finalize-v6-calibrated` estimates subsequent tool text at two
UTF-8 bytes per token and uses a 1024-token safety margin per reserved request.
With `N = last_input + last_output + ceil(trailing_tool_bytes / 2)`, its reserve
is `max(32000, 3*N + 29216)`. It still pays for one search/read request, a
20,000-byte next tool result replayed in both later requests, final submission,
and one repair. The v5 baseline uses one byte per token and 4096 safety, yielding
`max(32000, 3*N + 58432)` with its own byte-based `N`.
The new parameters covered all 805 observed tool-to-next-request transitions
in the 60-query `20261013T083704Z-1` pilot. This is calibration-set coverage,
not a guarantee for future or longer trajectories. The pre-dispatch guard
continues to use one byte per token and 4096 safety; per-request context checks
and the actual cumulative usage limit are unchanged. Control events include
the forecast components for audit. No wall-time reserve is introduced.

The earlier calibration comparison is frozen in `scripts/harness_budget_round2_plan.json`:
first a fresh paired 6x10 diagnostic run, then a separate paired 6x10 holdout
run after the stability review. Both compare `budgeted` to
`budgeted-calibrated`, use the same warmed endpoint, and alternate profile
order across tasks. The holdout excludes diagnostic IDs and takes the first
ten per task by SHA256 of `dci-budget-round2-v1:<task>:<query_id>`, using query
metadata only. Its IDs and metadata hashes are frozen in
`scripts/harness_pilot_holdout_queries.json`; missing or changed freeze inputs
must fail before evaluation. It was excluded from this calibration, but may
have appeared in earlier full-benchmark evaluations. Report these two suites
separately and retain all-sample, valid-only, paired quality and cost metrics.

The next opt-in comparison uses `dci-mvp-v3.4-pilot` profiles and is specified in
`scripts/harness_selection_round3_plan.json`. `budgeted-stable` retains the v5
reserve and all existing gates. Its `stable-rg-v1` search policy uses trusted
`/usr/bin/rg` with a read-only `RIPGREP_CONFIG_PATH` containing `--sort=path`.
Files are ordered before `head` or the tool byte cap, while line order inside a
file and stdin pipelines are preserved. The configuration text and its model
notice participate in the contract hash; the file is checked against that text
and included in the source archive. CLI flags (`--sort=none`, `--no-config`) can
override this default. Other bash commands retain their original semantics;
this does not promise deterministic arbitrary shell programs or model outputs.
Sorting makes ripgrep traversal single-threaded, so report its latency cost.

`budgeted-selective` inherits that exact search environment and v5 budget.
It changes only the final-submission and final-repair prompts, with policy
version `budgeted-finalize-v7-selective`: review relevance to the query and task,
remove candidates supported only by keywords or broad topic overlap, and retain
multiple relevant documents when justified. There is no new model request,
host-side relevance filter, qrels access, schema relaxation, or requirement to
read every returned document. Natural finals are still accepted normally; the
new final prompt applies only when the controller requests finalization, and
the new repair prompt applies during tools-disabled repairs. Analyze natural
finals separately when measuring this treatment's coverage.

Run both profiles on the existing diagnostic 6x10 suite first (120 evaluations),
alternating their order by task. Both previously evaluated suites are now
development evidence, so the old `holdout` name is historical and cannot support
a fresh independent validation claim for this candidate. Keep the previous
outputs intact; use a new paired run instead of comparing only against saved
scores. The immediate goal is fewer extra documents without losing relevant
ones, at the conservative budget. Larger independent validation and protocol
freezing remain separate decisions. Old profiles and their contract hashes are
unchanged, and the launcher now preflights every explicitly selected profile
before CUDA/model startup.

The pilot shell launcher performs synthetic serving warmup after readiness and
before starting any profile. Two rounds cover automatic tools, named `read`,
and tools-disabled final responses at the configured sample concurrency. Only
synthetic prompts are sent; no task queries, corpus files or qrels are read, and
no returned tool call is executed. A failed HTTP, usage or response-shape check
stops the run before evaluation. Named `read` calls accept `finish_reason=stop`
or `tool_calls`, covering vLLM's named-tool response format; automatic calls
require `tool_calls`. Both paths still require the expected tool and valid
arguments. Truncated responses (`length`) are rejected.
The separate `warmup.json` backend artifact records request counts, durations
and usage. Failure details include the round, request shape/index, HTTP status,
actual finish reason, tool names and up to 16 KiB of response body. Cluster logs
also include these details with a 2 KiB response preview; truncation is explicit.
Request headers and API keys are not recorded. Aggregate counts and usage in a
failed warmup cover successful batches only; the failing response is retained
in `error.details`. Startup warmup has a configurable
600-second total deadline and per-request timeout by default; each request is
also bounded by the remaining total deadline. Its
tokens and time are excluded from all query budgets and retrieval metrics.
Warmup reduces first-request cold-start effects but does not establish latency
or retrieval quality for real long-context requests.

## Dataset To Workspace

- Source data is the official `mteb/llm-eval-*` held-out retrieval data at the
  pinned revisions in `dci_bench.data.registry`.
- The generated workspace contains only:
  - `README.md`
  - `corpus/<url-percent-encoded-doc-id>.md`
- Each document markdown file contains the optional title and original text.
- Runtime tools expose only `corpus/`; the generated README is not mounted.
- Query records, qrels, sample metadata, and manifests are host-side metadata.
- The builder does not create embeddings, BM25 indexes, summaries, keywords, or
  generated document metadata.

## Pi To Ranked JSON

- Final answer schema: `{"ranked_doc_ids": ["..."]}`.
- `ranked_doc_ids` is a non-empty, duplicate-free string array of at most 10 ids.
- The id is the exact original corpus `_id`. To recover it from a filename,
  the agent removes the final filesystem `.md` suffix and percent-decodes once.
  The final-answer parser never strips or decodes an already recovered ID.
  Thus original IDs such as `report.md`, `100%`, and `a%2Fb` remain intact.
- Each file `read` starts with a JSON `doc_id` metadata line derived from the
  trusted corpus filename, which the agent can copy directly. Every submitted ID
  must exactly match an original corpus ID. Unknown IDs are rejected for repair;
  feedback includes exact filename-to-ID mappings when available, never qrels.
  No ID is silently normalized or replaced by the scorer.
- Tool allowlist: `read`, `bash`.
- `read.path` accepts an exact literal corpus filename such as `report.md.md`,
  optionally prefixed with `./`, as well as `corpus/<filename>` and
  `/workspace/corpus/<filename>`. Bare `.md` filenames resolve only inside the
  corpus; other relative paths resolve from `/workspace`. A shell `cd` never
  changes read resolution. Filename percent escapes are not decoded for path
  lookup. Canonical containment and rejection of symlinks and hard-linked files
  apply to every accepted spelling. Missing/invalid paths return actionable
  feedback without treating a failed read as evidence.
- Fixed limits:
  - `top_k`: 10
  - `max_agent_steps`: 1200
  - `sample_timeout_seconds`: 600
  - `tool_result_max_bytes`: 20000 (UTF-8, including status/truncation markers)
  - `max_total_model_tokens`: 128000
  - `max_final_output_tokens`: 1024
  - `max_model_output_tokens`: 1024 (every model turn, including tools and repairs)
- Canonical generation parameters are `temperature=0`, `seed=0`,
  `max_tokens=1024`, and `enable_thinking=false`. Inspect owns their execution;
  the bridge does not forward client generation defaults. Backend manifests
  and final artifacts record the resolved settings; traces identify the
  contract version. Backend-facing
  request tests verify the parameters rather than only checking manifest text.
- Context compaction is disabled for the MVP.
- MiniCPM5-2B/SGLang and Qwen3.8-27B/vLLM launchers both default to a
  65536-token context window per request (input plus output), separate from
  the 128000-token cumulative budget per query. The backend manifest records
  the configured window and passes it to the agent's context preflight.
- A sample succeeds on the first assistant message that parses as the final
  answer schema after at least one **nonempty `read`** of corpus text. Bash
  remains a general search tool but never satisfies the evidence gate. Each
  qualifying read records its corpus path, original `doc_id`, Unicode code point offset, actual
  returned code points/bytes, and SHA-256 of the visible source text. Directory
  listings, whitespace-only reads, beyond-EOF reads and truncation markers are
  not evidence. The metadata prefix counts toward the output byte cap but is
  excluded from body evidence, receipt hashes and source offsets. This minimum gate does not claim that every ranked document was
  read or that its relevance has been proven. Timeout, max steps, max tokens, unrecovered
  tool failures, or a model stop without valid JSON are failures.
- Every model turn must return usage. Pi accumulates input plus output tokens
  across normal and repair turns. A total at or above 128000 is `token_limit`,
  including when the crossing turn contains otherwise valid final JSON.
  This is a post-turn acceptance limit: the crossing request may consume more
  than the remaining budget, but no subsequent model request is allowed.
- `read.offset` and `read.limit` count Unicode code points, not lines, bytes or
  UTF-16 code units. Both tools enforce the same UTF-8 byte limit and preserve
  complete code points. Bash combines stdout/stderr; output overflow terminates
  its process group and returns a bounded prefix plus termination status.
- The 600-second Inspect sample wall deadline includes bridge startup/waiting,
  Pi startup, model/backend waiting and tools. Pi receives only the time remaining
  after the bridge is ready, and an absolute deadline established before Node
  starts; startup never grants a fresh 600 seconds. Cancellation aborts the HTTP
  request and tools. The host gives the process group five seconds to publish
  diagnostics before SIGKILL; answers arriving during this grace are failures.
- The versioned adapter policy accepts the single-key `top_documents` alias,
  strips supported leading thinking blocks only when direct JSON parsing fails,
  and recognizes text-encoded read/bash calls. Literal tags inside JSON strings
  remain unchanged.
  These deterministic adapters add no model calls; up to three explicit repair
  prompts share the same step, token and wall budgets. Native truncated tool
  calls retain `finish_reason=length` and are not executed as complete calls.
- Disabled capabilities: web, retriever, MCP, skills, sub-agents, memory,
  interactive clarification, and provider-specific agent enhancements.
- Inspect `sandbox="local"` is transport plumbing for the model bridge, not the
  security boundary. Every agent `bash` process runs in a Bubblewrap namespace
  with no network and no host data mounts beyond the current read-only corpus;
  `read` performs an independent canonical-path corpus check. Missing or broken
  Bubblewrap is a hard failure and never falls back to host execution.
- Bash has no interactive standard input (`/dev/null`). Pathless `rg` therefore
  searches the current directory; pipes created inside a shell command still
  carry their normal input. Harness source hashes distinguish this correction
  from historical executions that left an unwritten input pipe open.
- Unknown document IDs cannot be accepted even when JSON is schema-valid.
  Exhausting repairs leaves an invalid output with the unknown-ID reason.
  Duplicate IDs, empty arrays, more than 10 IDs, non-string IDs, and extra JSON
  keys are also invalid outputs.

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
Inspect's terminal `sample.limit` takes precedence over a secondary Pi error:
time/working limits are `timeout`, including when cancellation caused `fetch
failed` or a final answer arrived too late. Timed-out samples cannot score as
successful submissions.

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
Pi atomically snapshots the partial trajectory throughout execution, including
completed messages and tool start/end events. Timeout handling preserves that
snapshot rather than replacing it with an empty trajectory.

Execution fingerprints include sample concurrency and runtime/dependency
identity as well as code, data, prompts, model and backend configuration.
Changing concurrency changes the conditions of a wall-time-limited experiment.
Historical v1/v2 runs remain immutable; v3 is a new protocol and cannot resume
into those runs. Protocol changes do not require regenerating unchanged corpus
files or qrels.

Scoring loads host-side qrels only after generation. Agent-visible workspaces
must not contain the qrels parquet or generated qrels JSON, and qrels must not
be placed in Inspect `Sample.target` or model-visible task state.
