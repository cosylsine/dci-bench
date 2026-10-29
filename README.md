# DCI-Bench

Benchmark for evaluating LLM Direct Corpus Interaction (DCI) capabilities.

Current progress (2026-10-28): [interactive data and evaluation overview](docs/project-progress-20261028/index.html),
[harness workflow](docs/project-progress-20261028/workflow.svg), and
[short collaborator update](docs/project-progress-20261028/progress.md).
The document track has 53 sources / 159 candidate pools and 159 successful real
Qwen3.8-27B smoke executions; this diagnostic run is not a full benchmark.

## Evaluation tracks

DCI-Bench separates **document** and **code** evaluation. The existing six-task
Inspect/Pi workflow and document scoring stay unchanged; its explicit API is
`dci_bench.scoring.document` (the original `scoring.retrieval` API remains valid).
`dci_bench.scoring.code` adds independent repository-file metrics,
line coverage, function/functional-evidence recall, explicit no-gold abstention,
and ARB budgeted context yield. Fine-grained metrics require host annotations;
retrieval coverage does not imply test execution coverage.

Run the included CPU-only synthetic example:

```bash
python -m dci_bench.scoring --track code \
  --labels examples/code_retrieval/labels.json \
  --predictions examples/code_retrieval/predictions.json
```

The code track also runs online through `scripts/run_code_task.py`: Pi retrieves
repository context, then Inspect scores each sample on the host. It accepts a
completed ARB export or custom query/label manifests and uses source-line reads
and an independent code output contract. Existing document launchers retain their
current behavior. Run the CPU mock through the online path:

```bash
python scripts/run_code_task.py \
  --queries examples/code_retrieval/online_queries.json \
  --labels examples/code_retrieval/labels.json --query-ids demo-trace \
  --mock-script examples/code_retrieval/mock_trace.json
```

Use `--preflight-only` for data/sandbox checks without requests. See
[the separate contracts, metrics, and input formats](docs/code_evaluation.md).

## Prepared evaluation data

New document preparations use task-local short IDs (`d000001`) with reversible,
host-only source-ID and qrels mappings. The new per-query candidate-pool workflow
uses small/medium/large capacities of 200/500/1000 and provisional cumulative
budgets of 96k/128k/192k tokens. See [document pools](docs/document_pools.md) for
score-ordered ID lists, shared corpus storage, temporary query workspaces,
positive completion, audit boundaries, and offline preparation.
The current 53-source pool catalog and KaLM retrieval audit are complete, and the
paired 159-execution smoke passed. Full candidate-pool evaluation remains pending;
historical frozen data and the code track retain their own contracts.

The [2026-10-15 protocol data index](docs/protocol_data_2026-10-15.md) covers
13 document tasks (2,399 queries, including WorkSurface) and 287 ARB code
queries. It also provides directly loadable queries for the 45-query code
subset whose positive file views are untruncated. The subset overlaps the full
code set. Original corpora, query IDs, labels, and frozen releases are preserved.

`scripts/prepare_protocol_data.py` builds or verifies the host-only index and
records commands for the existing document/code runners. Corpus files remain
in their frozen source packages; the index is not a standalone corpus copy.

## MVP Status

Phase 0 through Phase 5 are implemented. Phase 5 adds strict pinned-revision
validation, real workspaces for all six registry tasks, and a sequential
all-task runner on top of the Phase 4 immutable result lifecycle:

- frozen MVP protocol constants: `dci_bench.protocol.contracts`
- pinned MTEB(LLM) task registry: `dci_bench.data.registry`
- local parquet loader and workspace builder: `dci_bench.data.workspace_builder`
- document retrieval scorer: `dci_bench.scoring.document` (legacy `scoring.retrieval`)
- canonical `dci-mvp-v3.3` prompt/tool/budget contract with task-aware guidance:
  `dci_bench/protocol/contract.json`
- batch task runner: `scripts/run_task.py`
- sequential six-task runner: `scripts/run_all.py`
- result schemas and integrity lifecycle: `schemas/results/` and `dci_bench.results`

Earlier-protocol MiniCPM5-2B + SGLang acceptance covers a two-query serial run, an isolated
two-query concurrent run, and a five-query run whose over-budget samples are
retained as audited `token_limit` failures. The 100-query full task is not run
automatically. Protocol v3.3 uses CPU/mock and sandbox regression checks;
full Qwen3.8-27B GPU acceptance must be run separately on the cluster.

Prepare all six tasks without overwriting a complete matching workspace:

```bash
scripts/prepare_data.sh
make test
make phase2-toy
```

Pass `--overwrite` explicitly to the builder only when a task workspace must be
rebuilt. Agent-visible corpus files are written under `data/workspaces/<task>`;
queries, qrels, and the pinned-revision manifest remain host-only under
`data/metadata/<task>`.

The default local data root is:

```text
/mnt/afs2/202608/embedding_models/dci-bench/eval_data/mteb_llm_retrieval
```

Phase 3 uses Inspect with `sandbox="local"` only for the local model bridge.
Agent `bash` calls run inside a fail-closed Bubblewrap sandbox that exposes only
the current read-only corpus, a per-sample scratch directory, and read-only
system runtime files. Agent `read` calls are independently restricted to the
current corpus. Qrels stay host-side and are not stored in `Sample.target`.

Each registry entry may define a trusted `task_instruction`. The generic DCI
security, corpus-inspection, tool, and final-JSON rules remain shared; only a
short retrieval-strategy block varies by task. The resolved instruction is
recorded in the run contract/fingerprint and Inspect/Pi artifacts. New tasks
may leave the field empty to retain the generic prompt. The registry is the
runtime source of truth; newly built metadata copies the instruction for audit.
Legacy metadata without that copy remains valid, while a present but stale copy
is rejected. Changing an instruction changes the run fingerprint, so an old
incomplete run cannot be resumed under different prompting.

The container loses apt-installed packages after a restart, so install
Bubblewrap before each new container lifetime:

```bash
apt install -y bubblewrap
```

Then run the `openai-api/vllm/...` Chat Completions provider against a vLLM
OpenAI-compatible server for `/mnt/afs/share/Qwen3-4B`:

```bash
conda activate dci-vllm
MODEL_PATH=/mnt/afs/share/Qwen3-4B SERVED_MODEL_NAME=Qwen3-4B scripts/serve_vllm_qwen38.sh

conda activate dci-bench
python scripts/phase3_single_sample.py --prepare --task LLMPublicHealthQA --query-id Q25
```

Retrieval scoring reports recall, F1, and nDCG at `1, 3, 5, 10, 20` by
default. Override the cutoffs for a run with, for example,
`--metric-ks 1 5 10`.
At cutoff `k`, precision uses the number of documents actually returned in
the first `k` positions, `min(k, returned_count)`, rather than treating missing
positions as retrieved documents. Recall uses the same prefix and divides by
the number of relevant documents; F1 is their harmonic mean. Historical run
artifacts retain the scoring rule used when they were created. New run
manifests record `dci-retrieval-scoring-v2` in their benchmark contract.

For a one-terminal cluster job that starts vLLM and then runs Phase 3:

```bash
scripts/run_phase3_with_vllm.sh
```

Defaults: `/mnt/afs/share/Qwen3.8-27B`, tensor parallel size 4, and
`QUERY_IDS="Q71 Q72"`. Override with environment variables when needed.

## MiniCPM5-2B with SGLang

The MiniCPM5 service and Phase 4 runner are intentionally separate. Start the
service in one terminal and leave it running; the evaluation script only
connects to the existing OpenAI-compatible endpoint.

SGLang compiles MiniCPM5's fused RoPE kernel at runtime, which needs a host
C++ compiler supporting C++20 and `<concepts>` (GCC 10–13 for the supplied
CUDA 12.6 toolkit). The launcher checks this with `nvcc` before loading weights
and exports it through `CXX`; the local SGLang source forwards that compiler to
`nvcc` with `-ccbin`. If the node's default compiler is too old, either load a
newer compiler module and set `SGLANG_JIT_CXX=/path/to/g++`, or install one
into `dci-sglang-minicpm5`:

```bash
conda activate dci-sglang-minicpm5
conda install -y -c conda-forge "gxx_linux-64=12"
```

Single GPU (use an explicit manifest path so the evaluator can consume the
exact service metadata):

```bash
CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 \
BACKEND_MANIFEST_PATH=results/backend-manifests/minicpm5-sglang.json \
scripts/serve_sglang_minicpm5.sh
```

Single-host tensor parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 scripts/serve_sglang_minicpm5.sh
```

After `/v1/models` is ready, run Q71/Q72 in one Inspect evaluation from another
terminal. The wrapper performs workspace, endpoint, usage and non-greedy tool
parser checks once, then invokes the batch runner once:

```bash
BACKEND_MANIFEST=results/backend-manifests/minicpm5-sglang.json \
QUERY_IDS="Q71 Q72" MAX_CONCURRENCY=1 \
scripts/run_phase4_with_sglang.sh
```

For an explicit two-query parallel acceptance run, set the concurrency instead
of looping over query IDs in the shell:

```bash
BACKEND_MANIFEST=results/backend-manifests/minicpm5-sglang.json \
QUERY_IDS="Q71 Q72" MAX_CONCURRENCY=2 BRIDGE_PORT=13131 \
scripts/run_phase4_with_sglang.sh
```

This still launches one Inspect `eval()`. The two active samples lease separate
bridge ports (`13131` and `13132`) and publish separate `Q71/` and `Q72/`
directories.

`127.0.0.1` is correct only when service and evaluation share a node. For
separate allocations, start the service with `SGLANG_HOST=0.0.0.0` and pass
`OPENAI_BASE_URL=http://<service-hostname>:30000/v1` to the evaluator (with
the cluster firewall allowing that port).

The backend-neutral entry point can also be called directly:

```bash
python scripts/run_task.py \
  --model-key MiniCPM5-2B \
  --task LLMPublicHealthQA \
  --query-ids Q71 Q72 \
  --backend-manifest results/backend-manifests/minicpm5-sglang.json \
  --max-concurrency 2 \
  --bridge-port 13131
```

Use `--all-queries` instead of `--query-ids ...` for full-task selection; the
two forms are mutually exclusive. Explicit IDs reject duplicates, unknown IDs,
and IDs without qrels before model calls. `--resume-run RUN_ID` only resumes an
incomplete run whose fingerprint still matches, and never retries an already
terminal sample in place.

After all six workspaces pass preflight, run every registry task sequentially:

```bash
python scripts/run_all.py \
  --model-key MiniCPM5-2B \
  --backend-manifest results/backend-manifests/minicpm5-sglang.json \
  --max-concurrency 1
```

The orchestrator validates all six manifests before the first model request.
It continues after a completed task containing failed samples (overall exit 1),
but stops on a configuration or integrity failure (exit 2). The all-task command
has unit and construction coverage; a costly real six-task GPU evaluation has
not been run automatically.

## One-job Phase 5 SGLang run

`scripts/run_phase5_with_sglang.sh` is the foreground entry point intended for
a cluster allocation. It prepares and validates all six workspaces, checks the
Pi/Bubblewrap runtime, starts SGLang in an isolated process group, waits until
the expected model appears at `/v1/models`, verifies a real tool call, runs all
six tasks, and always stops the service it started.

For one node with four visible GPUs:

```bash
cd /mnt/afs/250010100/zxp/dci-bench
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONTEXT_LENGTH=65536 \
MAX_CONCURRENCY=4 \
BRIDGE_PORT=18891 \
scripts/run_phase5_with_sglang.sh
```

`TP_SIZE` controls SGLang tensor parallelism. `MAX_CONCURRENCY` controls the
number of simultaneous query samples sharing that backend; it is not a GPU
count. Start with 1, then increase through 2/4/8 while watching valid-output
rate, P95 latency and KV-cache pressure; long-context prefill and local sandbox
startup can otherwise hide the backend throughput gain. Bridge startup is
serialized to avoid Inspect 0.3.260's shared tool-injection race, while model
requests still run concurrently. Each task receives a disjoint bridge port
range, and every task subprocess is isolated so leaked bridge children can be
cleaned up without killing unrelated listeners.

`CONTEXT_LENGTH` is the per-request SGLang limit for prompt plus generated
tokens; its default is 65536, matching Qwen3.8-27B/vLLM's `MAX_MODEL_LEN`.
Both models use the separate 128000-token benchmark budget, cumulative across
every model turn in one sample. Raising the cumulative budget
does not make an oversized single request fit, and raising the context window
increases the worst-case KV-cache and prefill cost.

The SGLang environment also checks that PyTorch sees at least `TP_SIZE` CUDA
devices before loading the model. The script defaults to a loopback
endpoint on port `17891` because service and evaluator run inside the same
allocation. This Phase 5-specific default avoids SGLang's commonly used
`30000` and the usual Linux ephemeral port range. Phase 5 bridge ranges start
at `18891` by default instead of Inspect's common `13131`.
`WAIT_ATTEMPTS=360 WAIT_SECONDS=5` gives the service up to 30 minutes to become
ready. Jobs sharing a host must use distinct `SGLANG_PORT` and `BRIDGE_PORT`
base values; an occupied SGLang port fails closed before model launch. Set
`DRY_RUN=1` to execute local preflights and render the SGLang command without
starting the service or model evaluation.

The task runner still records exit code `1` when a completed task contains
invalid or failed samples. The cluster launcher preserves that status in its
summary and logs, but defaults to a successful job exit after all six immutable
runs are complete so a scheduler does not misclassify produced evaluation
results as a crashed job. Set `FAIL_JOB_ON_SAMPLE_FAILURES=1` to restore strict
nonzero job exit behavior. Configuration, preflight, and integrity failures
remain nonzero in either mode.

## Qwen3.8-27B: three nodes, twelve RTX 5090 GPUs

Request **3 nodes with 4 RTX 5090 GPUs per node**, with this checkout, prepared
data, model weights and the result directory mounted on every node. Configure
the scheduler to run **one Bash process per node**. Use the same fresh run ID
on all three nodes:

```bash
cd /mnt/afs/250010100/zxp/dci-bench
WORLD_SIZE=3 DCI_RUN_ID=qwen38-dci-v3-3n-01 \
  bash scripts/run_phase5_qwen38_vllm_node.sh
```

The script reads `SENSECORE_PYTORCH_NODE_RANK` or `SLURM_PROCID`. If the platform
does not supply a rank, add `NODE_RANK=0`, `1`, or `2` to the command on the
corresponding node. For Slurm, use `srun --nodes=3 --ntasks=3 --ntasks-per-node=1`
inside an allocation containing four GPUs per node to execute that same command.
Do not use `torchrun`: each node starts an independent TP=4 vLLM replica.

| Node rank | Tasks, run sequentially |
| --- | --- |
| 0 | LLMAILAStatutes, LLMFQuADRetrieval |
| 1 | LLMHC3FinanceRetrieval, LLMTwitterHjerneRetrieval |
| 2 | LLMLegalBenchConsumerContractsQA, LLMPublicHealthQA |

Defaults are `VLLM_ENV=dci-vllm-qwen38`, `BENCH_ENV=dci-bench`,
`MODEL_PATH=/mnt/afs/share/Qwen3.8-27B`, `MAX_CONCURRENCY=1`,
`MAX_MODEL_LEN=65536`, `VLLM_MAX_NUM_SEQS=4`, and
`VLLM_GPU_MEMORY_UTILIZATION=0.80`. Override `CONDA_SH`, the environment names,
`MODEL_PATH`, `WORKSPACE_ROOT`, `METADATA_ROOT`, and `RESULTS_ROOT` if the other
cluster mounts them elsewhere. Keep these settings identical across nodes.
`CONDA_ROOT` defaults to `/mnt/afs/250010100/miniconda`; `BENCH_PYTHON` defaults
to its `envs/$BENCH_ENV/bin/python`. Override it explicitly for a custom prefix.
Concurrency counts active queries per node, independently of the four GPUs.

The launcher validates all six existing workspaces without rebuilding them.
It checks CUDA 12.8+, `sm_120`, four RTX 5090 devices, an actual CUDA kernel on
each device, and vLLM support for the checkpoint's `qwen3_5` architecture.
It then verifies `/v1/models`, a real tool call and returned token usage before
evaluating the assigned full tasks. `/usr/bin/bwrap` and `/usr/bin/rg` are
required; the launcher can install them when the node runs as root. Missing
Conda environments or model weights fail preflight rather than triggering
large automatic downloads or installations.

The first generation can compile 5090 kernels; the tool smoke allows
`TOOL_SMOKE_TIMEOUT_SECONDS=600`. The service readiness loop allows 30 minutes
by default. Model manifests record published LFS SHA-256 values and check the
recorded file sizes when available; weights without LFS digests are hashed
locally, adding sequential disk I/O before the benchmark. Serving runtime
and hardware diagnostics are saved with the run. The benchmark's model requests
use the v3 canonical generation settings, separate from the tool-parser probe.

Each node stops its own service after its tasks finish, then waits for the
other nodes' terminal states before exiting. Shared terminal-state coordination
propagates infrastructure failures and rejects reuse of an old run group.
The final barrier allows 48 hours (`BARRIER_TIMEOUT_SECONDS=172800`) and
detects a missing heartbeat after 30 minutes, so a fast node does not end the
allocation while another node is still evaluating its tasks.
Choose a new `DCI_RUN_ID` for every submission. Do not edit the shared checkout
or replace dependencies while a job is running.

Results remain immutable under `results/Qwen3.8-27B/<task>/runs/` (or the
configured results root). Logs and diagnostics are separate. A completed run
may include failed samples: `runner_exit=1` is retained in diagnostics, while
the launcher defaults to job success if all assigned runs completed. Set
`FAIL_JOB_ON_SAMPLE_FAILURES=1` for strict nonzero job status. Infrastructure
and integrity errors always fail the job.

CPU-only assignment preview, with no service or evaluation:

```bash
WORLD_SIZE=3 NODE_RANK=0 DRY_RUN=1 bash scripts/run_phase5_qwen38_vllm_node.sh
```

## Harness protocol v3.3

`dci-mvp-v3.3` requires a nonempty `read` receipt, exposes the exact original
`doc_id` in read results, and rejects unknown submitted IDs for repair. It also
classifies Inspect time limits ahead of secondary Pi errors and shares the
remaining sample deadline with Pi. Read accepts literal corpus filenames as well
as corpus paths, while retaining containment and link checks. Generation settings, byte caps, partial
trajectories and execution fingerprints remain explicit. Historical results
are retained under their original protocol; use a fresh run for v3.3.
See [the protocol](docs/protocol.md) for exact
budgets, adapter behavior and timeout scope.

Local validation uses CPU/mock endpoints and Bubblewrap, not model inference:

```bash
conda activate dci-bench
python -m unittest discover -s tests -v
cd pi-dci
npm run check
npm run test:dci
```

The three-node full-model acceptance must run on the allocated cluster; local
checks do not establish GPU memory fit, distributed scheduling or final scores.

## Small harness regression pilot

The opt-in `dci-mvp-v3.3-pilot` profiles compare unbudgeted `v3`, temperature-only
`v3-temp1`, budget-aware `budgeted`, and `budgeted-temp1`. The default protocol
remains the unbudgeted `v3` arm, with the shared v3.3 ID and path checks. Budget-aware
profiles reserve tokens for submission, restrict repeated/no-read
searches to at most two read-only responses, and disable tools during finalization.
After a successful required read, search resumes if the remaining tokens permit.
Time remains a hard timeout only and never triggers early read/finalization.
They retain the nonempty read requirement, JSON schema, 65536-token request
context, 128000-token cumulative budget and 600-second hard deadline.

After the server is ready, the launcher runs two rounds of synthetic requests
at the configured concurrency, covering automatic tool calls, a named `read`
call and a final response with tools disabled. These requests do not use task
queries or corpus data and do not execute tools. `warmup.json` records their
latency, token usage and success separately from evaluated samples. Warmup
failure stops the launcher before evaluation. `WARMUP_DEADLINE_SECONDS=600`
and `WARMUP_HTTP_TIMEOUT_SECONDS=600` bound this startup check; they do not
change the query's token budget or trigger early finalization.

Test the repaired budget-aware profile on ten fixed queries from each of the six
tasks at temperature 0, on one host with four H100 or RTX 5090 GPUs (60 samples):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAX_CONCURRENCY=4 \
  bash scripts/run_harness_pilot.sh --profiles budgeted
```

Use `--profiles v3 budgeted` for a paired 120-sample run. The unbudgeted arm
intentionally omits loop/finalization control; it is not the repaired budgeted
profile. Both arms use the new ID contract and fresh output directories.

For the next relevance-selection comparison, use the two opt-in v3.4 profiles:

- `budgeted-stable`: the original conservative v5 budget and gates, with default
  ripgrep file ordering fixed to `--sort=path` before truncation.
- `budgeted-selective`: the same search environment and budget, changing only
  the tools-disabled final and repair prompts to review relevance and exclude
  weakly related candidates while retaining multiple relevant documents.

Run the existing diagnostic ten queries per task, paired across all six tasks
(120 evaluations; full corpus for each task):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAX_CONCURRENCY=4 \
  bash scripts/run_harness_pilot.sh \
  --profiles budgeted-stable budgeted-selective \
  --query-suite diagnostic --profile-order alternating
```

This uses Qwen3.8-27B BF16 by default, context 65536, cumulative budget 128000,
temperature 0 and the existing 600-second hard deadline. It starts a fresh
output directory and preflights both profiles before model startup. No extra
final review request is added. Explicit ripgrep flags may override sorting;
arbitrary bash output and model trajectories are not guaranteed deterministic.
See [`scripts/harness_selection_round3_plan.json`](scripts/harness_selection_round3_plan.json)
for exact hashes, metrics and decision criteria. The previous diagnostic and
holdout results informed this candidate; neither is a fresh validation set now.

For the next independent query validation, keep these two profiles frozen and
use ten new query IDs per task, excluding both diagnostic and round-2 holdout:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAX_CONCURRENCY=4 \
  bash scripts/run_harness_pilot.sh \
  --profiles budgeted-stable budgeted-selective \
  --query-suite validation --profile-order alternating
```

[`scripts/harness_pilot_validation_queries.json`](scripts/harness_pilot_validation_queries.json)
freezes the 60 IDs by SHA256 with seed `dci-selection-validation-v1`, using
query metadata only, without qrels or score filtering. Historical full
evaluations may have included these IDs. Missing or changed query metadata,
exclusion files, freeze, or profile contracts stop before model startup. Each
run archives `validation-freeze.json` and uses a fresh result directory. Report
first-final success and repair counts alongside the existing quality, validity,
cost and paired comparison metrics; retain the original round-3 decision gates.

For the earlier token-calibration comparison, use the original `budgeted` arm and
the new opt-in `budgeted-calibrated` arm. The candidate estimates tool growth
at 2 UTF-8 bytes per token with 1024 safety tokens per reserved request; the
32000-token reserve floor and all hard limits stay unchanged. Its parameters
covered all 805 observed transitions in the latest 60-query calibration run;
new trajectories still require model evaluation. The complete frozen design,
metrics and decision criteria are in
[`scripts/harness_budget_round2_plan.json`](scripts/harness_budget_round2_plan.json).

Stage 1 reruns both arms on the existing six-task diagnostic suite (120 sample
evaluations):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MAX_CONCURRENCY=4 \
  bash scripts/run_harness_pilot.sh \
  --profiles budgeted budgeted-calibrated \
  --query-suite diagnostic --profile-order alternating
```

After both arms produce 60/60 valid outputs and pass integrity/pairing checks,
run stage 2 with the same command and `--query-suite holdout` (another 120
evaluations). That suite freezes ten other queries per task by seeded SHA256,
without score or qrels filtering, and checks the source query hashes before
running. It excludes this round's calibration queries; historical full runs may
have used these queries. Keep the candidate frozen between stages and report
the suites separately. `alternating` runs both arms together for each task,
reversing their order on odd-numbered tasks so each arm runs first three times.
The historical four-profile default stays unchanged; calibration is opt-in.

The locally available `Qwen3.8-27B-NVFP4` checkpoint also quantizes its output
head to FP8. The installed vLLM 0.17.0 rejects its quantization configuration;
0.19.1 accepts the configuration but needs output-head support. Any isolated
serving compatibility patch must be retained and recorded with the backend
manifest. Do not silently replace the shared environment or model configuration.
The local H100 MIG pilot used a private vLLM 0.19.1 overlay and the Triton GDN
prefill backend; its NVFP4 layers ran with Marlin weight-only kernels. This
quantized pilot establishes local feasibility, not parity with the BF16 model
or native NVFP4 kernels on RTX 5090.

The default query selection is ten fixed queries from each of the six tasks.
Both profiles use the same IDs in the same order. For a smaller smoke run,
add `--tasks LLMFQuADRetrieval --query-limit 2` to the command above.
Omit `--profiles` to include both temperature-1 diagnostic arms (240 sample
evaluations with the default query selection).
Set `MAX_CONCURRENCY=4` (or add `--max-concurrency 4`) to evaluate up to four
queries at once within each task. Tasks and profiles still run sequentially.
The default is 1; `VLLM_MAX_NUM_SEQS` follows the requested concurrency unless
explicitly set to a larger value. The launcher records both settings and reserves
separate bridge port ranges for every task/profile. Use the same concurrency for
both profiles. `--query-limit 1` still runs only one query per task, so it cannot
exercise sample concurrency. Increased concurrency needs a new GPU acceptance
run; the 600-second per-sample deadline includes backend waiting.
`MODEL_PATH`, `SERVED_MODEL_NAME`, `SERVING_PYTHON`, `BENCH_PYTHON`, `VLLM_PORT`
and `PILOT_ROOT` can be overridden. `DRY_RUN=1` prints settings without starting
inference. Before loading the model, the launcher verifies source archiving and
file hashing. It also attempts Git revision/patch capture for both checkouts;
Git metadata is optional, and any failure is recorded in `git-audit.json` with a
warning while evaluation continues. Archive/hash failures still stop the run.
Audit commands specify each
checkout's Git directory and work tree explicitly, supporting shared-mount
ownership without changing global configuration. Git errors include the checkout,
command and stderr.
This CPU-only check can also be run with
`python scripts/run_harness_pilot.py --preflight-only`.
The shell launcher also checks `/usr/bin/bwrap` and `/usr/bin/rg` before model
startup. Missing packages are installed with `apt-get` when running as root;
otherwise it stops with an environment error. It then validates the six prepared
workspaces and runs Pi's actual sandbox preflight. Installation or sandbox
failures stop before CUDA probing or model loading. The Python
`--preflight-only` command above checks source archiving only.
The launcher applies the CUDA library exports, probes a CUDA kernel, starts one
vLLM replica, and stops that service on exit. A model's quantization must be
supported by the serving environment.

For an existing endpoint, use `scripts/run_harness_pilot.py --backend-manifest
<manifest.json>`. Each profile gets a new manifest and subprocess. Results,
full model-request logs, source snapshots, patches and `summary.json` are saved
under a fresh `results/harness-pilots/` directory; existing runs are never reused.
The fixed IDs in `scripts/harness_pilot_queries.json` had valid outputs in v2
and failed in v3; valid outputs need not retrieve relevant documents.
Each task retains its original two pilot IDs first; the eight additional IDs
were selected from the remaining eligible cases by ascending SHA256 of the
UTF-8 string `task:query_id`, without filtering on retrieval scores.
Their scores diagnose regressions and are not an unbiased
estimate of the six-task benchmark. Validate any selected candidate on fresh
queries and the original BF16 checkpoint before adopting it as the default.

## Phase 4 result protocol

Every invocation creates a new immutable run:

```text
results/<model_key>/<task>/
├── index.json
└── runs/<run_id>/
    ├── run-manifest.json
    ├── task-result.json
    ├── Qxx/{result.json,final.json,trace.json}
    ├── artifacts/{inspect,backend}/...
    └── _COMPLETE
```

`result.json` uses Inspect sample usage/timing as the normative token/time
source and records Pi trace usage for cross-audit. Successful samples require
usage, timing and the `local`/`bubblewrap` attestations; unavailable values on a
failed sample remain `null`, never fake zero. Task metrics use all planned
queries as the macro denominator with failed/invalid samples scored as zero,
and separately report `valid_only`. `_COMPLETE` is written only after schema
and SHA-256 verification. Exit codes are 0 for all-success, 1 for a completed
run containing sample failures, and 2 for configuration/preflight/integrity
failure.

All ordinary `.json` audit files use two-space indentation, recursively sorted
keys, and one trailing newline. Canonical compact JSON is used only internally
for deterministic fingerprints and content hashes. JSONL inputs remain one
record per line.

## Data acquisition and construction pipeline

`scripts/data_pipeline.py` supports existing retrieval labels, evidence QA, and
raw corpora. Source adapters, immutable raw/cleaned snapshots, quality reports,
local query generation and adjudication, independent candidate mining, and
200/500/1000-document pools are documented in [Data pipeline](docs/data_pipeline.md).
The source catalog is [sources.json](configs/data_pipeline/sources.json); local
model roles and deployment settings are in [models.json](configs/data_pipeline/models.json).
Use `catalog` and `doctor` for read-only inspection. Acquisition requires explicit
source IDs; inference is a separate bounded stage.

## 代码、数据与结果目录

- `scripts/`、`dci_bench/`、`pi-dci/`：维护中的启动脚本与实现。159 次 pool smoke 的入口为 `bash scripts/launch_pool_smoke.sh`。
- `releases/`：冻结的可执行代码、依赖快照及发布配置；目录说明见 [releases/README.md](releases/README.md)。
- AFS2 的 `raw_data/`、`eval_data/`：数据和数据发布凭证，不放启动脚本或运行时代码。
- `results/`：评测输出、日志、诊断报告与来源证据，不作为可执行代码的来源。
