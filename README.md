# DCI-Bench

Benchmark for evaluating LLM Direct Corpus Interaction (DCI) capabilities.

## MVP Status

Phase 0 through Phase 5 are implemented. Phase 5 adds strict pinned-revision
validation, real workspaces for all six registry tasks, and a sequential
all-task runner on top of the Phase 4 immutable result lifecycle:

- frozen MVP protocol constants: `dci_bench.protocol.contracts`
- pinned MTEB(LLM) task registry: `dci_bench.data.registry`
- local parquet loader and workspace builder: `dci_bench.data.workspace_builder`
- offline retrieval scorer: `dci_bench.scoring.retrieval`
- canonical `dci-mvp-v2` prompt/tool/budget contract with task-aware guidance:
  `dci_bench/protocol/contract.json`
- batch task runner: `scripts/run_task.py`
- sequential six-task runner: `scripts/run_all.py`
- result schemas and integrity lifecycle: `schemas/results/` and `dci_bench.results`

MiniCPM5-2B + SGLang real acceptance covers a two-query serial run, an isolated
two-query concurrent run, and a five-query run whose over-budget samples are
retained as audited `token_limit` failures. The 100-query full task is not run
automatically.

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

## Qwen3.8-27B: four 5090 nodes, six tasks, vLLM

Configure the SenseCore job for **4 nodes with 4 RTX 5090 GPUs per node** and
the shared `/mnt/afs/250010100/zxp/dci-bench` and
`/mnt/afs/share/Qwen3.8-27B` mounts. Paste this command into the platform's
Bash startup-command field; the platform runs it once on each node:

```bash
cd /mnt/afs/250010100/zxp/dci-bench && \
WORLD_SIZE="$SENSECORE_PYTORCH_NNODES" \
NODE_RANK="$SENSECORE_PYTORCH_NODE_RANK" \
MAX_CONCURRENCY=1 \
VLLM_MAX_NUM_SEQS=4 \
VLLM_GPU_MEMORY_UTILIZATION=0.80 \
bash scripts/run_phase5_qwen38_vllm_node.sh
```

Do not wrap this command in `torchrun`: each node starts its own local vLLM
replica with tensor parallel size 4, then evaluates its assigned tasks against
`127.0.0.1:17892/v1`. No cross-node GPU collective or model-service network
connection is needed. The script checks that the platform reports four nodes
and four visible RTX 5090s on each node, that PyTorch was built with CUDA 12.8+
and `sm_120`, and that vLLM supports the checkpoint's `qwen3_5` architecture.
It rejects an occupied service port, checks `/v1/models`, then requires a real
OpenAI tool call and token usage before starting DCI queries. The service is
stopped when that node finishes or receives a termination signal.

`/v1/models` confirms the HTTP server is ready; the first chat completion also
exercises inference and may take longer while the 5090 workers finish compiling
their first generation path. The tool-call check allows 600 seconds by default
(`TOOL_SMOKE_TIMEOUT_SECONDS`), while the `/v1/models` probe keeps its 30-second
request limit. The node log records the chat request's elapsed time and prints
the end of the vLLM service log if the request fails. A timeout at this stage
means inference was not verified; inspect the service log and the saved request
under `results/job-artifacts/` before changing model or GPU settings.

The task assignment is fixed and disjoint:

| Node rank | DCI tasks |
| --- | --- |
| 0 | LLMAILAStatutes, LLMFQuADRetrieval |
| 1 | LLMHC3FinanceRetrieval, LLMTwitterHjerneRetrieval |
| 2 | LLMLegalBenchConsumerContractsQA |
| 3 | LLMPublicHealthQA |

Each node validates all six prepared workspaces before loading the model.
Prepare the data once before submitting the four-node job; the launcher does
not rebuild shared data concurrently. The evaluator uses the `dci-bench`
environment and writes immutable runs under
`results/Qwen3.8-27B/<task>/runs/`. Launcher logs and endpoint diagnostics go
to `results/job-logs/` and `results/job-artifacts/`; the vLLM backend manifest
goes to `results/backend-manifests/`. These paths must be writable and shared
across the nodes. The image must provide `/usr/bin/bwrap` and `/usr/bin/rg`;
the launcher installs them with `apt-get` only if it is running as root.

`MAX_CONCURRENCY` is the number of simultaneous queries per node, separate from
the four GPUs used by tensor parallelism. Start at 1. vLLM's independent
`VLLM_MAX_NUM_SEQS` defaults to 4, and `VLLM_GPU_MEMORY_UTILIZATION` defaults
to 0.80. These limits avoid the 5090 startup OOM caused by vLLM's default
256-request sampler warmup; `VLLM_MAX_NUM_SEQS` must be at least
`MAX_CONCURRENCY`. Both values are recorded in the backend manifest. The
per-request model limit defaults to `MAX_MODEL_LEN=65536`; change it only after checking 5090
memory use and context failures. `DRY_RUN=1` checks rank-to-task selection
without starting the model. A completed task with failed samples leaves an
audited result and a `runner_exit=1` line in the node log while the job exits
successfully by default; use `FAIL_JOB_ON_SAMPLE_FAILURES=1` for strict job
status. Infrastructure and integrity errors always exit nonzero.

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
