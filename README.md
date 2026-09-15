# DCI-Bench

Benchmark for evaluating LLM Direct Corpus Interaction (DCI) capabilities.

## MVP Status

Phase 0 through Phase 4 are implemented for the first task data path. Phase 4
adds a backend-neutral batch entry point and immutable, versioned result records:

- frozen MVP protocol constants: `dci_bench.protocol.contracts`
- pinned MTEB(LLM) task registry: `dci_bench.data.registry`
- local parquet loader and workspace builder: `dci_bench.data.workspace_builder`
- offline retrieval scorer: `dci_bench.scoring.retrieval`
- canonical `dci-mvp-v1` prompt/tool/budget contract: `dci_bench/protocol/contract.json`
- batch task runner: `scripts/run_task.py`
- result schemas and integrity lifecycle: `schemas/results/` and `dci_bench.results`

MiniCPM5-2B + SGLang real acceptance covers a two-query serial run, an isolated
two-query concurrent run, and a five-query run whose over-budget samples are
retained as audited `token_limit` failures. The 100-query full task is not run
automatically.

Default Phase 1 task:

```bash
make prepare TASK=LLMPublicHealthQA
make test
make phase2-toy
```

This writes agent-visible corpus files to `data/workspaces/LLMPublicHealthQA`
and host-only metadata, including qrels and manifest, to
`data/metadata/LLMPublicHealthQA`.

The default local data root is:

```text
/mnt/afs2/202608/embedding_models/dci-bench/mteb_llm_retrieval/mteb_llm_retrieval
```

Phase 3 uses Inspect with `sandbox="local"` only for the local model bridge.
Agent `bash` calls run inside a fail-closed Bubblewrap sandbox that exposes only
the current read-only corpus, a per-sample scratch directory, and read-only
system runtime files. Agent `read` calls are independently restricted to the
current corpus. Qrels stay host-side and are not stored in `Sample.target`.

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
