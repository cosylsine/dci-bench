# DCI-Bench

Benchmark for evaluating LLM Direct Corpus Interaction (DCI) capabilities.

## MVP Status

Phase 0 and Phase 1 are implemented for the first task data path:

- frozen MVP protocol constants: `dci_bench.protocol.contracts`
- pinned MTEB(LLM) task registry: `dci_bench.data.registry`
- local parquet loader and workspace builder: `dci_bench.data.workspace_builder`
- offline retrieval scorer: `dci_bench.scoring.retrieval`

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

The MiniCPM5 service and Phase 3 runner are intentionally separate. Start the
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

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 scripts/serve_sglang_minicpm5.sh
```

Single-host tensor parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 scripts/serve_sglang_minicpm5.sh
```

After `/v1/models` is ready, run the two Phase 3 samples from another terminal:

```bash
scripts/run_phase3_with_sglang.sh
```

`127.0.0.1` is correct only when service and evaluation share a node. For
separate allocations, start the service with `SGLANG_HOST=0.0.0.0` and pass
`OPENAI_BASE_URL=http://<service-hostname>:30000/v1` to the evaluator (with
the cluster firewall allowing that port).

Its defaults are `LLMPublicHealthQA`, `QUERY_IDS="Q71 Q72"`, and
`http://127.0.0.1:30000/v1`. Canonical query results are written below
`results/phase3_sglang_minicpm5_2b/LLMPublicHealthQA/Q*/`, while timestamped
endpoint checks, Inspect logs, and summaries are kept below the task's `_runs/`
directory. It validates the SGLang MiniCPM5 tool-call path before evaluating
either query, keeps both sample artifacts even if one fails, and returns nonzero
when either final DCI output is invalid. Set `OUTPUT_DIR=/path/to/model-results`
to choose an explicit model result root.
