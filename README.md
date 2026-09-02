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

Phase 3 uses Inspect with `sandbox="local"` and the `openai-api/vllm/...`
Chat Completions provider against a vLLM OpenAI-compatible server for
`/mnt/afs/share/Qwen3-4B`:

```bash
conda activate dci-vllm
MODEL_PATH=/mnt/afs/share/Qwen3-4B SERVED_MODEL_NAME=Qwen3-4B scripts/serve_model.sh

conda activate dci-bench
python scripts/phase3_single_sample.py --prepare --task LLMPublicHealthQA --query-id Q25
```
