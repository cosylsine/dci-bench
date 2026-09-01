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
```

This writes agent-visible corpus files to `data/workspaces/LLMPublicHealthQA`
and host-only metadata, including qrels and manifest, to
`data/metadata/LLMPublicHealthQA`.

The default local data root is:

```text
/mnt/afs2/202608/embedding_models/dci-bench/mteb_llm_retrieval/mteb_llm_retrieval
```
