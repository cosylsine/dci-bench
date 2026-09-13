"""Registry for the six pinned MTEB(LLM) retrieval tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(
    "/mnt/afs2/202608/embedding_models/dci-bench/mteb_llm_retrieval"
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_dataset: str
    revision: str
    local_dir_name: str


TASKS: dict[str, TaskSpec] = {
    "LLMAILAStatutes": TaskSpec(
        name="LLMAILAStatutes",
        hf_dataset="mteb/llm-eval-aila-statutes",
        revision="a2acf12d293e",
        local_dir_name="llm-eval-aila-statutes",
    ),
    "LLMFQuADRetrieval": TaskSpec(
        name="LLMFQuADRetrieval",
        hf_dataset="mteb/llm-eval-fquad",
        revision="dc5443dbfad5",
        local_dir_name="llm-eval-fquad",
    ),
    "LLMHC3FinanceRetrieval": TaskSpec(
        name="LLMHC3FinanceRetrieval",
        hf_dataset="mteb/llm-eval-hc3-finance",
        revision="8733760a6f3e",
        local_dir_name="llm-eval-hc3-finance",
    ),
    "LLMLegalBenchConsumerContractsQA": TaskSpec(
        name="LLMLegalBenchConsumerContractsQA",
        hf_dataset="mteb/llm-eval-legalbench-consumer-contracts",
        revision="642870c78f65",
        local_dir_name="llm-eval-legalbench-consumer-contracts",
    ),
    "LLMPublicHealthQA": TaskSpec(
        name="LLMPublicHealthQA",
        hf_dataset="mteb/llm-eval-public-health-qa",
        revision="b05938525381",
        local_dir_name="llm-eval-public-health-qa",
    ),
    "LLMTwitterHjerneRetrieval": TaskSpec(
        name="LLMTwitterHjerneRetrieval",
        hf_dataset="mteb/llm-eval-twitter-hjerne",
        revision="31f9b918c30e",
        local_dir_name="llm-eval-twitter-hjerne",
    ),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task {name!r}. Available tasks: {choices}") from exc
