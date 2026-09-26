"""Registry for the six pinned MTEB(LLM) retrieval tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = Path(
    "/mnt/afs2/202608/embedding_models/dci-bench/eval_data/mteb_llm_retrieval"
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_dataset: str
    revision: str
    local_dir_name: str
    task_instruction: str = ""


TASKS: dict[str, TaskSpec] = {
    "LLMAILAStatutes": TaskSpec(
        name="LLMAILAStatutes",
        hf_dataset="mteb/llm-eval-aila-statutes",
        revision="a2acf12d293e",
        local_dir_name="llm-eval-aila-statutes",
        task_instruction=(
            "The query is a long legal case narrative, while corpus documents are statutes or "
            "statutory provisions. First reduce the narrative to a short list of legal issues, "
            "offences, procedures, Acts, and section concepts. Search those statutory concepts "
            "and their legal synonyms; do not expect party names, dates, or the case narrative "
            "to occur verbatim. Once the strongest provisions have been read and verified, rank "
            "them and stop instead of continuing broad searches."
        ),
    ),
    "LLMFQuADRetrieval": TaskSpec(
        name="LLMFQuADRetrieval",
        hf_dataset="mteb/llm-eval-fquad",
        revision="dc5443dbfad5",
        local_dir_name="llm-eval-fquad",
        task_instruction=(
            "Queries are French questions and corpus documents are answer-bearing French "
            "passages. Search two to four distinctive entities, noun phrases, or short stems "
            "from the question, trying a small number of accent or inflection variants only when "
            "needed. Read the best passages and return as soon as one directly answers the "
            "question; avoid exhaustive searches after a verified answer is found."
        ),
    ),
    "LLMHC3FinanceRetrieval": TaskSpec(
        name="LLMHC3FinanceRetrieval",
        hf_dataset="mteb/llm-eval-hc3-finance",
        revision="8733760a6f3e",
        local_dir_name="llm-eval-hc3-finance",
        task_instruction=(
            "Queries are finance questions and corpus documents are candidate answers. Extract "
            "the distinctive financial topic, instruments, constraints, and user situation; "
            "search those terms plus only a few standard finance synonyms. Prefer a passage that "
            "directly addresses the question over a document that merely shares generic finance "
            "words, then stop after verifying the best candidate."
        ),
    ),
    "LLMLegalBenchConsumerContractsQA": TaskSpec(
        name="LLMLegalBenchConsumerContractsQA",
        hf_dataset="mteb/llm-eval-legalbench-consumer-contracts",
        revision="642870c78f65",
        local_dir_name="llm-eval-legalbench-consumer-contracts",
        task_instruction=(
            "Queries describe consumer legal issues, while corpus documents are contract "
            "clauses. Translate the issue into likely clause language such as forum selection, "
            "arbitration, termination, liability, payment, privacy, or governing law. Search the "
            "relevant clause terms and named service, read the matching clauses, and do not rely "
            "on the user's question appearing verbatim. Stop once the controlling clause is "
            "verified."
        ),
    ),
    "LLMPublicHealthQA": TaskSpec(
        name="LLMPublicHealthQA",
        hf_dataset="mteb/llm-eval-public-health-qa",
        revision="b05938525381",
        local_dir_name="llm-eval-public-health-qa",
        task_instruction=(
            "Queries are short public-health questions and corpus documents are concise answer "
            "passages. Search the disease, intervention, population, symptom, or exposure term "
            "together with the requested fact, using a small number of medical synonyms when "
            "necessary. Read the most specific passages and return once a passage directly "
            "answers the question; avoid repeated searches on generic words."
        ),
    ),
    "LLMTwitterHjerneRetrieval": TaskSpec(
        name="LLMTwitterHjerneRetrieval",
        hf_dataset="mteb/llm-eval-twitter-hjerne",
        revision="31f9b918c30e",
        local_dir_name="llm-eval-twitter-hjerne",
        task_instruction=(
            "Queries and documents are Danish social posts and may contain hashtags, compounds, "
            "inflections, spelling variants, or conversational replies. Remove the hashtag marker "
            "when searching, try a bounded set of Danish word stems or compound parts, and read "
            "the directly responsive posts. Several documents can be relevant, so return the "
            "responsive set in relevance order, but stop when additional search variants no "
            "longer add distinct candidates."
        ),
    ),
}

# Registry insertion order is the canonical Phase 5 execution order.  Keep it
# aligned with the task table in DCI-Bench_MVP_Implementation_Plan.md.
TASK_NAMES: tuple[str, ...] = tuple(TASKS)


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task {name!r}. Available tasks: {choices}") from exc


def validate_task_manifest(task: TaskSpec, manifest: Mapping[str, Any]) -> str:
    """Validate the dataset identity and pinned revision of generated metadata."""

    if manifest.get("task") != task.name:
        raise ValueError(f"data manifest task does not match registry task {task.name}")
    if manifest.get("dataset") != task.hf_dataset:
        raise ValueError(f"data manifest dataset does not match registry dataset {task.hf_dataset}")
    requested_revision = manifest.get("revision")
    if requested_revision != task.revision:
        raise ValueError(
            f"data manifest revision {requested_revision!r} does not match pinned revision "
            f"{task.revision!r}"
        )
    resolved_revision = manifest.get("resolved_revision")
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise ValueError("data manifest resolved_revision must be a non-empty string")
    if not resolved_revision.startswith(task.revision):
        raise ValueError(
            f"data manifest resolved_revision {resolved_revision!r} does not match pinned "
            f"revision prefix {task.revision!r}"
        )
    # Manifests produced before dci-mvp-v2 do not have this audit copy. Keep
    # those pinned datasets usable, but fail closed if a present copy diverges
    # from the registry value that will actually be sent to the agent.
    if "task_instruction" in manifest:
        manifest_instruction = manifest["task_instruction"]
        if not isinstance(manifest_instruction, str):
            raise ValueError("data manifest task_instruction must be a string")
        if manifest_instruction != task.task_instruction:
            raise ValueError(
                "data manifest task_instruction does not match the registry instruction"
            )
    return resolved_revision
