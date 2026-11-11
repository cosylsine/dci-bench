#!/usr/bin/env python3
"""Publish the 2026-10-31 gapfill report after sealed receipts are ready.

This entry point performs the small boundary checks needed before calling the
main repository's ``update()`` renderer. Manifest/audit hashes, query IDs,
pool capacities, and suite body checks remain owned by the existing generators
and ``verify_suite.py``. Running this script is the publishing action; this
file itself never downloads data or runs inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import re
import sys
from pathlib import Path


CHINA_TZ = timezone(timedelta(hours=8))
BLOCKED = {"software_zh", "enterprise_zh"}
SOURCES = {
    "gapfill_arguana_en": (8674, 1183, 1183),
    "gapfill_android_en": (22998, 697, 1432),
    "gapfill_browsecomp_plus_en": (99930, 825, 5023),
    "gapfill_acord_en_v3": (3931, 63, 676),
    "gapfill_cedar_zh_v2": (25794, 241, 463),
}


def load(path: Path, label: str):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def publish(repo: Path) -> None:
    root = repo.resolve()
    gap = root / "results/implementation/20261031-gapfill"
    old = load(root / "results/implementation/20261029-lite-full/status.json", "old status")
    prepared = load(gap / "prepared-manifest.json", "prepared manifest")
    audit = load(gap / "data-audit.json", "data audit")
    suite = load(gap / "suite-validation.json", "suite validation")
    quality = load(gap / "retrieval-quality.json", "retrieval quality")
    preflight = load(gap / "data-preflight/preflight.json", "data preflight")
    review = load(gap / "source-review.json", "source review")
    preview = load(gap / "sampling-preview.json", "sampling preview")
    unit_log_path = gap / "data-unit-tests.log"
    if not unit_log_path.is_file() or not unit_log_path.read_text(encoding="utf-8").strip():
        raise ValueError(f"missing or empty data unit test log: {unit_log_path}")
    unit_log = unit_log_path.read_text(encoding="utf-8")

    tasks = prepared["tasks"]
    if len(tasks) != 20:
        raise ValueError(f"prepared task count is {len(tasks)}, expected 20")
    ready = {task for task, row in tasks.items() if row["status"] == "ready"}
    blocked = {task for task, row in tasks.items() if row["status"] == "blocked"}
    if len(ready) != 18 or blocked != BLOCKED:
        raise ValueError(f"prepared boundary is ready={len(ready)}, blocked={sorted(blocked)}")
    if audit["task_count"] != 20 or audit["ready_for_formal"] is not False:
        raise ValueError("data audit is not the expected 20-task data-only boundary")

    expected_suite = {
        "old_ready": 13,
        "new_status": {"ready": 18, "blocked": 2},
        "stable_old_ready_tasks": 13,
        "dev_groups": len({row["query_groups"][qid] for row in tasks.values() if row["status"] == "ready" for qid in row["query_ids"]["dev"]}),
        "test_groups": len({row["query_groups"][qid] for row in tasks.values() if row["status"] == "ready" for qid in row["query_ids"]["test"]}),
        "pool_records_checked": 2160,
        "self_match_records_checked": 240,
        "acord_train_excluded": 51,
    }
    if suite != expected_suite:
        raise ValueError(f"suite validation differs from verify_suite receipt: {suite}")

    if (quality["embedding_gpu_inference_completed"] is not True
            or quality["benchmark_model_inference_completed"] is not False):
        raise ValueError("retrieval quality has an unexpected inference boundary")
    quality_rows = {row["source"]: row for row in quality["sources"]}
    if set(quality_rows) != set(SOURCES):
        raise ValueError("retrieval quality source set differs from the five frozen sources")
    for source, counts in SOURCES.items():
        row = quality_rows[source]
        if tuple(row[key] for key in ("documents", "queries", "positive_pairs")) != counts:
            raise ValueError(f"retrieval quality counts differ for {source}")
        if row["content_hashes_verified"] is not True:
            raise ValueError(f"content hashes are not verified for {source}")

    if review["benchmark_model_inference_completed"] is not False:
        raise ValueError("source review claims benchmark model inference")
    review_rows = {row["source"]: row for row in review["sources"]}
    if set(review_rows) != set(SOURCES):
        raise ValueError("source review source set differs from the five frozen sources")
    for source, counts in SOURCES.items():
        row = review_rows[source]
        if tuple(row[key] for key in ("documents", "queries", "positive_pairs")) != counts:
            raise ValueError(f"source review counts differ for {source}")
    dqa = review["software_zh"]
    if (dqa["status"] != "blocked" or dqa["native_provenance_verified"] is not False
            or dqa["general_knowledge_with_tags"] != {
                "documents": 7481, "positive_pairs": 7490, "queries": 7490}):
        raise ValueError("software_zh DQA boundary changed")
    if review["enterprise_zh"]["status"] != "deferred":
        raise ValueError("enterprise_zh must remain deferred")

    if (preflight["jobs"] != 36 or preflight["model_calls"] != 0
            or preflight["status"] != "data_preflight_passed"):
        raise ValueError("data preflight is not the expected 36-job zero-call receipt")
    if (preview["ready_source_count"] != 18
            or preview["old_13_query_selections_unchanged"] is not True
            or preview["group_leakage"] is not False):
        raise ValueError("sampling preview boundary changed")
    test_counts = re.findall(r"(\d+) passed", unit_log)
    if (not test_counts or int(test_counts[-1]) < 59 or "FAILED tests/" in unit_log
            or "ERROR:" in unit_log or re.search(r"\b[1-9]\d* (?:failed|errors?)\b", unit_log)):
        raise ValueError("data unit tests did not complete successfully")

    old_notes = {row["task"]: row for row in old["data_tasks"]}
    details = {
        "argumentation_en": f"ArguAna {SOURCES['gapfill_arguana_en'][0]}/{SOURCES['gapfill_arguana_en'][1]}/{SOURCES['gapfill_arguana_en'][2]}；排除218歧义query，topic分组并排除query自身；池审计通过。",
        "software_en": f"Android {SOURCES['gapfill_android_en'][0]}/{SOURCES['gapfill_android_en'][1]}/{SOURCES['gapfill_android_en'][2]}；排除78472歧义与19971的262正例溢出；池审计通过。",
        "legal_en": f"ACORD {SOURCES['gapfill_acord_en_v3'][0]}/{SOURCES['gapfill_acord_en_v3'][1]}/{SOURCES['gapfill_acord_en_v3'][2]}；仅official valid6+test57，排除train51；3/4/5星→1/2/3；DCI采样非官方评分；池审计通过。",
        "argumentation_zh": f"CEDAR {SOURCES['gapfill_cedar_zh_v2'][0]}/{SOURCES['gapfill_cedar_zh_v2'][1]}/{SOURCES['gapfill_cedar_zh_v2'][2]}；显式support/against边，整题排歧义、exact-text alias、每topic一题、排除自身；unjudged不作negative；池审计通过。",
        "web_en": f"BrowseComp+ {SOURCES['gapfill_browsecomp_plus_en'][0]}/{SOURCES['gapfill_browsecomp_plus_en'][1]}/{SOURCES['gapfill_browsecomp_plus_en'][2]}；官方evidence关系和超1MB整题排除已审；池审计通过。",
    }
    blocked_details = {
        "software_zh": "DQA公开样例7490 query / 7481 answer / 7490 QA；论文记录翻译/生成来源且无逐行native provenance，blocked。",
        "enterprise_zh": "中文企业知识检索暂缓；尚无通过同任务语义、原生中文 corpus/query/qrels 与候选池审计的来源。",
    }
    data_tasks = []
    for task in sorted(tasks):
        row = tasks[task]
        if task in BLOCKED:
            detail = blocked_details[task]
            data_tasks.append({"task": task, "status": "阻塞", "source": detail.split("；", 1)[0], "detail": detail})
            continue
        source = row["source_id"]
        detail = details.get(task, old_notes[task]["detail"])
        data_tasks.append({"task": task, "status": "已准备", "source": source, "detail": detail})

    status = {
        "as_of": datetime.now(CHINA_TZ).date().isoformat(),
        "summary": "数据准备18/20任务；开发180/200、正式900/1,000；五个新源已完成源审查、排名、候选池与suite验收，software_zh与enterprise_zh仍阻塞，正式模型评测未执行。",
        "stages": [
            {"name": "五源来源审查与版本冻结", "status": "通过", "detail": "BrowseComp+、ArguAna、Android、ACORD、CEDAR 的 source counts、pin、关系边界和限制已由 source-review 核验。"},
            {"name": "候选池与采样元数据", "status": "已验收", "detail": "18/20任务具备 seed42 的 dev10/test50/smoke1 与 Lite200/Full1000 元数据；新五源 Lite/Full 正文hash校验与旧13任务历史证据分开。"},
            {"name": "suite 一致性审计", "status": "通过（metadata-only）", "detail": "历史13个 ready 任务保持稳定；新边界为 ready18/blocked2。"},
            {"name": "数据预检与单元测试", "status": "通过", "detail": "36 个 data-preflight jobs、0 model calls；59项数据单元测试通过。"},
            {"name": "历史 CPU/并发链路", "status": "历史记录保留，未重跑", "detail": "仅复用 2026-10-29 timing_rows；CPU/并发实现未改动，也未重新执行历史20次 CPU 链路。"},
            {"name": "GPU/API 标定与正式评测", "status": "未执行", "detail": "新实际模型调用仅限 KaLM 数据编码；没有 GPU/API agent 评测标定或正式 benchmark model inference。"},
        ],
        "limitations": [
            "software_zh 保持 blocked（DQA 公开样例 7,490 query / 7,481 answer / 7,490 QA，缺逐行 native provenance）；enterprise_zh 暂缓。",
            "正式模型评测、GPU/API agent 标定和质量分数尚未执行；KaLM 编码只用于数据准备。",
            "新五源 suite audit 当前是 metadata-only；其 Full pool 正文hash校验与旧13任务历史证据分开保存。",
            "ACORD使用官方valid6+test57重新采样，排除train51，不等于官方ACORD测试协议；CEDAR仅以显式支持/反对关系构造qrels，未标注关系不是负例。",
        ],
        "data_tasks": data_tasks,
        "evidence": [
            {"label": "新 prepared manifest", "path": "../../results/implementation/20261031-gapfill/prepared-manifest.json"},
            {"label": "新数据审计", "path": "../../results/implementation/20261031-gapfill/data-audit.json"},
            {"label": "五源 suite validation", "path": "../../results/implementation/20261031-gapfill/suite-validation.json"},
            {"label": "五源 retrieval quality", "path": "../../results/implementation/20261031-gapfill/retrieval-quality.json"},
            {"label": "数据预检（36 jobs，无模型调用）", "path": "../../results/implementation/20261031-gapfill/data-preflight/preflight.json"},
            {"label": "18源抽样与 group leakage 预检查（非最终 pool）", "path": "../../results/implementation/20261031-gapfill/sampling-preview.json"},
            {"label": "数据单元测试", "path": "../../results/implementation/20261031-gapfill/data-unit-tests.log"},
            {"label": "五源与中文缺口 source review", "path": "../../results/implementation/20261031-gapfill/source-review.json"},
            {"label": "历史 CPU 阶段计时（未重跑）", "path": "../../results/implementation/20261029-lite-full/cpu-stage-observations.json"},
            {"label": "Gapfill 数据接入与限制", "path": "../gapfill_20261031.md"},
        ],
        "timing_rows": old["timing_rows"],
        "data_preparation": {
            "embedding_gpu_inference_completed": True,
            "benchmark_model_inference_completed": False,
            "note": "KaLM embedding GPU inference is data preparation only; no agent benchmark model/API call was run.",
        },
    }
    sys.path.insert(0, str(root))
    from scripts.update_benchmark_progress import update
    update(root / "docs/project-progress-20261028", status,
           config_path=root / "configs/benchmark_v2.json",
           prepared_path=gap / "prepared-manifest.json",
           audit_path=gap / "data-audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    publish(parser.parse_args().repo_root)
