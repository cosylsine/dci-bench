#!/usr/bin/env python3
"""Bounded CFQA corporate-report acquisition and page-grounded preparation.

Only CFQA's original 1-based physical PDF page annotations define positives.
Lexical answer anchors can reject a mismatched report/page; they never add a
positive. The frozen 2022 slice is a derived enterprise task, not CFQA's
official financial-answer-generation evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = "ygan/CFQA"
REVISION = "61c9ec3c4335d0411a1735cd228af8b3ead114fc"
SOURCE = "gapfill_cfqa_enterprise_zh"
YEAR = 2022
API = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"}
HAN = re.compile(r"[\u3400-\u9fff]")
SPACE = re.compile(r"\s+")
BUSINESS = re.compile(r"业务|产品|研发|技术|创新|治理|人才|人力|培训|风险|竞争|战略|生产|采购|经营模式|供应链|发展规划|发展计划|组织|企业文化|内部控制|社会责任|环境保护|员工|客户|营销|质量")
QUERY_BUSINESS = re.compile(r"业务|产品|研发|技术|创新|治理|人才|人力|培训|风险|竞争|战略|生产|采购|经营模式|供应链|发展规划|发展计划|企业文化|内部控制")
QUANTITATIVE = re.compile(r"多少|几个|几家|比例|比重|人数|金额|费用|收入|利润|市值|股本|股东|现金|负债|净资产|毛利|占比|增长率|收益率|持股|成本|销量|销售量|生产量|库存量|产销量|产量|组件量|数量|投入|薪酬|余额|账龄|流动性|减值|财务负责人|出货量|受益期限|外币|所得税|信用风险|利率风险|汇率风险|金融工具")
NO_ANSWER = re.compile(r"无法.{0,25}(?:找到|信息|回答)|未.{0,8}(?:提及|披露|提供)|没有.{0,8}(?:信息|提及)|暂无|不适用|无年报|无20\d{2}年报|无法找到答案")
FINANCIAL_HEADING = re.compile(r"^\s*第[一二三四五六七八九十]+节\s*财务报告\s*$", re.MULTILINE)
ALIASES = {"深康佳A": "康佳", "南玻A": "南玻", "深天马A": "天马", "江航装备": "江航"}
POLICY_VERSION = "cfqa-enterprise-physical-pages-20261102-v4"
# v3's source review is immutable and contains the first six findings.  v4
# records the parent review decision separately while retaining that source
# identity and adding the two newly frozen whole-query exclusions.
CONTENT_REVIEW_ID = "cfqa_enterprise_content_review_20261102_v4"
CONTENT_REVIEW_SOURCE_ID = "cfqa_enterprise_content_review_20261102"
DEFAULT_CONTENT_REVIEW = (ROOT / "results/implementation/20261102-zh-gapfill" /
                          "cfqa-content-review-v1/review.json")
MANUAL_REVIEW_QUERY_IDS = (
    "cfqa-train-1305", "cfqa-dev-6252", "cfqa-train-1677",
    "cfqa-train-445", "cfqa-test-493", "cfqa-test-489",
    "cfqa-train-7120", "cfqa-test-2493",
)
# The review file is the authoritative evidence when available.  Keeping this
# fallback makes the CPU preparation and its unit fixtures self-contained while
# preserving the eight frozen whole-query dispositions if the review checkout
# is not mounted.  These are assistant content exclusions; upstream human
# qrels remain unchanged and no replacement gold is created.
MANUAL_REVIEW_FALLBACK = {
    "cfqa-train-1305": {
        "category": "answer_page_numeric_mismatch",
        "severity": "query_level_review_required",
        "reason": "The annotated page gives different R&D counts for all three numeric claims in the original answer.",
        "disposition": "do_not_relabel_positive; exclude whole query; no new gold",
        "annotated_page": "report-000962-2022-1216369624-p0011",
    },
    "cfqa-dev-6252": {
        "category": "answer_page_evidence_gap",
        "severity": "query_level_review_required",
        "reason": "The named pension platforms in the original answer are absent from the annotated page text.",
        "disposition": "do_not_relabel_positive; exclude whole query; no new gold",
        "annotated_page": "report-688579-2022-1216406643-p0012",
    },
    "cfqa-train-1677": {
        "category": "answer_page_evidence_gap",
        "severity": "query_level_review_required",
        "reason": "The annotated page supports a 650V R&D row but not the original answer's 900V marketed-product and 8-inch planning claims.",
        "disposition": "do_not_relabel_positive; exclude whole query; no new gold",
        "annotated_page": "report-688396-2022-1216672203-p0027",
    },
    "cfqa-train-445": {
        "category": "answer_page_evidence_gap",
        "severity": "query_level_review_required",
        "reason": "The annotated page mentions securities companies but has no direct futures reference for the combined securities-and-futures answer.",
        "disposition": "do_not_relabel_positive; exclude whole query; no new gold",
        "annotated_page": "report-688118-2022-1216476529-p0034",
    },
    "cfqa-test-493": {
        "category": "question_answer_scope_gap",
        "severity": "query_level_review_required",
        "reason": "The annotated page supports business types and regions but does not provide the operating scale requested by the question.",
        "disposition": "do_not_relabel_positive; exclude whole query; no new gold",
        "annotated_page": "report-000056-2022-1216705336-p0011",
    },
    "cfqa-test-489": {
        "category": "question_answer_scope_gap",
        "severity": "query_level_review_required",
        "reason": "The original answer describes配套服务 while the question asks for the运营服务 business model; this is a neighboring-template collision.",
        "disposition": "do_not_relabel_positive; exclude whole query; no new gold",
        "annotated_page": "report-000056-2022-1216705336-p0011",
    },
    "cfqa-train-7120": {
        "category": "question_answer_scope_gap",
        "severity": "query_level_review_required",
        "query_snippet": "国电南瑞2022年区块链技术的推广前景如何",
        "answer_snippet": "国电南瑞2022年区块链技术的推广前景：公司以先进的控制技术和信息技术为基础，利用“大数据、云计算、物联网、移动互联、人工智能、区块链”等技术，为电网、发电、轨道交通、水利水务、市政公用、工矿等行业和客户提供软硬件产品、整体解决方案及应用服务。",
        "annotated_page": "report-600406-2022-1216662905-p0016 (physical page 16)",
        "page_snippet": "公司以先进的控制技术和信息技术为基础，利用“大数据、云计算、物联网、移动互联、人工智能、区块链”等技术，为电网、发电、轨道交通、水利水务、市政公用、工矿等行业和客户提供软硬件产品、整体解决方案及应用服务。",
        "reason": "The question asks about blockchain promotion prospects, but the original answer and annotated physical page only describe current technologies, served industries, products, solutions, and services; they provide no forward-looking promotion or prospect evidence.",
        "disposition": "assistant content exclusion; upstream human qrels unchanged; do_not_relabel_positive; no new gold",
    },
    "cfqa-test-2493": {
        "category": "outside_narrative_enterprise_scope",
        "severity": "query_level_review_required",
        "query_snippet": "国林科技2022年生产经营情况如何？",
        "answer_snippet": "国林科技2022年公司实现营业收入 29262.94 万元，同比下降 40.95%；实现利润总额 1841.07 万元，同比下降 78.93%；归属于母公司所有者的净利润 1801.29 万元，同比下降 76.31%；归属于母公司所有者扣除非经常性损益的净利润 1694.46 万元，同比下降 77.09%。报告期末，公司总资产为 170744.43 万元，较期初增长 10.58%，归属于上市公司股东的净资产 124916.46 万元，较期初增长 0.42%。",
        "annotated_page": "report-300786-2022-1216580743-p0011 and report-300786-2022-1216580743-p0015 (physical pages 11 and 15)",
        "page_snippet": "The original answer is dominated by营业收入、利润总额、净利润、总资产及净资产 figures; those financial disclosures appear on pages 11 and 15.",
        "reason": "The production/operations answer is predominantly revenue, profit, assets, and other financial-number reporting, outside the frozen narrative enterprise scope; exclude the whole query and retain no partial page.",
        "disposition": "assistant content exclusion; upstream human qrels unchanged; do_not_relabel_positive; no new gold",
    },
}
SEMANTIC_TEMPLATE_RULES = [
    {"id": "procurement_mode", "pattern": r"采购模式"},
    {"id": "production_mode", "pattern": r"生产模式"},
    {"id": "business_overview", "pattern": r"^(?:主要业务|从事业务情况如何|涉及业务有哪些)$"},
    {"id": "research_direction", "pattern": r"研发.*方向"},
    {"id": "technology_advantage", "pattern": r"技术(?:研发创新)?优势"},
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _review_entry(query_id: str, entry: dict) -> dict:
    normalized = dict(entry)
    source_disposition = normalized.get("disposition")
    normalized.update({
        "query_id": query_id,
        "review_id": CONTENT_REVIEW_ID,
        "source_review_id": CONTENT_REVIEW_SOURCE_ID,
        "review_authority": "assistant_content_review",
        "upstream_human_qrels": "unchanged",
        "exclusion_scope": "whole_query",
        "new_gold_created": False,
        "source_disposition": source_disposition,
        "disposition": "assistant content exclusion; upstream human qrels unchanged; do_not_relabel_positive; no new gold",
    })
    return normalized


def load_content_review(path: Path | None = None) -> tuple[dict[str, dict], str]:
    """Load the immutable source review plus v4's two frozen additions."""
    review_path = Path(path) if path is not None else DEFAULT_CONTENT_REVIEW
    if review_path.exists():
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        if payload.get("review_id") not in {CONTENT_REVIEW_SOURCE_ID, CONTENT_REVIEW_ID}:
            raise ValueError("content review identity differs from the frozen review")
        rows = payload.get("query_level_evidence_followups")
        if not isinstance(rows, list):
            raise ValueError("content review has no query-level evidence follow-ups")
        entries = {row.get("query_id"): row for row in rows
                   if isinstance(row, dict) and row.get("query_id")}
        # The external source review predates v4 and therefore intentionally
        # has only the first six rows.  v4's supplemental rows are frozen in
        # this script and merged without modifying that source checkout.
        for query_id in MANUAL_REVIEW_QUERY_IDS:
            if query_id not in entries and query_id in MANUAL_REVIEW_FALLBACK:
                entries[query_id] = MANUAL_REVIEW_FALLBACK[query_id]
        missing = set(MANUAL_REVIEW_QUERY_IDS) - set(entries)
        if missing:
            raise ValueError(f"content review is missing frozen query IDs: {sorted(missing)}")
        return {query_id: _review_entry(query_id, entries[query_id])
                for query_id in MANUAL_REVIEW_QUERY_IDS}, str(review_path.absolute())
    if path is not None:
        raise FileNotFoundError(f"content review not found: {review_path}")
    return {query_id: _review_entry(query_id, MANUAL_REVIEW_FALLBACK[query_id])
            for query_id in MANUAL_REVIEW_QUERY_IDS}, str(review_path.absolute())


def compact(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]", "", text)


def original_pages(value) -> list[int]:
    if not isinstance(value, list) or not value:
        return []
    if any(not isinstance(group, list) or not group for group in value):
        return []
    flat = [page for group in value for page in group]
    if any(type(page) is not int or page < 1 for page in flat):
        return []
    return sorted(set(flat))


def query_reason(row: dict) -> str | None:
    question, answer = row["问题"], row["答案"]
    if set(re.findall(r"20\d{2}", question)) != {str(YEAR)}:
        return "not_single_frozen_2022_report"
    if not isinstance(answer, str) or not answer.strip() or NO_ANSWER.search(answer):
        return "empty_or_unanswerable"
    pages = original_pages(row.get("答案出自"))
    if not pages:
        return "missing_or_invalid_original_physical_pages"
    if len(pages) > 200:
        return "positive_count_overflow_whole_query"
    if not QUERY_BUSINESS.search(question) or QUANTITATIVE.search(question):
        return "outside_narrative_enterprise_scope"
    if len(HAN.findall(answer)) < 25:
        return "answer_too_short_for_narrative_scope"
    prefix_year = re.search(r"20\d{2}", answer[:35])
    if prefix_year and prefix_year.group() != str(YEAR) and not re.search(r"未来|计划|战略", question):
        return "answer_year_conflicts_with_question"
    if re.search(r"(?:实现|为|达到).{0,20}(?:万元|亿元|元[，。；]|人[。；]|%)", answer) and len(HAN.findall(answer)) < 70:
        return "predominantly_numeric_answer"
    return None


def load_queries(raw: Path) -> tuple[list[dict], list[dict], dict]:
    receipt = json.loads((raw / "acquisition-receipt.json").read_text())
    if receipt["repo"] != DATASET or receipt["revision"] != REVISION:
        raise ValueError("CFQA source revision differs from frozen revision")
    files = {item["path"]: item for item in receipt["files"]}
    rows, exclusions, inputs = [], [], {}
    for split in ("dev", "test", "train"):
        name = f"dataset/split_by_company/split_by_company_{split}.json"
        path = raw / name
        actual = sha(path)
        if actual != files[name]["sha256"] or path.stat().st_size != files[name]["bytes"]:
            raise ValueError(f"source file differs from acquisition receipt: {name}")
        inputs[name] = {"sha256": actual, "bytes": path.stat().st_size}
        for row in json.loads(path.read_text()):
            row = {**row, "original_split": split, "query_id": f"cfqa-{split}-{row['id']}"}
            reason = query_reason(row)
            if reason:
                exclusions.append({"query_id": row["query_id"], "reason": reason})
            else:
                rows.append(row)
    return rows, exclusions, inputs


def query_template(row: dict) -> str:
    text = row["问题"].replace(row["公司"], "")
    text = text.replace(ALIASES.get(row["公司"], row["公司"]), "")
    text = re.sub(r"20\d{2}年?|请问|公司|的|是否|截至", "", text)
    return compact(text)


def semantic_template(row: dict) -> str:
    template = query_template(row)
    for rule in SEMANTIC_TEMPLATE_RULES:
        if re.search(rule["pattern"], template):
            return "semantic:" + rule["id"]
    return "exact:" + template


def lookup(company: str) -> dict:
    import requests
    terms = list(dict.fromkeys((company, ALIASES.get(company, company))))
    attempts, candidates = [], []
    for term in terms:
        form = {"pageNum": "1", "pageSize": "50", "column": "szse", "tabName": "fulltext",
                "searchkey": term, "category": "category_ndbg_szsh;", "seDate": "2023-01-01~2023-12-31",
                "sortName": "time", "sortType": "asc", "isHLtitle": "true"}
        response = requests.post(API, data=form, headers=HEADERS, timeout=45)
        response.raise_for_status()
        payload = response.json()
        attempts.append({"url": API, "form": form, "response": payload})
        for item in payload.get("announcements") or []:
            title = html.unescape(re.sub(r"<[^>]+>", "", item["announcementTitle"]))
            secname = html.unescape(re.sub(r"<[^>]+>", "", item["secName"]))
            if not re.search(r"2022\s*年(?:年度|度)?报告", title):
                continue
            if re.search(r"摘要|英文|取消|更正|修订|更新|英文本", title):
                continue
            if term not in secname and term not in title:
                continue
            candidates.append({"company": company, "identity_term": term, "year": YEAR,
                               "announcement_id": item["announcementId"], "secName": secname,
                               "secCode": item["secCode"], "title": title,
                               "url": "https://static.cninfo.com.cn/" + item["adjunctUrl"],
                               "declared_size_kb": item["adjunctSize"]})
        if candidates:
            break
    candidates = list({row["announcement_id"]: row for row in candidates}.values())
    return {"company": company, "attempts": attempts,
            "selected": candidates[0] if len(candidates) == 1 else None,
            "rejection": None if len(candidates) == 1 else "missing_or_ambiguous_original_annual_report"}


def download(raw: Path, target: Path, *, reuse_dir: Path | None = None,
             max_candidate_companies: int = 30,
             max_bytes: int = 150_000_000,
             extend_from: Path | None = None) -> dict:
    import requests
    if type(max_candidate_companies) is not int or max_candidate_companies < 1:
        raise ValueError("max_candidate_companies must be a positive integer")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    rows, exclusions, inputs = load_queries(raw)
    target.mkdir(parents=True, exist_ok=True)
    plan_path = target / "download-plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["inputs"] != inputs:
            raise ValueError("existing plan is bound to different source files")
    else:
        counts = Counter(row["公司"] for row in rows)
        companies = sorted(counts, key=lambda name: (-counts[name], name))[:max_candidate_companies]
        base_plan = None
        base_companies = []
        base_reports = []
        base_lookups = []
        if extend_from is not None:
            base_plan_path = Path(extend_from) / "download-plan.json"
            base_plan = json.loads(base_plan_path.read_text())
            if base_plan["inputs"] != inputs:
                raise ValueError("base acquisition plan is bound to different source files")
            base_companies = [item["company"] for item in base_plan.get("lookups", [])]
            if not base_companies or len(base_companies) >= max_candidate_companies:
                raise ValueError("base acquisition does not leave an extension range")
            if companies[:len(base_companies)] != base_companies:
                raise ValueError("base acquisition companies are not the frozen candidate prefix")
            base_lookups = list(base_plan.get("lookups", []))
            base_reports = list(base_plan.get("reports", []))
            companies = companies[len(base_companies):]
            companies = companies[:max_candidate_companies - len(base_companies)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            extension_lookups = list(executor.map(lookup, companies))
        lookups = base_lookups + extension_lookups
        reports, excluded_reports, total = [], [], 0
        for item in ([{"selected": report, "company": report["company"],
                       "rejection": None} for report in base_reports] + extension_lookups):
            selected = item["selected"]
            if selected is None:
                excluded_reports.append({"company": item["company"], "reason": item["rejection"]})
                continue
            estimate = int(selected["declared_size_kb"]) * 1024 + 1024
            if total + estimate > max_bytes:
                excluded_reports.append({"company": item["company"],
                                         "reason": f"{max_bytes}-byte_acquisition_budget"})
                continue
            reports.append({**selected, "candidate_queries": counts[item["company"]]})
            total += estimate
        plan = {"source": SOURCE, "revision": REVISION, "inputs": inputs, "lookups": lookups,
                "reports": reports, "excluded_reports": excluded_reports,
                "candidate_count": len(rows), "initial_query_exclusions": exclusions,
                "selection_policy": (("Frozen 2022; candidate count descending, company Unicode ascending; "
                                      f"first {max_candidate_companies} candidate companies; unique original annual report; "
                                      f"cumulative {max_bytes}-byte cap; no replacement based on retrieval results.")
                                     if extend_from is None else
                                     ("Frozen base acquisition prefix plus the next deterministic candidate companies; "
                                      "candidate count descending, company Unicode ascending; unique original annual report; "
                                      f"cumulative {max_bytes}-byte cap; no replacement based on retrieval results.")),
                "candidate_company_limit": max_candidate_companies,
                "max_reports": max_candidate_companies, "max_bytes": max_bytes,
                "estimated_bytes": total,
                "base_acquisition": str(Path(extend_from).absolute()) if extend_from else None,
                "base_plan_sha256": sha(Path(extend_from) / "download-plan.json") if extend_from else None,
                "base_candidate_company_count": len(base_companies),
                "extension_candidate_companies": companies if extend_from else []}
        save(plan_path, plan)

    def fetch(row):
        filename = f"{row['secCode']}-{YEAR}-{row['announcement_id']}.PDF"
        path = target / "pdf" / filename
        receipt_path = target / "receipts" / (filename + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if not path.is_file() or sha(path) != receipt["sha256"]:
                raise ValueError(f"existing PDF differs from receipt: {path}")
            return receipt
        if path.exists():
            raise FileExistsError(f"unsealed existing PDF must be inspected: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        reuse = reuse_dir / filename if reuse_dir else None
        if reuse is not None and reuse.is_file():
            shutil.copyfile(reuse, path)
            reuse_receipt = reuse.parent.parent / "receipts" / (filename + ".json")
            if reuse_receipt.is_file():
                expected = json.loads(reuse_receipt.read_text())
                if expected.get("sha256") != sha(path):
                    raise ValueError(f"reused PDF differs from its acquisition receipt: {reuse}")
        else:
            temporary = path.with_suffix(".PDF.part")
            if temporary.exists():
                index = 1
                while True:
                    retry = path.with_suffix(f".PDF.part.retry-{index}")
                    if not retry.exists():
                        temporary = retry
                        break
                    index += 1
            with requests.get(row["url"], headers=HEADERS, stream=True, timeout=120) as response:
                response.raise_for_status()
                total = 0
                with temporary.open("xb") as out:
                    for block in response.iter_content(1024 * 1024):
                        total += len(block)
                        if total > int(row["declared_size_kb"]) * 1024 + 1024:
                            raise ValueError(f"PDF exceeds frozen size allowance: {row['company']}")
                        out.write(block)
            temporary.rename(path)
        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"download is not PDF: {path}")
        receipt = {**row, "path": str(path.absolute()), "bytes": path.stat().st_size, "sha256": sha(path)}
        save(receipt_path, receipt)
        print(json.dumps({"downloaded": row["company"], "bytes": receipt["bytes"]}, ensure_ascii=False), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(fetch, plan["reports"]))
    result = {"source": SOURCE, "revision": REVISION, "plan_sha256": sha(plan_path),
              "report_count": len(receipts), "bytes": sum(row["bytes"] for row in receipts), "reports": receipts}
    if len(receipts) > int(plan["max_reports"]) or result["bytes"] > int(plan["max_bytes"]):
        raise ValueError("acquisition budget exceeded")
    complete = target / "download-complete.json"
    if complete.exists():
        if json.loads(complete.read_text()) != result:
            raise ValueError("completed acquisition changed")
    else:
        save(complete, result)
    return result


def page_body(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(line.strip() for line in lines
                     if line.strip() and not re.fullmatch(r"\s*\d+(?:\s*/\s*\d+)?\s*", line)
                     and not re.fullmatch(r".*2022\s*年(?:度|年度)?报告(?:全文)?\s*", line)).strip()


def page_reason(text: str, *, in_financial_section: bool) -> str | None:
    if in_financial_section:
        return "financial_statement_section"
    han = len(HAN.findall(text))
    if han < 120:
        return "too_little_chinese_prose"
    if len(re.findall(r"\d", text)) / max(1, len(compact(text))) > .22:
        return "numeric_table_dominated"
    if not BUSINESS.search(text):
        return "no_enterprise_content_anchor"
    if len(re.findall(r"[，。；：]", text)) < 3:
        return "not_narrative_page"
    if re.search(r"^\s*目\s*录\s*$", text, re.MULTILINE):
        return "table_of_contents"
    return None


def collapse_pages(documents: list[dict]) -> tuple[list[dict], dict[str, str], list[dict]]:
    groups = defaultdict(list)
    for row in documents:
        groups[row["text"]].append(row)
    kept, mapping, aliases = [], {}, []
    for rows in groups.values():
        rows.sort(key=lambda row: row["_id"])
        canonical = dict(rows[0])
        originals = [row["_id"] for row in rows]
        canonical["metadata"] = {**canonical["metadata"], "original_page_ids": originals,
                                 "source_aliases": [row["metadata"] for row in rows]}
        kept.append(canonical)
        mapping.update({key: canonical["_id"] for key in originals})
        if len(rows) > 1:
            aliases.append({"canonical_id": canonical["_id"], "original_page_ids": originals})
    return sorted(kept, key=lambda row: row["_id"]), mapping, aliases


def anchor_present(answer: str, page: str) -> bool:
    answer, page = compact(answer), compact(page)
    return any(answer[start:start + 12] in page for start in range(max(0, len(answer) - 11))
               if len(HAN.findall(answer[start:start + 12])) >= 8)


def prepare(raw: Path, acquisition: Path, output: Path, evidence: Path,
            *, content_review: Path | None = None) -> dict:
    import pymupdf
    from dci_bench.data.benchmark_v2 import (
        SamplingShortageError, sample_global_query_splits, sample_query_splits,
    )
    from dci_bench.data.pipeline.core import publish_bundle, load_bundle, verify_release
    if output.exists() or evidence.exists():
        raise FileExistsError("use new output and evidence directories")
    review_entries, review_path = load_content_review(content_review)
    rows, exclusions, inputs = load_queries(raw)
    downloaded = json.loads((acquisition / "download-complete.json").read_text())
    plan = json.loads((acquisition / "download-plan.json").read_text())
    if inputs != plan["inputs"] or downloaded["plan_sha256"] != sha(acquisition / "download-plan.json"):
        raise ValueError("acquisition/source identity mismatch")
    documents, page_exclusions, parse_reports = [], [], []
    page_ids, page_texts, report_for = {}, {}, {}
    for report in downloaded["reports"]:
        path = Path(report["path"])
        if sha(path) != report["sha256"] or path.stat().st_size != report["bytes"]:
            raise ValueError(f"PDF identity changed: {path}")
        with pymupdf.open(path) as pdf:
            texts = [page.get_text() for page in pdf]
        cover = compact(" ".join(texts[:2]))
        if compact(report["identity_term"]) not in cover or str(YEAR) not in cover:
            parse_reports.append({"company": report["company"], "status": "identity_mismatch", "pages": len(texts)})
            continue
        report_for[report["company"]] = report
        financial_start = next((i for i, text in enumerate(texts, 1) if FINANCIAL_HEADING.search(text)), None)
        if financial_start is None:
            parse_reports.append({"company": report["company"], "status": "missing_financial_section_boundary", "pages": len(texts)})
            continue
        retained, parsed = 0, 0
        for physical, text in enumerate(texts, 1):
            if len(HAN.findall(text)) >= 20:
                parsed += 1
            body = page_body(text)
            reason = page_reason(body, in_financial_section=physical >= financial_start)
            did = f"report-{report['secCode']}-{YEAR}-{report['announcement_id']}-p{physical:04d}"
            page_ids[(report["company"], physical)] = did
            if reason:
                page_exclusions.append({"document_id": did, "reason": reason})
                continue
            page_texts[did] = body
            retained += 1
            documents.append({"_id": did, "title": f"{report['company']} {YEAR}年年度报告 PDF第{physical}页",
                              "text": body, "language": "zh", "metadata": {
                                  "source_id": SOURCE, "original_document_id": did,
                                  "company": report["company"], "year": YEAR, "physical_page_1based": physical,
                                  "pdf_url": report["url"], "pdf_sha256": report["sha256"],
                                  "announcement_id": report["announcement_id"], "source_secCode": report["secCode"],
                                  "raw_page_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "domain": "enterprise", "translation_status": "native",
                                  "annotation_source": "source", "extraction": "PyMuPDF physical page; header/page-number line removal only"}})
        parse_reports.append({"company": report["company"], "status": "parsed", "pages": len(texts),
                              "chinese_text_pages": parsed, "parse_rate": parsed / len(texts),
                              "financial_section_start": financial_start, "retained_enterprise_pages": retained})
    documents, mapping, aliases = collapse_pages(documents)
    queries, qrels, retention, answer_page_checks = [], {}, [], []
    for row in sorted(rows, key=lambda row: (row["公司"], row["query_id"])):
        company, qid = row["公司"], row["query_id"]
        pages = original_pages(row["答案出自"])
        ids = [page_ids.get((company, page)) for page in pages]
        reason = None
        if company not in report_for:
            reason = "report_not_selected_or_identity_rejected"
        elif any(did not in mapping for did in ids):
            reason = "missing_or_filtered_positive_page_whole_query"
        else:
            review = review_entries.get(qid)
            if review is not None:
                reason = "manual_content_review_whole_query_exclusion"
            elif any(not anchor_present(row["答案"], page_texts[did]) for did in ids):
                reason = "answer_anchor_absent_on_original_positive_page_whole_query"
        answer_page_checks.append({
            "query_id": qid, "original_pages": pages, "original_page_ids": ids,
            "status": "excluded" if reason else "pass",
            "reason": reason,
            "review_category": review_entries.get(qid, {}).get("category") if qid in review_entries else None,
            "anchor_checked": reason != "manual_content_review_whole_query_exclusion",
        })
        template = query_template(row)
        semantic = semantic_template(row)
        if reason:
            item = {"query_id": qid, "reason": reason, "original_pages": pages,
                    "original_page_ids": ids, "template": template,
                    "semantic_template": semantic}
            if qid in review_entries:
                item.update({"exclusion_scope": "whole_query", "new_gold_created": False,
                             "review_id": CONTENT_REVIEW_ID, "review_path": review_path,
                             "review_evidence": review_entries[qid]})
            exclusions.append(item)
            continue
        report = report_for[company]
        group = "report-" + hashlib.sha256(f"{company}|{YEAR}".encode()).hexdigest()[:20]
        text = row["问题"] if company in row["问题"] else f"{company}（{YEAR}年年度报告）：{row['问题']}"
        queries.append({"_id": qid, "text": text, "language": "zh", "group_id": group,
                        "metadata": {"original_text": row["问题"], "original_answer": row["答案"],
                                     "original_id": row["id"], "original_split": row["original_split"],
                                     "original_stock_code": row["股票代码"], "company": company, "year": YEAR,
                                     "verified_secCode": report["secCode"], "original_pages": pages,
                                     "original_page_annotation": row["答案出自"], "query_template": template,
                                     "semantic_template": semantic, "report_group_id": group,
                                     "source_id": SOURCE, "domain": "enterprise",
                                     "annotation_source": "source", "translation_status": "native",
                                     "group_id": group}})
        qrels[qid] = {mapping[did]: 1 for did in ids}
        for page, did in zip(pages, ids):
            retention.append({"query_id": qid, "original_split": row["original_split"],
                              "original_page_1based": page, "original_document_id": did,
                              "canonical_document_id": mapping[did], "grade": 1,
                              "label_source": "original CFQA 答案出自 physical-page annotation",
                              "anchor_used_only_for_exclusion": True})
    # Report identity, equivalent question templates, and positive aliases
    # form connected groups; none may cross the derived dev/test boundary.
    parent = {q["group_id"]: q["group_id"] for q in queries}
    def root(group):
        while parent[group] != group:
            group = parent[group]
        return group
    owners = {}
    for query in queries:
        links = ["template:" + query["metadata"]["semantic_template"],
                 *["positive:" + did for did in qrels[query["_id"]]]]
        for link in links:
            if link in owners:
                left, right = sorted((root(query["group_id"]), root(owners[link])))
                parent[right] = left
            owners[link] = query["group_id"]
    for query in queries:
        query["group_id"] = root(query["group_id"])
        query["metadata"]["group_id"] = query["group_id"]
    sampling, global_sampling, blockers = None, None, []
    if len(documents) < 1000:
        blockers.append("fewer_than_1000_natural_enterprise_pages")
    if len(queries) < 60:
        blockers.append("fewer_than_60_all_positive_retained_queries")
    if len(queries) >= 60:
        try:
            sampling = sample_query_splits(queries, dev_size=10, test_size=50, smoke_size=1,
                                           seed=42, task_id="enterprise_zh", group_key="group_id")
        except SamplingShortageError as error:
            blockers.append("seed42_group_split_infeasible: " + str(error))
        try:
            global_sampling = sample_global_query_splits(
                {"enterprise_zh": queries},
                source_context_by_task={"enterprise_zh": {
                    "source_id": SOURCE, "source_family": "CFQA",
                }},
                dev_size=10, test_size=50, smoke_size=1, seed=42,
            )
        except SamplingShortageError as error:
            blockers.append("seed42_global_split_infeasible: " + str(error))
    evidence.mkdir(parents=True)
    save(evidence / "query-exclusions.json", exclusions)
    save(evidence / "content-review-query-exclusions.json", {
        "review_id": CONTENT_REVIEW_ID, "review_path": review_path,
        "source_review_id": CONTENT_REVIEW_SOURCE_ID,
        "review_authority": "assistant_content_review",
        "upstream_human_qrels": "unchanged",
        "policy_version": POLICY_VERSION, "whole_query": True,
        "new_gold_created": False,
        "query_exclusions": [
            {"query_id": query_id, **review_entries[query_id]}
            for query_id in MANUAL_REVIEW_QUERY_IDS
        ],
    })
    save(evidence / "answer-page-checks.json", {
        "policy": "Every eligible query checks each original annotated physical page; the answer anchor may only exclude, never add a label.",
        "review_id": CONTENT_REVIEW_ID, "review_path": review_path,
        "source_review_id": CONTENT_REVIEW_SOURCE_ID,
        "review_authority": "assistant_content_review",
        "upstream_human_qrels": "unchanged",
        "checks": answer_page_checks,
    })
    save(evidence / "page-exclusions.json", page_exclusions)
    save(evidence / "exact-body-aliases.json", aliases)
    save(evidence / "original-positive-retention.json", retention)
    save(evidence / "pdf-parse-report.json", parse_reports)
    save(evidence / "template-groups.json", {
         "rules": SEMANTIC_TEMPLATE_RULES,
         "policy": "Union report groups, explicit semantic templates, remaining exact templates, and shared positive aliases; no template-based query deletion.",
         "queries": [{"query_id": q["_id"], "original_split": q["metadata"]["original_split"],
                      "exact_template": q["metadata"]["query_template"],
                      "semantic_template": q["metadata"]["semantic_template"],
                      "report_group_id": q["metadata"]["report_group_id"], "group_id": q["group_id"]}
                     for q in queries]})
    save(evidence / "sampling-preview.json", {
        "seed": 42, "dev_size": 10, "test_size": 50, "smoke_size": 1,
        "splits": sampling, "global_splits": global_sampling,
        "groups": dict(Counter(q["group_id"] for q in queries)),
        "policy": "company/report plus explicit semantic/exact question templates and shared positive aliases; all eligible queries retained",
    })
    result = {"source": SOURCE, "revision": REVISION, "status": "blocked" if blockers else "ready",
              "blockers": blockers, "documents": len(documents), "queries": len(queries),
              "positive_pairs": sum(map(len, qrels.values())), "original_positive_pairs": len(retention),
              "groups": len({q['group_id'] for q in queries}), "reports": len(downloaded["reports"]), "downloaded_bytes": downloaded["bytes"],
              "inputs": inputs, "acquisition_sha256": sha(acquisition / "download-complete.json"),
              "page_exclusion_counts": dict(Counter(r["reason"] for r in page_exclusions)),
              "query_exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
              "native_language_evidence": f"https://github.com/ygan/CFQA/blob/{REVISION}/README.md",
              "content_review_id": CONTENT_REVIEW_ID,
              "content_review_source_id": CONTENT_REVIEW_SOURCE_ID,
              "content_review_authority": "assistant_content_review",
              "upstream_human_qrels": "unchanged",
              "content_review_exclusions": len(MANUAL_REVIEW_QUERY_IDS),
              "content_review_path": review_path,
              "answer_page_checks": len(answer_page_checks),
              "global_sampling": global_sampling,
              "policy_version": POLICY_VERSION,
              "boundary": "CPU source/page/sampling preparation; no KaLM ranking, pool, GPU or model benchmark"}
    if not blockers:
        bundle = {"schema_version": "dci-data-bundle-v1", "input_kind": "retrieval",
                  "source": {"dataset": DATASET, "revision": REVISION}, "documents": documents,
                  "queries": queries, "qrels": qrels, "evidence": [], "judgments": [],
                  "provenance": {"source_id": SOURCE, "source_family": "CFQA", "domain": "enterprise",
                     "language": "zh", "translation_status": "native", "label_source": "source",
                     "license": "MIT for CFQA annotations; corporate PDF disclosures have no separate license asserted",
                     "original_split_policy": "train/dev/test retained in host query metadata; derived DCI splits isolate reports",
                     "qrels_semantics": "binary relevance for every original annotated physical PDF page; all positives retained under exact-text aliases",
                     "answer_anchor_policy": "12-character overlap containing at least 8 Chinese characters can only exclude a query; never creates labels",
                     "answer_page_consistency_policy": "Every eligible original query is checked against every annotated physical page; obvious answer/page conflicts exclude the whole query and never create replacement gold",
                     "content_review_id": CONTENT_REVIEW_ID,
                     "content_review_source_id": CONTENT_REVIEW_SOURCE_ID,
                     "content_review_authority": "assistant_content_review",
                     "upstream_human_qrels": "unchanged",
                     "content_review_exclusions": list(MANUAL_REVIEW_QUERY_IDS),
                     "content_review_path": review_path,
                     "policy_version": POLICY_VERSION,
                     "body_policy": "physical pages before financial-report section; Chinese narrative enterprise content; no synthetic/padded documents",
                     "task_instruction": "根据企业年度报告检索回答企业业务、研发、治理、人才与经营风险问题所需的文档页。",
                     "evidence_directory": str(evidence.absolute()), "acquisition_receipt": str((acquisition / "download-complete.json").absolute())}}
        publish_bundle(bundle, output, specification={"id": SOURCE, "input_kind": "retrieval", "domain": "enterprise", "language": "zh"},
                       policy={"max_positive_per_query": 200, "drop_near_duplicates": False})
        actual = load_bundle(output)
        if actual["qrels"] != qrels or actual["queries"] != queries or actual["documents"] != documents:
            raise ValueError("pipeline preparation changed the audited rows")
        manifest = verify_release(output)
        if manifest["state"] != "labels_ready":
            raise ValueError("published source is not labels_ready")
        result.update(release=str(output.absolute()), release_state=manifest["state"], bundle_hash=manifest["bundle_hash"])
    save(evidence / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "prepare"))
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--reuse-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--content-review", type=Path)
    parser.add_argument("--max-candidate-companies", type=int, default=30)
    parser.add_argument("--max-bytes", type=int, default=150_000_000)
    parser.add_argument("--extend-from", type=Path)
    args = parser.parse_args()
    if args.action == "download":
        result = download(args.raw_root, args.acquisition, reuse_dir=args.reuse_dir,
                          max_candidate_companies=args.max_candidate_companies,
                          max_bytes=args.max_bytes, extend_from=args.extend_from)
        print(json.dumps({k: v for k, v in result.items() if k != "reports"}, ensure_ascii=False))
        return 0
    else:
        if args.output is None or args.evidence is None:
            parser.error("prepare requires --output and --evidence")
        result = prepare(args.raw_root, args.acquisition, args.output, args.evidence,
                         content_review=args.content_review)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
