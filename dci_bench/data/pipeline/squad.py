"""SQuAD-format evidence adapter, including original-language TyDiQA-GoldP.

This constructs passage retrieval tasks from QA supervision. It does not claim
to reproduce the native reading-comprehension benchmark or exhaustive qrels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LANGUAGES = {"english": "en", "arabic": "ar", "bengali": "bn", "finnish": "fi",
             "indonesian": "id", "swahili": "sw", "korean": "ko", "russian": "ru", "telugu": "te"}


def load_squad(spec, snapshot: Path, manifest):
    from .connectors import _base_bundle, _find_all_files
    bundle = _base_bundle(spec, manifest)
    options = spec.get("options", {})
    paths = [p for p in _find_all_files(snapshot / "data") if p.suffix == ".json"]
    if not paths:
        raise ValueError("SQuAD connector requires JSON data files")
    docs, seen, exclusions = {}, set(), []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("SQuAD file must contain a data list")
        for ai, article in enumerate(data["data"]):
            title = article.get("title", "")
            for pi, paragraph in enumerate(article["paragraphs"]):
                context = paragraph["context"]
                if not isinstance(context, str) or not context.strip():
                    raise ValueError("Empty or non-text passage")
                for qa in paragraph["qas"]:
                    qid = qa["id"]
                    if not isinstance(qid, str) or not qid or qid in seen:
                        raise ValueError("Duplicate or invalid original QA ID")
                    seen.add(qid)
                    language = qa.get("language", spec.get("language", "unknown"))
                    if options.get("language_from_id") == "tydiqa":
                        language = next((code for name, code in LANGUAGES.items()
                                         if qid.startswith(name + "-")), None)
                        if language is None:
                            raise ValueError(f"Unknown TyDiQA language prefix: {qid}")
                    # QA sources usually have no passage ID. Preserve query IDs;
                    # derive an explicit content identity and retain source location.
                    docid = "passage-" + hashlib.sha256((language + "\0" + context).encode()).hexdigest()
                    if docid not in docs:
                        docs[docid] = {"_id": docid, "title": title, "text": context, "language": language,
                                       "metadata": {"source_file": path.relative_to(snapshot).as_posix(),
                                                    "source_article_index": ai, "source_paragraph_index": pi,
                                                    "source_title": title, "id_derivation": "sha256(language,NUL,context)",
                                                    "translation_status": options.get("translation_status", "unknown"),
                                                    "group_id": "article:" + language + ":" + title}}
                        if options.get("parallel_family"):
                            ids = sorted(q["id"] for q in paragraph["qas"])
                            docs[docid]["metadata"]["family_id"] = str(options["parallel_family"]) + ":passage:" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    answers = qa.get("answers", [])
                    if qa.get("is_impossible") or not answers:
                        exclusions.append({"query_id": qid, "reason": "unanswerable_or_no_span", "source_file": path.name})
                        continue
                    evidence = []
                    for answer in answers:
                        start, quote = answer["answer_start"], answer["text"]
                        if isinstance(start, bool) or not isinstance(start, int) or start < 0 or not isinstance(quote, str) or not quote:
                            raise ValueError(f"Invalid answer span for {qid}")
                        end = start + len(quote)
                        if context[start:end] != quote:
                            raise ValueError(f"Answer offsets do not match original passage: {qid}")
                        row = {"query_id": qid, "doc_id": docid, "quote": quote, "start": start, "end": end,
                               "annotation_source": "source_human", "offset_unit": "unicode_codepoints"}
                        if row not in evidence:
                            evidence.append(row)
                    bundle["queries"].append({"_id": qid, "text": qa["question"], "language": language,
                                               "metadata": {"annotation_source": "source_human",
                                                            "translation_status": options.get("translation_status", "unknown"),
                                                            "group_id": "article:" + language + ":" + title}})
                    if options.get("parallel_family"):
                        bundle["queries"][-1]["metadata"]["family_id"] = str(options["parallel_family"]) + ":query:" + qid
                    bundle["qrels"][qid] = {docid: 1}
                    bundle["evidence"].extend(evidence)
    bundle["documents"] = list(docs.values())
    bundle["provenance"].update({"annotation_source": "source_human", "exclusions": exclusions,
                                  "task_adaptation": "evidence QA to passage retrieval",
                                  "qrels_completeness": "seed positives only; unjudged documents are not verified negatives"})
    return bundle
