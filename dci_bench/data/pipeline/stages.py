"""Bounded model stages and disagreement mining, kept separate from acquisition."""
from __future__ import annotations

import copy
from pathlib import Path
import uuid

from .core import digest, load_bundle, publish_bundle, read_json, verify_release, write_json


def _new_output(path: Path) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")
    if any(p.is_symlink() for p in path.parents):
        raise ValueError("Output cannot have symlink ancestors")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def mine_review(release: Path, output: Path, *, top_k: int = 20,
                dense_files: list[Path] | None = None) -> dict:
    """Union independent retrieval views and all seed positives; unknown stays unknown."""
    from .retrieval import BM25Index
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be positive")
    output = _new_output(output)
    bundle = load_bundle(release)
    if not bundle["queries"]:
        raise ValueError("Mining requires queries; generate or import them first")
    index = BM25Index(bundle["documents"])
    known = {d["_id"] for d in bundle["documents"]}
    qids = {q["_id"] for q in bundle["queries"]}
    views = []
    for path in dense_files or []:
        view = read_json(path)
        if view.get("bundle_hash") != digest(bundle):
            raise ValueError("Dense mining view belongs to a different bundle")
        if not isinstance(view.get("results"), dict) or set(view["results"]) != qids:
            raise ValueError("Dense mining view query IDs mismatch")
        views.append(view)
    candidates, disagreements = {}, []
    for query in bundle["queries"]:
        qid = query["_id"]
        lexical = index.search(query["text"], top_k=top_k)["ranked_doc_ids"]
        ordered = list(bundle["qrels"].get(qid, {})) + lexical
        sets = [set(lexical)]
        for view in views:
            ids = view["results"][qid]["ranked_doc_ids"][:top_k]
            if not isinstance(ids, list) or any(d not in known for d in ids):
                raise ValueError("Dense view contains unknown document IDs")
            ordered.extend(ids)
            sets.append(set(ids))
        candidates[qid] = list(dict.fromkeys(ordered))
        union = set().union(*sets)
        intersection = set.intersection(*sets)
        disagreements.append({"query_id": qid, "retriever_disagreement": sorted(union - intersection),
                              "unjudged_candidates": [d for d in candidates[qid] if d not in bundle["qrels"].get(qid, {})]})
    result = {"schema_version": "dci-review-candidates-v1", "bundle_hash": digest(bundle),
              "candidate_doc_ids": candidates, "top_k_per_retriever": top_k,
              "retrievers": ["unicode-bm25"] + [v["model"] for v in views],
              "review": disagreements, "unjudged_is_negative": False}
    write_json(output, result)
    return {"output": str(output), "queries": len(candidates),
            "candidate_pairs": sum(map(len, candidates.values())), "dense_views": len(views)}


def run_model_stage(release: Path, output: Path, *, role: str, models_config: Path,
                    cache_dir: Path, stage_config: dict, max_calls: int = 50,
                    max_total_tokens: int = 200000, candidates: Path | None = None) -> dict:
    from .models import LocalChatClient, generate_queries, judge_queries
    if role not in ("generator", "judge"):
        raise ValueError("Unknown model role")
    output = _new_output(output)
    manifest = verify_release(release)
    bundle = load_bundle(release)
    models = read_json(models_config)
    model = models["roles"][role]
    if max_calls <= 0 or max_total_tokens <= 0:
        raise ValueError("Inference budgets must be positive")
    # Identity is explicit: config digest does not claim to hash large model weights.
    revision = model.get("weight_revision") or "deployment-config:" + digest(model)
    client = LocalChatClient(model["base_url"], model["served_model_name"], cache_dir=cache_dir,
                             max_calls=max_calls, max_total_tokens=max_total_tokens,
                             revision=revision, expected_server_model=model["served_model_name"], use_proxy=False)
    options = copy.deepcopy(stage_config)
    if role == "generator":
        if not any(k in options for k in ("selected_doc_ids", "doc_ids", "document_ids", "evidence_set")):
            raise ValueError("Generation requires explicit document IDs or evidence_set in --stage-config")
        result = generate_queries(bundle, client, options)
    else:
        if candidates is None:
            raise ValueError("Judging requires --candidates from mine-review")
        plan = read_json(candidates)
        if plan.get("bundle_hash") != digest(bundle):
            raise ValueError("Candidate plan is stale or belongs to a different release")
        mapping = plan["candidate_doc_ids"]
        qids = {q["_id"] for q in bundle["queries"]}
        dids = {d["_id"] for d in bundle["documents"]}
        if not qids or not isinstance(mapping, dict) or set(mapping) != qids:
            raise ValueError("Candidate plan must cover every query exactly")
        for qid, ids in mapping.items():
            if (not isinstance(ids, list) or not ids or any(not isinstance(d, str) for d in ids)
                    or len(set(ids)) != len(ids) or set(ids) - dids):
                raise ValueError(f"Invalid or empty candidates for {qid}")
            if set(bundle["qrels"].get(qid, {})) - set(ids):
                raise ValueError(f"Candidate plan omits seed positives for {qid}")
        batch = options.pop("candidate_batch_size", 4)
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError("candidate_batch_size must be positive")
        result = bundle
        for qid, ids in mapping.items():
            for start in range(0, len(ids), batch):
                result = judge_queries(result, client, {**options, "query_ids": [qid],
                    "candidate_doc_ids": {qid: ids[start:start + batch]}})
    result.setdefault("provenance", {})["model_stage"] = {
        "role": role, "model_id": model["model_id"], "deployment_identity": revision,
        "weight_bytes_verified": False,
        "calls": client.call_count, "usage_tokens": client.total_tokens,
        "audit": client.audit_log, "max_calls": max_calls, "max_total_tokens": max_total_tokens,
    }
    return publish_bundle(result, output, specification=manifest["specification"],
                          raw_snapshot=Path(manifest["raw_snapshot"]) if manifest.get("raw_snapshot") else None,
                          parent_release=release)


def rank_source(output: Path, *, suite: Path | None = None, task: str | None = None,
                release: Path | None = None, cache_dir: Path, model_path: str | None = None,
                device: str = "cpu", batch_size: int = 8, max_length: int = 8192,
                top_k: int = 1000, audit_model: bool = False,
                query_prompt: str | None = None, document_prompt: str | None = None,
                cache_shard_size: int = 0, search_device: str = "cpu",
                search_query_batch_size: int = 32,
                verify_content_hashes: bool = True,
                weight_identity: str | None = None) -> dict:
    """Real local embeddings; the saved official rankings remain bound to prepared DCI bytes."""
    from .retrieval import (EmbeddingConfig, SentenceTransformerEncoder, dense_search,
                            export_saved_retrieval, load_prepared_source)
    output = _new_output(output)
    if (release is None) == (suite is None):
        raise ValueError("Choose exactly one of --release or --source-suite")
    if suite is not None and not task:
        raise ValueError("--source-suite requires --task")
    if suite is not None and audit_model:
        raise ValueError("An audit retriever cannot produce official candidate-pool rankings")
    if not isinstance(verify_content_hashes, bool):
        raise ValueError("verify_content_hashes must be boolean")
    if weight_identity is not None and (not isinstance(weight_identity, str) or not weight_identity):
        raise ValueError("weight_identity must be a non-empty string or None")
    supplied_weight_identity = weight_identity
    if isinstance(cache_shard_size, bool) or not isinstance(cache_shard_size, int) or cache_shard_size < 0:
        raise ValueError("cache_shard_size must be a non-negative integer")
    if (
        isinstance(search_query_batch_size, bool)
        or not isinstance(search_query_batch_size, int)
        or search_query_batch_size <= 0
    ):
        raise ValueError("search_query_batch_size must be a positive integer")
    bundle = (
        load_bundle(release, verify_content_hashes=verify_content_hashes)
        if release
        else load_prepared_source(suite, task, verify_content_hashes=verify_content_hashes)
    )
    if not bundle["queries"] or not bundle["documents"]:
        raise ValueError("Ranking requires documents and queries")
    kwargs = {"device": device, "batch_size": batch_size, "max_length": max_length}
    if query_prompt is not None:
        kwargs["query_prompt"] = query_prompt
    if document_prompt is not None:
        kwargs["document_prompt"] = document_prompt
    if model_path:
        kwargs["model_path"] = model_path
    if weight_identity is not None:
        kwargs["weight_identity"] = weight_identity
    if audit_model:
        kwargs.update(trust_remote_code=False, query_prompt="{text}", document_prompt="{text}")
    config = EmbeddingConfig(**kwargs)
    encoder = SentenceTransformerEncoder(config)
    docs, queries = bundle["documents"], bundle["queries"]
    cache_kwargs = {} if cache_shard_size == 0 else {"cache_shard_size": cache_shard_size}
    dv = encoder.encode_with_cache([d["text"] for d in docs], kind="document", cache_dir=cache_dir, **cache_kwargs)
    doc_metadata = copy.deepcopy(encoder.last_metadata)
    qv = encoder.encode_with_cache([q["text"] for q in queries], kind="query", cache_dir=cache_dir, **cache_kwargs)
    # Reuse the encoder's already computed weight identity when exporting;
    # otherwise export_saved_retrieval would hash multi-gigabyte weights a
    # second time after all document shards had completed.
    if hasattr(encoder, "_identities"):
        effective_weight_identity = encoder._identities()[0]
        export_config = EmbeddingConfig(
            **{**config.to_dict(), "weight_identity": effective_weight_identity}
        )
    else:
        # Keep lightweight test doubles and compatible encoder facades
        # working; the production encoder always exposes _identities.
        export_config = config
    results = dense_search(qv, dv, [d["_id"] for d in docs], query_ids=[q["_id"] for q in queries],
                           top_k=top_k, qrels=bundle["qrels"], search_device=search_device,
                           search_query_batch_size=search_query_batch_size)
    if suite is not None:
        payload = export_saved_retrieval(bundle, results, model=export_config)
    else:
        payload = {"schema_version": "dci-mining-rankings-v1", "bundle_hash": digest(bundle),
                   "model": config.to_dict(), "results": results}
    payload["encoding_audit"] = {"documents": doc_metadata, "queries": encoder.last_metadata}
    effective_weight_identity = export_config.weight_identity
    if effective_weight_identity is None:
        effective_weight_identity = payload.get("model", {}).get("weight_identity")
    verification_policy = (
        "not-applicable" if suite is None
        else "content-hashes-verified" if verify_content_hashes else "declared-source-digests"
    )
    verification_audit = {
        "verification_policy": verification_policy,
        "content_hashes_verified": verify_content_hashes if suite is not None else None,
        "content_hash_verification": (
            "not_applicable" if suite is None
            else "verified" if verify_content_hashes else "declared_saved_digest"
        ),
        "weight_identity": effective_weight_identity,
        "weight_identity_source": "supplied" if supplied_weight_identity is not None else "computed_from_local_weights",
        "weight_identity_supplied": supplied_weight_identity is not None,
        "weight_identity_verified": supplied_weight_identity is None,
        "weight_bytes_verified": supplied_weight_identity is None,
    }
    payload["verification_audit"] = verification_audit
    payload["encoding_audit"]["verification"] = copy.deepcopy(verification_audit)
    effective_batches = {
        int(row.get("search_query_batch_size_effective", search_query_batch_size))
        for row in results.values()
    }
    payload["search_audit"] = {
        "search_device": search_device,
        "requested_query_batch_size": search_query_batch_size,
        "effective_query_batch_sizes": sorted(effective_batches),
        "tf32": False if search_device.startswith("cuda") else None,
    }
    write_json(output, payload)
    return {"output": str(output), "queries": len(queries), "documents": len(docs),
            "top_k": top_k, "schema_version": payload["schema_version"]}


def partition_languages(release: Path, output: Path) -> dict:
    """Explicit monolingual views retaining every document in each source language."""
    from .core import slug
    output = _new_output(output)
    bundle = load_bundle(release)
    manifest = verify_release(release)
    languages = sorted({d["language"] for d in bundle["documents"]})
    if not languages:
        raise ValueError("Language partition requires documents")
    unknown_query_languages = {q["language"] for q in bundle["queries"]} - set(languages)
    if unknown_query_languages:
        raise ValueError("Query language has no candidate corpus")
    parts = {}
    for language in languages:
        slug(language)
        part = copy.deepcopy(bundle)
        part["documents"] = [d for d in part["documents"] if d["language"] == language]
        part["queries"] = [q for q in part["queries"] if q["language"] == language]
        dids = {d["_id"] for d in part["documents"]}
        qids = {q["_id"] for q in part["queries"]}
        part["qrels"] = {q: labels for q, labels in part["qrels"].items() if q in qids}
        if any(set(gold) - dids for gold in part["qrels"].values()):
            raise ValueError("Cannot partition a cross-language positive; preserve the multilingual task")
        for key in ("evidence", "judgments", "review_queue"):
            if key in part:
                selected = [row for row in part[key] if row.get("query_id") in qids]
                if any(row.get("doc_id") is not None and row["doc_id"] not in dids for row in selected):
                    raise ValueError("Cannot partition cross-language evidence/judgment; preserve the multilingual task")
                part[key] = selected
        part["provenance"].update(language=language, language_partition=True)
        parts[language] = part
    stage = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    stage.mkdir()
    results = {}
    try:
        for language, part in parts.items():
            results[language] = publish_bundle(part, stage / language, specification=manifest["specification"],
                                               parent_release=release,
                                               raw_snapshot=Path(manifest["raw_snapshot"]) if manifest.get("raw_snapshot") else None)
        write_json(stage / "_PARTITIONS.json", {"parent_release": str(release),
                                               "parent_bundle_hash": manifest["bundle_hash"],
                                               "languages": list(parts)})
        stage.rename(output)
    except BaseException as exc:
        write_json(stage / "_FAILED.json", {"stage": "partition-languages", "exception": type(exc).__name__})
        raise
    return {k: {f: v[f] for f in ("state", "num_documents", "num_queries")} for k, v in results.items()}
