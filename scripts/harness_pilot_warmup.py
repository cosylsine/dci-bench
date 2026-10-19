#!/usr/bin/env python
"""Warm a local OpenAI-compatible serving endpoint before the pilot starts.

The requests in this script are synthetic protocol probes.  They never read a
DCI query, corpus, or qrels file and they never execute a returned tool call.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA_VERSION = "dci-harness-pilot-warmup-v2"
DEFAULT_ROUNDS = 2
OUTPUT_LIMIT = 1024
RESPONSE_CAPTURE_MAX_BYTES = 16384
SHAPES = ("auto_tool", "named_read", "tools_disabled_final")
_HTTP_OPENER = build_opener(ProxyHandler({}))


class WarmupError(RuntimeError):
    """A bounded warmup failure with a stable audit category."""

    def __init__(self, kind: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.kind = kind
        self.details = details or {}
        self.batches: list[dict[str, Any]] = []
        self.completed_requests = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _default_model() -> str:
    return os.environ.get("SERVED_MODEL_NAME") or Path(
        os.environ.get("MODEL_PATH", "warmup-model")
    ).name


def _default_base_url() -> str:
    return os.environ.get(
        "BACKEND_BASE_URL",
        f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '28099')}/v1",
    )


def _chat_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a query or fragment")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions"


def _contract_tool(name: str) -> dict[str, Any]:
    """Load a current repository tool schema without duplicating protocol fields."""

    try:
        from dci_bench.protocol.contracts import TOOL_DEFINITIONS
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load the protocol tool contract: {type(exc).__name__}") from exc
    for definition in TOOL_DEFINITIONS:
        if isinstance(definition, dict) and definition.get("name") == name:
            schema = definition.get("input_schema")
            if not isinstance(schema, dict):
                break
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(definition.get("description", "Synthetic warmup tool.")),
                    "parameters": deepcopy(schema),
                },
            }
    raise RuntimeError(f"protocol tool contract does not define {name!r}")


def _payload(model: str, shape: str) -> dict[str, Any]:
    common = {
        "model": model,
        "temperature": 0,
        "max_tokens": OUTPUT_LIMIT,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    if shape == "auto_tool":
        return {
            **common,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "This is a synthetic warmup probe. Emit the requested synthetic tool "
                        "call. The harness records it without executing it."
                    ),
                },
                {
                    "role": "user",
                    "content": "Call bash exactly once with command='printf warmup'. Do not call read.",
                },
            ],
            "tools": [_contract_tool("read"), _contract_tool("bash")],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
    if shape == "named_read":
        return {
            **common,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "This is a synthetic warmup probe. Emit the requested synthetic tool "
                        "call. The harness records it without executing it."
                    ),
                },
                {
                    "role": "user",
                    "content": "Call read exactly once with path='/workspace/corpus/__warmup__.md'.",
                },
            ],
            "tools": [
                _contract_tool("read")
            ],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "parallel_tool_calls": False,
        }
    if shape == "tools_disabled_final":
        return {
            **common,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "This is a synthetic warmup probe. Do not access a filesystem, "
                        "network, corpus, or external tool."
                    ),
                },
                {
                    "role": "user",
                    "content": "Return a short synthetic final response without calling a tool.",
                },
            ],
            "tool_choice": "none",
        }
    raise ValueError(f"unknown warmup shape: {shape}")


def _int_usage(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WarmupError("usage_invalid", f"usage.{field} is missing or invalid")
    return value


def _validate_tool_arguments(arguments: Any, tool_name: str) -> dict[str, Any]:
    """Validate returned arguments against the current protocol JSON schema."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - project dependency is present in the runner
        raise WarmupError("response_structure", "jsonschema is unavailable") from exc
    if not isinstance(arguments, dict):
        raise WarmupError("response_structure", "tool arguments are not an object")
    schema = _contract_tool(tool_name)["function"]["parameters"]
    try:
        errors = sorted(
            Draft202012Validator(schema).iter_errors(arguments),
            key=lambda error: tuple(str(part) for part in error.path),
        )
    except Exception as exc:  # pragma: no cover - invalid repository contract
        raise WarmupError("response_structure", "tool schema could not be applied") from exc
    if errors:
        raise WarmupError("response_structure", f"{tool_name} arguments failed the protocol schema")
    return arguments


def _validate_response(payload: Any, shape: str) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise WarmupError("response_structure", "response is not a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise WarmupError("response_structure", "response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise WarmupError("response_structure", "choice.message is missing or invalid")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise WarmupError("usage_missing", "response usage is missing")
    usage_record = {
        field: _int_usage(usage.get(field), field)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if usage_record["total_tokens"] < (
        usage_record["prompt_tokens"] + usage_record["completion_tokens"]
    ):
        raise WarmupError("usage_invalid", "usage.total_tokens is below prompt plus completion tokens")

    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise WarmupError("response_structure", "message.tool_calls is invalid")
    if shape in {"auto_tool", "named_read"}:
        expected_name = "bash" if shape == "auto_tool" else "read"
        # vLLM 0.17 preserves "stop" for a named tool choice, while auto
        # tools return "tool_calls". Neither permits a truncated completion.
        allowed_finishes = ("stop", "tool_calls") if shape == "named_read" else ("tool_calls",)
        finish = choice.get("finish_reason")
        if finish not in allowed_finishes:
            raise WarmupError(
                "unexpected_finish_reason",
                f"{shape}: expected finish_reason in {allowed_finishes}, got {str(finish)[:128]!r}",
            )
        if len(calls) != 1 or not isinstance(calls[0], dict):
            raise WarmupError("unexpected_tool", f"expected exactly one {expected_name} tool call")
        function = calls[0].get("function")
        if not isinstance(function, dict) or function.get("name") != expected_name:
            raise WarmupError("unexpected_tool", f"expected {expected_name} tool call")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise WarmupError("response_structure", "tool arguments are not JSON") from exc
        arguments = _validate_tool_arguments(arguments, expected_name)
        expected_arguments = (
            {"command": "printf warmup"}
            if shape == "auto_tool"
            else {"path": "/workspace/corpus/__warmup__.md"}
        )
        for field, expected in expected_arguments.items():
            if arguments.get(field) != expected:
                raise WarmupError("unexpected_tool", f"{expected_name} argument {field} is not synthetic")
    else:
        if choice.get("finish_reason") != "stop":
            raise WarmupError("response_structure", "tools-disabled response did not stop")
        if calls:
            raise WarmupError("unexpected_tool", "tools-disabled response contains a tool call")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise WarmupError("response_structure", "tools-disabled response has no final content")
    return usage_record


def _response_details(raw: bytes, *, http_status: int) -> dict[str, Any]:
    """Capture bounded synthetic response evidence, never request/auth headers."""
    encoded = raw.decode("utf-8", errors="replace").encode("utf-8")
    details: dict[str, Any] = {
        "http_status": http_status,
        "response_bytes_read": len(raw),
        "response_excerpt": encoded[:RESPONSE_CAPTURE_MAX_BYTES].decode("utf-8", errors="ignore"),
        "response_truncated": len(encoded) > RESPONSE_CAPTURE_MAX_BYTES,
    }
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return details
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            details["finish_reason"] = str(choice.get("finish_reason"))[:128]
            message = choice.get("message")
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if isinstance(calls, list):
                details["tool_names"] = [
                    str(call["function"].get("name"))[:128]
                    for call in calls[:10]
                    if isinstance(call, dict) and isinstance(call.get("function"), dict)
                ]
    return details


def _post(url: str, body: dict[str, Any], timeout: float, api_key: str | None) -> tuple[Any, bytes]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with _HTTP_OPENER.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise WarmupError("http_error", f"chat completion returned HTTP {response.status}")
            raw = response.read(8 * 1024 * 1024 + 1)
    except HTTPError as exc:
        details: dict[str, Any] = {"http_status": exc.code}
        try:
            with exc:
                details.update(_response_details(exc.read(RESPONSE_CAPTURE_MAX_BYTES + 1), http_status=exc.code))
        except OSError as read_exc:
            details["response_read_error"] = type(read_exc).__name__
        raise WarmupError("http_error", f"chat completion returned HTTP {exc.code}", details=details) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise WarmupError("timeout", "chat completion request timed out") from exc
    except (URLError, OSError) as exc:
        raise WarmupError("http_error", f"chat completion request failed: {type(exc).__name__}") from exc
    if len(raw) > 8 * 1024 * 1024:
        raise WarmupError("response_structure", "chat completion response is too large",
                          details=_response_details(raw, http_status=200))
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WarmupError("response_structure", "chat completion response is not JSON",
                          details=_response_details(raw, http_status=200)) from exc
    return payload, raw


def _request_one(
    url: str,
    body: dict[str, Any],
    shape: str,
    deadline: float,
    http_timeout: float,
    api_key: str | None,
) -> tuple[dict[str, int], float]:
    started = time.monotonic()
    remaining = deadline - started
    if remaining <= 0:
        raise WarmupError("deadline", "warmup deadline expired before request")
    response, raw = _post(url, body, min(http_timeout, remaining), api_key)
    try:
        usage = _validate_response(response, shape)
    except WarmupError as exc:
        exc.details.update(_response_details(raw, http_status=200))
        raise
    return usage, time.monotonic() - started


def _run_batch(
    *,
    url: str,
    model: str,
    shape: str,
    round_number: int,
    concurrency: int,
    deadline: float,
    http_timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    started = time.monotonic()
    bodies = [_payload(model, shape) for _ in range(concurrency)]
    results: list[tuple[dict[str, int], float] | None] = [None] * concurrency
    errors: list[WarmupError] = []
    stop = threading.Event()

    def worker(index: int) -> None:
        if stop.is_set():
            return
        try:
            results[index] = _request_one(
                url, bodies[index], shape, deadline, http_timeout, api_key
            )
        except WarmupError as exc:
            exc.details.update({"round": round_number, "shape": shape, "request_index": index})
            errors.append(exc)
            stop.set()
        except Exception as exc:  # pragma: no cover - defensive boundary for worker threads
            errors.append(WarmupError("request_failed", type(exc).__name__))
            stop.set()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop.set()
            errors.append(WarmupError("deadline", "warmup deadline expired during request batch"))
            break
        thread.join(timeout=remaining)
    if any(thread.is_alive() for thread in threads):
        stop.set()
        errors.append(WarmupError("deadline", "warmup request batch exceeded deadline"))
    if errors:
        errors[0].details.update({"round": round_number, "shape": shape})
        raise errors[0]
    if any(item is None for item in results):
        raise WarmupError("request_failed", "warmup request batch did not complete")
    usage = {
        field: sum(item[0][field] for item in results if item is not None)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "round": round_number,
        "shape": shape,
        "requests": concurrency,
        "successes": concurrency,
        "duration_seconds": round(time.monotonic() - started, 6),
        "usage": usage,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_warmup(
    *,
    base_url: str,
    model: str,
    concurrency: int,
    deadline_seconds: float,
    http_timeout_seconds: float,
    api_key: str | None = None,
    rounds: int = DEFAULT_ROUNDS,
) -> dict[str, Any]:
    if concurrency < 1 or rounds < 1:
        raise ValueError("concurrency and rounds must be positive")
    if (
        not math.isfinite(deadline_seconds)
        or not math.isfinite(http_timeout_seconds)
        or deadline_seconds <= 0
        or http_timeout_seconds <= 0
    ):
        raise ValueError("deadline and HTTP timeout must be positive")
    url = _chat_url(base_url)
    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()
    deadline = started + deadline_seconds
    batches: list[dict[str, Any]] = []
    aggregate = {field: 0 for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    completed_requests = 0
    for round_number in range(1, rounds + 1):
        for shape in SHAPES:
            try:
                batch = _run_batch(
                    url=url,
                    model=model,
                    shape=shape,
                    round_number=round_number,
                    concurrency=concurrency,
                    deadline=deadline,
                    http_timeout=http_timeout_seconds,
                    api_key=api_key,
                )
            except WarmupError as exc:
                exc.batches = list(batches)
                exc.completed_requests = completed_requests
                exc.usage = dict(aggregate)
                raise
            batches.append(batch)
            completed_requests += batch["successes"]
            for field in aggregate:
                aggregate[field] += batch["usage"][field]
    finished_wall = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "stage": "warmup",
        "phase": "pre_evaluation",
        "base_url": base_url.rstrip("/"),
        "model": model,
        "rounds": rounds,
        "concurrency": concurrency,
        "requested_requests": rounds * len(SHAPES) * concurrency,
        "completed_requests": completed_requests,
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "deadline_seconds": deadline_seconds,
        "http_timeout_seconds": http_timeout_seconds,
        "shapes": list(SHAPES),
        "batches": batches,
        "usage": aggregate,
        "success": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=_default_base_url())
    parser.add_argument("--model", default=_default_model())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=_positive_int(os.environ.get("MAX_CONCURRENCY", "1")),
    )
    parser.add_argument(
        "--deadline-seconds",
        type=_positive_float,
        default=_positive_float(os.environ.get("WARMUP_DEADLINE_SECONDS", "600")),
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=_positive_float,
        default=_positive_float(os.environ.get("WARMUP_HTTP_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--rounds", type=_positive_int, default=DEFAULT_ROUNDS)
    parser.add_argument("--dry-run", action="store_true", help="validate configuration without HTTP requests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "synthetic": True,
                    "stage": "warmup",
                    "phase": "pre_evaluation",
                    "base_url": args.base_url.rstrip("/"),
                    "model": args.model,
                    "rounds": args.rounds,
                    "concurrency": args.concurrency,
                    "shapes": list(SHAPES),
                    "success": None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()
    try:
        result = run_warmup(
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            deadline_seconds=args.deadline_seconds,
            http_timeout_seconds=args.http_timeout_seconds,
            api_key=args.api_key,
            rounds=args.rounds,
        )
    except Exception as exc:
        kind = exc.kind if isinstance(exc, WarmupError) else "configuration_or_request"
        result = {
            "schema_version": SCHEMA_VERSION,
            "synthetic": True,
            "stage": "warmup",
            "phase": "pre_evaluation",
            "base_url": args.base_url.rstrip("/"),
            "model": args.model,
            "rounds": args.rounds,
            "concurrency": args.concurrency,
            "requested_requests": args.rounds * len(SHAPES) * args.concurrency,
            "completed_requests": getattr(exc, "completed_requests", 0),
            "started_at": started_wall.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 6),
            "deadline_seconds": args.deadline_seconds,
            "http_timeout_seconds": args.http_timeout_seconds,
            "shapes": list(SHAPES),
            "batches": getattr(exc, "batches", []),
            "usage": getattr(exc, "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "success": False,
            "accounting_scope": "successful_batches_only",
            "error": {"kind": kind, "message": str(exc), "details": getattr(exc, "details", {})},
        }
        try:
            _write_json(args.output, result)
        except OSError as write_exc:
            print(f"[warmup] failed to write failure artifact: {write_exc}", file=sys.stderr)
        print(f"[warmup] failed ({kind}): {exc}", file=sys.stderr)
        # Cluster users may only receive stdout/stderr; include a compact
        # response preview there as well as the larger persisted excerpt.
        diagnostic = dict(result["error"]["details"])
        if "response_excerpt" in diagnostic:
            excerpt = diagnostic["response_excerpt"].encode("utf-8")
            diagnostic["response_excerpt"] = excerpt[:2048].decode("utf-8", errors="ignore")
            diagnostic["response_truncated"] = diagnostic["response_truncated"] or len(excerpt) > 2048
        print(f"[warmup] failure details: {json.dumps(diagnostic, ensure_ascii=False)}", file=sys.stderr)
        print(f"[warmup] failure artifact: {args.output}", file=sys.stderr)
        return 2
    _write_json(args.output, result)
    print(f"[warmup] complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
