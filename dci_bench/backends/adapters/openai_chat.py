"""The existing Inspect OpenAI Chat Completions transport and parameter mapping."""

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request


class OpenAIChatAdapter:
    def validate(self, backend: Mapping[str, Any]) -> None:
        if backend.get("kind") == "remote_api" and backend["version"] != "openai-compatible-chat-v1":
            raise ValueError("remote backend must use the supported Chat Completions transport")

    def thinking_body(self, backend: Mapping[str, Any], generation: Mapping[str, Any]) -> dict:
        if backend.get("kind") != "remote_api" and "thinking_policy" not in backend:
            return {"chat_template_kwargs": {"enable_thinking": generation["enable_thinking"]}}
        policy = backend.get("thinking_policy")
        if not isinstance(policy, dict) or set(policy) - {"mode", "evidence", "extra_body"}:
            raise ValueError("backend requires an explicit thinking_policy")
        if not isinstance(policy.get("evidence"), str) or not policy["evidence"].strip():
            raise ValueError("thinking_policy requires model-specific evidence")
        body = policy.get("extra_body", {})
        if not isinstance(body, dict):
            raise ValueError("thinking_policy.extra_body must be an object")
        if policy.get("mode") == "not_supported":
            if body:
                raise ValueError("non-reasoning models must not send thinking parameters")
        elif policy.get("mode") == "disabled":
            if not body:
                raise ValueError("disabled thinking requires the provider's extra_body parameters")
        else:
            raise ValueError("thinking_policy.mode must be not_supported or disabled")
        reserved = {"model", "messages", "tools", "tool_choice", "stream", "temperature", "seed",
                    "max_tokens", "max_completion_tokens", "n", "top_p", "stop", "parallel_tool_calls"}
        if reserved.intersection(body):
            raise ValueError("thinking_policy cannot override generation or tool controls")
        json.dumps(body, allow_nan=False)
        return deepcopy(body)

    def generation_kwargs(self, generation: Mapping[str, Any], thinking: dict) -> dict:
        kwargs = {"temperature": generation["temperature"], "max_tokens": generation["max_tokens"],
                  "seed": generation["seed"], "parallel_tool_calls": False}
        if thinking:
            kwargs["extra_body"] = deepcopy(thinking)
        return kwargs

    def inspect_kwargs(self, backend: Mapping[str, Any], model: Mapping[str, Any], *,
                       api_key: str, disable_retries: bool) -> dict:
        model_args = {"api_key": api_key, "responses_api": False}
        if disable_retries:
            model_args["max_retries"] = 0
        return {
            "model": model.get("inspect_model") or
                     f"openai-api/{backend['openai_service']}/{model['served_model_name']}",
            "model_base_url": backend.get("base_url"),
            "model_args": model_args,
        }

    def warmup_request(self, base_url: str, body: dict, api_key: str | None) -> Request:
        value = base_url.rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base URL must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base URL must not contain a query or fragment")
        url = value if value.endswith("/chat/completions") else f"{value}/chat/completions"
        return Request(url, data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                       headers={"Content-Type": "application/json",
                                **({"Authorization": f"Bearer {api_key}"} if api_key else {})}, method="POST")

    def normalize_warmup_response(self, payload: Any) -> Any:
        # The shared probe consumes this canonical chat-shaped representation.
        # Other transports convert their response here before semantic checks.
        return payload
