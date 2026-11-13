"""Provider extensions consumed by runners through one model-call boundary.

Register reviewed implementations here. Manifest values select an implementation;
they never import arbitrary modules or override the evaluation contract.
"""

from collections.abc import Mapping
from typing import Any, Protocol
from urllib.request import Request

from .openai_chat import OpenAIChatAdapter


class BackendAdapter(Protocol):
    def validate(self, backend: Mapping[str, Any]) -> None: ...

    def thinking_body(self, backend: Mapping[str, Any], generation: Mapping[str, Any]) -> dict: ...

    def generation_kwargs(self, generation: Mapping[str, Any], thinking: dict) -> dict: ...

    def inspect_kwargs(self, backend: Mapping[str, Any], model: Mapping[str, Any], *,
                       api_key: str, disable_retries: bool) -> dict: ...

    def warmup_request(self, base_url: str, body: dict, api_key: str | None) -> Request: ...

    def normalize_warmup_response(self, payload: Any) -> Any: ...


ADAPTERS: dict[str, BackendAdapter] = {"openai-chat-completions": OpenAIChatAdapter()}


def get_backend_adapter(backend: Mapping[str, Any] | None = None) -> BackendAdapter:
    name = (backend or {}).get("adapter", "openai-chat-completions")
    if not isinstance(name, str) or name not in ADAPTERS:
        raise ValueError(f"unsupported backend adapter: {name!r}")
    return ADAPTERS[name]


def inspect_model_kwargs(backend: Mapping[str, Any], model: Mapping[str, Any], *,
                         api_key: str, disable_retries: bool = False) -> dict[str, Any]:
    return get_backend_adapter(backend).inspect_kwargs(
        backend, model, api_key=api_key, disable_retries=disable_retries,
    )
