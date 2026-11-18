"""Count Pi chat requests after the Inspect OpenAI-compatible conversion.

JSONL subprocess protocol; production uses the running vLLM CPU /tokenize route.
Local tokenizer mode is for offline regression tests, never an implicit fallback.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from inspect_ai.agent._bridge.completions import tools_from_openai_tools
from inspect_ai.model._openai import messages_to_openai, openai_chat_tools
from inspect_ai.model._openai_convert import messages_from_openai
from inspect_ai.util._json import JSON_SCHEMA_EXTENDED_FIELDS


async def normalize_request(request: dict, model: str) -> dict:
    messages = await messages_from_openai(request["messages"], model)
    tools = openai_chat_tools(
        tools_from_openai_tools(request.get("tools", [])),
        exclude=JSON_SCHEMA_EXTENDED_FIELDS,
    )
    for tool in tools:
        tool["function"]["strict"] = True
    return {
        "model": model,
        "messages": await messages_to_openai(messages),
        **({"tools": tools} if tools else {}),
        "chat_template_kwargs": request["chat_template_kwargs"],
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url")
    source.add_argument("--local-tokenizer")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = None
    if args.local_tokenizer:
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer, local_files_only=True)
        # Transformers 5 reconstructs Qwen's pre-tokenizer differently from
        # the pinned serving version. Preserve the frozen tokenizer.json here.
        encoding = Tokenizer.from_file(str(Path(args.local_tokenizer) / "tokenizer.json"))
    with asyncio.Runner() as runner:
        for line in sys.stdin:
            request = runner.run(normalize_request(json.loads(line), args.model))
            if tokenizer is not None:
                # vLLM chat_utils._postprocess_messages parses OpenAI's JSON
                # argument strings before rendering the HF tool-use template.
                for message in request["messages"]:
                    for call in message.get("tool_calls", []):
                        call["function"]["arguments"] = json.loads(call["function"]["arguments"] or "{}")
                rendered = tokenizer.apply_chat_template(
                    request["messages"], tools=request.get("tools"), tokenize=False,
                    add_generation_prompt=True, **request["chat_template_kwargs"],
                )
                count = len(encoding.encode(rendered, add_special_tokens=False).ids)
            else:
                endpoint = args.base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
                payload = json.dumps(request).encode()
                with urlopen(Request(endpoint, data=payload, headers={"Content-Type": "application/json"}), timeout=30) as response:
                    count = json.load(response)["count"]
            if type(count) is not int or count < 0:
                raise ValueError("tokenizer returned an invalid count")
            print(json.dumps({"count": count}), flush=True)


if __name__ == "__main__":
    main()
