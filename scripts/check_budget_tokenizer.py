#!/usr/bin/env python3
"""Small serving probe: normalized Inspect payloads must agree with vLLM usage."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.agents.token_count import normalize_request
from dci_bench.protocol.contracts import TOOL_DEFINITIONS


async def probe(base_url: str, model: str, output: Path) -> dict:
    tools = [{"type": "function", "function": {"name": tool["name"],
              "description": tool["description"], "parameters": tool["input_schema"]}}
             for tool in TOOL_DEFINITIONS]
    initial = [{"role": "system", "content": "Synthetic token-count probe; do not execute tools."},
               {"role": "user", "content": "Reply with one short JSON object."}]
    history = initial + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "read-probe", "type": "function",
         "function": {"name": "read", "arguments": '{"path":"d000001.md"}'}}]},
        {"role": "tool", "tool_call_id": "read-probe", "content": '{"doc_id":"d000001"}\n中文 evidence 😀'},
    ]
    report = {"schema_version": "dci-tokenizer-probe-v1", "model": model, "synthetic": True,
              "success": False, "checks": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, messages, offered in [("search", initial, tools), ("read", history, tools[:1]),
                                     ("final", history + [{"role": "user", "content": "Return final JSON."}], []),
                                     ("repair", history + [{"role": "assistant", "content": "invalid"},
                                      {"role": "user", "content": "Repair the JSON; no tools."}], [])]:
        raw = {"messages": deepcopy(messages), "tools": offered,
               "chat_template_kwargs": {"enable_thinking": False}}
        payload = await normalize_request(raw, model)
        tokenize_url = base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
        with urlopen(Request(tokenize_url, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}), timeout=30) as response:
            predicted = json.load(response)["count"]
        payload.pop("add_special_tokens")
        payload.pop("add_generation_prompt")
        payload.update(max_tokens=64, temperature=0, seed=0, parallel_tool_calls=False)
        if offered:
            payload["tool_choice"] = {"type": "function", "function": {"name": "read"}} if name == "read" else "auto"
        with urlopen(Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}), timeout=120) as response:
            usage = json.load(response)["usage"]
        actual = usage["prompt_tokens"]
        report["checks"].append({"shape": name, "predicted_input": predicted, "actual_input": actual,
                                "delta": actual - predicted, "usage": usage})
        output.write_text(json.dumps(report, indent=2) + "\n")
        if predicted != actual:
            raise RuntimeError(f"{name}: tokenizer {predicted} != model usage {actual}; benchmark not started")
    report["success"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(probe(args.base_url, args.model, args.output))
    print(f"[tokenizer] 4/4 request shapes matched usage: {args.output}", flush=True)
