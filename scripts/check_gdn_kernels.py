#!/usr/bin/env python3
"""Exercise Qwen GDN Triton kernels on every visible GPU without model weights.

This is a kernel check, not a model/long-context/concurrent-service acceptance.
The serving worker repeats warmup in its own profiling process before KV cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

from dci_bench.backends.vllm_gdn_worker import REPAIR_ID, WARMUP_LENGTHS, warmup_prefill

PROBE_LENGTHS = (*WARMUP_LENGTHS, 664, 2048)  # observed failing request and prefill chunk cap

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config_bytes = args.model_config.read_bytes()
    config = json.loads(config_bytes)["text_config"]
    if args.tensor_parallel_size < 1 or any(
        config[key] % args.tensor_parallel_size
        for key in ("linear_num_key_heads", "linear_num_value_heads")
    ):
        raise ValueError("GDN heads must divide evenly across tensor-parallel ranks")
    if not torch.cuda.is_available() or not torch.cuda.device_count():
        raise RuntimeError("GDN kernel probe requires a visible CUDA device")
    # Match Qwen3_5ForConditionalGenerationConfig's auto-cache resolution.
    state_dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.bfloat16, "auto", config.get("mamba_ssm_dtype", "auto"))
    records = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            layer = SimpleNamespace(
                prefix=f"kernel-probe-gpu-{index}", tp_size=args.tensor_parallel_size,
                num_k_heads=config["linear_num_key_heads"], num_v_heads=config["linear_num_value_heads"],
                head_k_dim=config["linear_key_head_dim"], head_v_dim=config["linear_value_head_dim"],
                get_state_dtype=lambda: state_dtypes,
                chunk_gated_delta_rule=chunk_gated_delta_rule,
            )
            warmup_prefill(layer, torch.empty(0, device=f"cuda:{index}", dtype=torch.bfloat16), PROBE_LENGTHS)
            records.append({"index": index, "name": torch.cuda.get_device_name(index),
                            "capability": list(torch.cuda.get_device_capability(index)), "status": "passed"})
            print(f"[gdn-kernel-probe] GPU {index}: native Triton lengths={PROBE_LENGTHS} passed", flush=True)
    record = {"status": "passed", "repair_id": REPAIR_ID, "native_triton": True,
              "model_weights_loaded": False, "model_inference_completed": False,
              "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "input_dtype": str(torch.bfloat16), "ssm_state_dtype": str(state_dtypes[1]),
              "tensor_parallel_size_for_head_shapes": args.tensor_parallel_size,
              "lengths": PROBE_LENGTHS, "gpus": records}
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
