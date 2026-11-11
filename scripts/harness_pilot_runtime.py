#!/usr/bin/env python
"""Capture the actual serving interpreter and test a CUDA kernel per device."""

import argparse
import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dci_bench.results.hardware import host_hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the serving interpreter")
    required_devices = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    if torch.cuda.device_count() < required_devices:
        raise RuntimeError(f"TP={required_devices} requires that many visible GPUs; found {torch.cuda.device_count()}")
    devices = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            value = torch.ones((32, 32), device=f"cuda:{index}")
            assert (value @ value).sum().item() == 32768
        prop = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": prop.name, "total_memory_bytes": prop.total_memory,
                        "capability": list(torch.cuda.get_device_capability(index))})
    packages = {}
    for name in ("vllm", "torch", "transformers", "compressed-tensors", "flashinfer-python"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    payload = {"schema_version": "dci-serving-runtime-v1", "captured_at": datetime.now(timezone.utc).isoformat(),
               "host_hardware": host_hardware(),
               "python": {"executable": sys.executable, "version": platform.python_version()},
               "packages": packages, "torch_cuda": torch.version.cuda,
               "launch_environment": {key: os.environ.get(key) for key in (
                   "MODEL_PATH", "SERVED_MODEL_NAME", "VLLM_EXTRA_ARGS", "VLLM_MAX_NUM_SEQS",
                   "VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_NUM_BATCHED_TOKENS", "VLLM_WORKER_CLS",
                   "VLLM_NVFP4_GEMM_BACKEND", "KV_CACHE_DTYPE")},
               "cuda": {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": devices,
                        "kernel_probe": "passed-on-every-visible-device"}}
    if os.environ.get("VLLM_WORKER_CLS") == "dci_bench.backends.vllm_gdn_worker.GDNWarmupWorker":
        from dci_bench.backends import vllm_gdn_worker
        import vllm.model_executor.models.qwen3_next as qwen3_next
        vllm_gdn_worker.install_profile_warmup()
        payload["serving_patch"] = {
            "repair_id": vllm_gdn_worker.REPAIR_ID,
            "upstream_commit": vllm_gdn_worker.UPSTREAM_COMMIT,
            "upstream_url": "https://github.com/vllm-project/vllm/pull/36599",
            "source_sha256": {str(Path(module.__file__).resolve()): hashlib.sha256(
                Path(module.__file__).read_bytes()).hexdigest()
                for module in (vllm_gdn_worker, qwen3_next)},
        }
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
