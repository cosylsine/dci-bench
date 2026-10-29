#!/usr/bin/env python3
"""Print a local vLLM deployment command; --execute explicitly starts the server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.core import read_json, file_digest


def deployment(config: dict, role: str, gpus: str, python: str, *, model_path: Path | None = None) -> dict:
    if not re.fullmatch(r"\d+(,\d+)*", gpus) or len(set(gpus.split(","))) != len(gpus.split(",")):
        raise ValueError("--gpus must contain distinct numeric GPU IDs")
    model = config["roles"][role]
    path = model_path or Path(model["path"])
    tp = model["tensor_parallel_size"]
    if len(gpus.split(",")) != tp:
        raise ValueError(f"This profile requires {tp} GPU IDs")
    endpoint = urlsplit(model["base_url"])
    if endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.port is None:
        raise ValueError("Deployment must bind to an explicit local endpoint")
    command = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(path),
               "--served-model-name", model["served_model_name"], "--host", "127.0.0.1", "--port", str(endpoint.port),
               "--tensor-parallel-size", str(tp), "--dtype", model["dtype"],
               "--max-model-len", str(model["max_model_len"]), "--max-num-seqs", str(model["max_num_seqs"]),
               "--gpu-memory-utilization", str(model["gpu_memory_utilization"])]
    if role == "generator":
        command += ["--reasoning-parser", "qwen3", "--language-model-only"]
    else:
        for key in ("tokenizer_mode", "config_format", "load_format"):
            if key in model:
                command.extend(["--" + key.replace("_", "-"), model[key]])
    identity_files = {}
    if path.is_dir():
        for filename in ("config.json", "params.json", "tokenizer_config.json", "model.safetensors.index.json"):
            file = path / filename
            if file.is_file():
                identity_files[filename] = file_digest(file)
    return {"role": role, "model_id": model["model_id"], "local_path_exists": path.is_dir(),
            "config_fingerprints": identity_files, "weight_bytes_verified": False,
            "env": {"CUDA_VISIBLE_DEVICES": gpus, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            "command": command, "copyable_command": f"CUDA_VISIBLE_DEVICES={gpus} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 " + shlex.join(command),
            "endpoint": model["base_url"], "boundary": "deployment plan, not a GPU capacity or throughput validation"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/data_pipeline/models.json")
    p.add_argument("--role", choices=("generator", "judge"), required=True)
    p.add_argument("--gpus", required=True)
    p.add_argument("--python", default="/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38/bin/python")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    plan = deployment(read_json(args.config), args.role, args.gpus, args.python, model_path=args.model_path)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.execute:
        if not plan["local_path_exists"] or not plan["config_fingerprints"]:
            raise ValueError("Local model/config not available; no automatic model download")
        endpoint = urlsplit(plan["endpoint"])
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", endpoint.port))
        env = os.environ.copy()
        env.update(plan["env"])
        raise SystemExit(subprocess.call(plan["command"], env=env))


if __name__ == "__main__":
    main()
