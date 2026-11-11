"""Reference hardware metadata, separate from benchmark quality controls."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess


def host_hardware() -> dict:
    """Observe the runner host; its GPUs need not serve the model endpoint."""
    cpuinfo = Path('/proc/cpuinfo')
    models = sorted({line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                     if line.startswith('model name')}) if cpuinfo.exists() else []
    meminfo = Path('/proc/meminfo')
    memory = dict(line.split(':', 1) for line in meminfo.read_text().splitlines()) if meminfo.exists() else {}
    gpus = None
    gpu_status = 'nvidia-smi unavailable'
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            gpu_status = 'nvidia-smi probe failed or timed out'
        else:
            gpu_status = 'available' if result.returncode == 0 else 'nvidia-smi probe failed'
            if result.returncode == 0:
                gpus = [{'index': row[0].strip(), 'name': row[1].strip(),
                         'memory_total_mib': row[2].strip(), 'driver_version': row[3].strip()}
                        for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 4]
    return {
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'runner_host_inventory; not proof of model device allocation',
        'os': platform.system(), 'architecture': platform.machine(),
        'cpu_models': models, 'logical_cpu_count': os.cpu_count(),
        'available_cpu_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'memory_total_bytes': int(memory['MemTotal'].split()[0]) * 1024 if 'MemTotal' in memory else None,
        'gpus': gpus, 'gpu_count': len(gpus) if gpus is not None else None,
        'gpu_status': gpu_status, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }


def execution_hardware(backend: dict, concurrency: int) -> dict:
    """Keep backend-declared serving hardware distinct from local observations."""
    runtime = backend.get('runtime') or {}
    cuda = runtime.get('cuda') or {}
    gpus = cuda.get('gpus')
    return {
        'runner_host': host_hardware(),
        'model_server': {
            'source': 'backend_manifest.runtime' if runtime else 'unknown',
            'captured_at': runtime.get('captured_at'),
            'host': runtime.get('host_hardware'),
            'gpus': gpus, 'gpu_count': len(gpus) if gpus is not None else None,
            'cuda_visible_devices': cuda.get('visible_devices'),
            'torch_cuda': runtime.get('torch_cuda'), 'packages': runtime.get('packages'),
            'unknown_reason': None if gpus is not None else 'serving hardware was not supplied by the backend',
        },
        'execution': {'concurrency': concurrency, **{key: backend.get(key) for key in (
            'kind', 'tensor_parallel_size', 'dtype', 'context_length', 'max_num_seqs')}},
    }
