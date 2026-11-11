"""Launch-scoped vLLM 0.17.0 GDN profiling fix (upstream PR #36599).

https://github.com/vllm-project/vllm/pull/36599
Warm the same prefill kernels before KV allocation, without changing installed
packages, attention math, precision, or the inference path. Unlike upstream's
warning-only handler, a failed warmup stops this diagnostic launch.
"""
from __future__ import annotations

from functools import wraps
from importlib.metadata import version
import logging

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.v1.worker.gpu_worker import Worker

REPAIR_ID = "vllm-0.17.0-gdn-profile-warmup-v1"
UPSTREAM_COMMIT = "3e64fe4a183aae43c039c9467fe2be49c68389fa"
WARMUP_LENGTHS = (16, 32, 64)
logger = logging.getLogger(__name__)


@torch.inference_mode()
def warmup_prefill(layer, mixed_qkv: torch.Tensor, lengths: tuple[int, ...] = WARMUP_LENGTHS) -> None:
    """Cover every chunk-output BT key using this layer's actual TP dimensions."""
    if getattr(layer, "_dci_prefill_warmed_up", False):
        return
    device, dtype = mixed_qkv.device, mixed_qkv.dtype
    key_heads = layer.num_k_heads // layer.tp_size
    value_heads = layer.num_v_heads // layer.tp_size
    _, state_dtype = layer.get_state_dtype()
    if device.type == "cuda":
        set_triton_allocator(device)
    # Keep profiling's synthetic input generation out of the model RNG stream.
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for length in lengths:
            q = torch.randn(1, length, key_heads, layer.head_k_dim, device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn(1, length, value_heads, layer.head_v_dim, device=device, dtype=dtype)
            g = torch.full((1, length, value_heads), -0.1, device=device, dtype=dtype)
            beta = torch.full_like(g, 0.5)
            state = torch.zeros(1, value_heads, layer.head_v_dim, layer.head_k_dim,
                                device=device, dtype=state_dtype)
            cu_seqlens = torch.tensor([0, length], device=device, dtype=torch.long)
            output, _ = layer.chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                output_final_state=False, cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            if not torch.isfinite(output).all().item():
                raise RuntimeError(f"GDN prefill warmup produced nonfinite output: {layer.prefix}, T={length}")
            del q, k, v, g, beta, state, cu_seqlens, output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    layer._dci_prefill_warmed_up = True
    logger.info("[dci-gdn-warmup] passed layer=%s device=%s lengths=%s before KV allocation",
                layer.prefix, device, lengths)


def install_profile_warmup() -> None:
    """Install the upstream profile hook inside each explicitly selected worker."""
    if version("vllm") != "0.17.0":
        raise RuntimeError(f"{REPAIR_ID} requires reviewed vLLM 0.17.0")
    original = Qwen3NextGatedDeltaNet._forward_core
    if getattr(original, "_dci_profile_warmup", False):
        return
    if hasattr(Qwen3NextGatedDeltaNet, "_warmup_prefill_kernels"):
        raise RuntimeError("vLLM already has an upstream GDN warmup; review and remove the local worker adapter")

    @wraps(original)
    def forward_core(self, mixed_qkv, b, a, core_attn_out):
        if get_forward_context().attn_metadata is None:
            warmup_prefill(self, mixed_qkv)
        return original(self, mixed_qkv, b, a, core_attn_out)

    forward_core._dci_profile_warmup = True
    Qwen3NextGatedDeltaNet._forward_core = forward_core
    logger.info("[dci-gdn-warmup] installed %s upstream=%s", REPAIR_ID, UPSTREAM_COMMIT)


class GDNWarmupWorker(Worker):
    def load_model(self) -> None:
        install_profile_warmup()
        super().load_model()

    def determine_available_memory(self) -> int:
        available = super().determine_available_memory()
        layers = [layer for layer in self.model_runner.model.modules()
                  if isinstance(layer, Qwen3NextGatedDeltaNet)]
        if not layers or any(not getattr(layer, "_dci_prefill_warmed_up", False) for layer in layers):
            raise RuntimeError("GDN profiling did not warm every layer; refusing KV cache allocation")
        logger.info("[dci-gdn-warmup] rank=%s all %s GDN layers passed; available_kv_bytes=%s",
                    self.rank, len(layers), available)
        return available
