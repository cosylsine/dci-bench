"""Run with the pinned dci-vllm-qwen38 interpreter; no model/GPU calls."""
from types import SimpleNamespace
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_SERVING_RUNTIME = find_spec("torch") is not None and find_spec("vllm") is not None
if HAS_SERVING_RUNTIME:
    import torch
    from dci_bench.backends import vllm_gdn_worker as repair


@unittest.skipUnless(HAS_SERVING_RUNTIME, "requires the pinned dci-vllm-qwen38 interpreter")
class GDNWarmupTests(unittest.TestCase):
    def layer(self, kernel):
        return SimpleNamespace(prefix="fixture", num_k_heads=16, num_v_heads=48,
                               tp_size=4, head_k_dim=128, head_v_dim=128,
                               get_state_dtype=lambda: (torch.bfloat16, torch.float32),
                               chunk_gated_delta_rule=kernel)

    def test_warmup_covers_all_bt_keys_actual_tp_shape_and_preserves_rng(self):
        calls = []

        def kernel(**kwargs):
            calls.append(kwargs)
            return torch.zeros_like(kwargs["v"]), None

        layer = self.layer(kernel)
        before = torch.get_rng_state().clone()
        sample = torch.empty(0, dtype=torch.bfloat16)
        repair.warmup_prefill(layer, sample)
        repair.warmup_prefill(layer, sample)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(layer._dci_prefill_warmed_up)
        self.assertEqual([row["q"].shape[1] for row in calls], [16, 32, 64])
        for length, row in zip((16, 32, 64), calls):
            self.assertEqual(tuple(row["q"].shape), (1, length, 4, 128))
            self.assertEqual(tuple(row["v"].shape), (1, length, 12, 128))
            self.assertEqual(tuple(row["initial_state"].shape), (1, 12, 128, 128))
            self.assertEqual(row["initial_state"].dtype, torch.float32)
            self.assertEqual(row["cu_seqlens"].tolist(), [0, length])
            self.assertFalse(row["output_final_state"])
            self.assertTrue(row["use_qk_l2norm_in_kernel"])

    def test_kernel_failure_is_fatal_and_does_not_mark_layer_warmed(self):
        calls = []

        def kernel(**kwargs):
            calls.append(kwargs["q"].shape[1])
            if len(calls) == 2:
                raise RuntimeError("synthetic autotuner OOM")
            return torch.zeros_like(kwargs["v"]), None

        layer = self.layer(kernel)
        with self.assertRaisesRegex(RuntimeError, "autotuner OOM"):
            repair.warmup_prefill(layer, torch.empty(0, dtype=torch.bfloat16))
        self.assertEqual(calls, [16, 32])
        self.assertFalse(getattr(layer, "_dci_prefill_warmed_up", False))

    def test_nonfinite_kernel_output_stops_launch(self):
        layer = self.layer(lambda **kwargs: (torch.full_like(kwargs["v"], float("nan")), None))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            repair.warmup_prefill(layer, torch.empty(0, dtype=torch.bfloat16))
        self.assertFalse(getattr(layer, "_dci_prefill_warmed_up", False))

    def test_profile_hook_precedes_profile_return_and_leaves_inference_unchanged(self):
        events = []

        class Layer:
            def _forward_core(self, *args):
                events.append("original")
                return "original-result"

        class InheritedLayer(Layer):
            pass

        context = SimpleNamespace(attn_metadata=None)
        with patch.object(repair, "Qwen3NextGatedDeltaNet", Layer), \
             patch.object(repair, "version", return_value="0.17.0"), \
             patch.object(repair, "get_forward_context", return_value=context), \
             patch.object(repair, "warmup_prefill", side_effect=lambda *_: events.append("warmup")):
            repair.install_profile_warmup()
            repair.install_profile_warmup()
            layer = InheritedLayer()
            self.assertEqual(layer._forward_core(None, None, None, None), "original-result")
            self.assertEqual(events, ["warmup", "original"])
            context.attn_metadata = {"fixture": object()}
            events.clear()
            self.assertEqual(layer._forward_core(None, None, None, None), "original-result")
            self.assertEqual(events, ["original"])

    def test_worker_installs_hook_before_loading_model(self):
        events = []
        with patch.object(repair, "install_profile_warmup", side_effect=lambda: events.append("install")), \
             patch.object(repair.Worker, "load_model", side_effect=lambda: events.append("load")):
            repair.GDNWarmupWorker.load_model(object.__new__(repair.GDNWarmupWorker))
        self.assertEqual(events, ["install", "load"])

    def test_kv_allocation_requires_every_layer_to_have_completed_warmup(self):
        class Layer:
            pass

        layer = Layer()
        worker = object.__new__(repair.GDNWarmupWorker)
        worker.rank = 0
        worker.model_runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: [layer]))
        with patch.object(repair, "Qwen3NextGatedDeltaNet", Layer), \
             patch.object(repair.Worker, "determine_available_memory", return_value=123):
            with self.assertRaisesRegex(RuntimeError, "refusing KV"):
                worker.determine_available_memory()
            layer._dci_prefill_warmed_up = True
            self.assertEqual(worker.determine_available_memory(), 123)

    def test_changed_vllm_version_requires_review(self):
        with patch.object(repair, "version", return_value="0.18.0"):
            with self.assertRaisesRegex(RuntimeError, "requires reviewed"):
                repair.install_profile_warmup()

    def test_installed_qwen35_inherits_the_profile_method(self):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
        self.assertIs(Qwen3_5GatedDeltaNet._forward_core, repair.Qwen3NextGatedDeltaNet._forward_core)

    def test_standalone_probe_matches_model_fp32_state_and_records_it(self):
        from scripts import check_gdn_kernels as probe
        config = {"text_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 48,
                  "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                  "mamba_ssm_dtype": "float32"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            with patch("sys.argv", ["probe", "--model-config", str(root / "config.json"),
                                    "--tensor-parallel-size", "4", "--output", str(root / "result.json")]), \
                 patch.object(probe.torch.cuda, "is_available", return_value=True), \
                 patch.object(probe.torch.cuda, "device_count", return_value=1), \
                 patch.object(probe.torch.cuda, "device"), \
                 patch.object(probe.torch.cuda, "get_device_name", return_value="synthetic"), \
                 patch.object(probe.torch.cuda, "get_device_capability", return_value=(12, 0)), \
                 patch.object(probe.torch, "empty", return_value=None), \
                 patch.object(probe, "warmup_prefill") as warmup:
                probe.main()
                layer, _, lengths = warmup.call_args.args
                self.assertEqual(layer.get_state_dtype(), (torch.bfloat16, torch.float32))
                self.assertEqual(lengths, (16, 32, 64, 664, 2048))
            record = json.loads((root / "result.json").read_text())
            self.assertEqual(record["ssm_state_dtype"], "torch.float32")
            self.assertFalse(record["model_weights_loaded"])


if __name__ == "__main__":
    unittest.main()
