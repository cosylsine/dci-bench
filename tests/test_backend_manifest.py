import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dci_bench.backends.manifest import (
    build_sglang_manifest,
    build_vllm_manifest,
    inspect_generation_kwargs,
    load_backend_manifest,
    resolved_generation_config_record,
    sanitize_base_url,
    write_json_atomic,
)


class BackendManifestTest(unittest.TestCase):
    def _model(self, root: Path) -> Path:
        model = root / "model"
        (model / ".hfd").mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({"_name_or_path": "org/model", "model_type": "test"}), encoding="utf-8"
        )
        (model / "weights.safetensors").write_bytes(b"weights")
        (model / ".hfd" / "repo_metadata.json").write_text(
            json.dumps(
                {
                    "id": "org/model",
                    "sha": "abc123",
                    "siblings": [
                        {
                            "rfilename": "weights.safetensors",
                            "lfs": {"sha256": "f" * 64, "size": 7},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return model

    def test_build_write_and_validate_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = build_sglang_manifest(
                model_key="MiniCPM5-2B",
                model_path=self._model(root),
                served_model_name="MiniCPM5-2B",
                base_url="http://127.0.0.1:30000/v1/",
                sglang_version="1.2.3",
                sglang_git_revision="deadbeef",
                tool_call_parser="minicpm5",
                sampling_backend="pytorch",
                tensor_parallel_size=1,
                context_length=32768,
                dtype="bfloat16",
            )
            output = root / "backend.json"
            write_json_atomic(output, payload)
            loaded = load_backend_manifest(output)
            self.assertEqual(loaded["model"]["revision"], "abc123")
            self.assertTrue(loaded["model"]["weights"]["all_digests_available"])
            self.assertEqual(loaded["backend"]["base_url"], "http://127.0.0.1:30000/v1")

    def test_url_rejects_credentials_and_query(self):
        with self.assertRaises(ValueError):
            sanitize_base_url("http://user:secret@example.test/v1")
        with self.assertRaises(ValueError):
            sanitize_base_url("https://example.test/v1?token=secret")

    def test_vllm_manifest_is_accepted_by_the_shared_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = build_vllm_manifest(
                model_key="Qwen3.8-27B",
                model_path=self._model(root),
                served_model_name="Qwen3.8-27B",
                base_url="http://127.0.0.1:17892/v1",
                vllm_version="0.17.0",
                tool_call_parser="qwen3_coder",
                reasoning_parser="qwen3",
                language_model_only=True,
                tensor_parallel_size=4,
                context_length=32768,
                dtype="bfloat16",
                max_num_seqs=4,
                gpu_memory_utilization=0.80,
            )
            output = root / "vllm.json"
            write_json_atomic(output, payload)
            loaded = load_backend_manifest(output)
            self.assertEqual(loaded["backend"]["kind"], "vllm")
            self.assertEqual(loaded["backend"]["openai_service"], "vllm")
            self.assertEqual(loaded["backend"]["reasoning_parser"], "qwen3")
            self.assertTrue(loaded["backend"]["language_model_only"])
            self.assertEqual(loaded["backend"]["max_num_seqs"], 4)
            self.assertEqual(loaded["backend"]["gpu_memory_utilization"], 0.80)

    def test_digest_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = build_sglang_manifest(
                model_key="model",
                model_path=self._model(root),
                served_model_name="model",
                base_url="http://localhost:30000/v1",
                sglang_version="1",
                sglang_git_revision=None,
                tool_call_parser="parser",
                sampling_backend="pytorch",
                tensor_parallel_size=1,
                context_length=1024,
                dtype="float16",
            )
            output = root / "backend.json"
            write_json_atomic(output, payload)
            tampered = json.loads(output.read_text(encoding="utf-8"))
            tampered["backend"]["dtype"] = "bfloat16"
            output.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_backend_manifest(output)

    def test_weight_manifest_uses_stream_hash_when_lfs_digest_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self._model(root)
            (model / ".hfd" / "repo_metadata.json").write_text(
                json.dumps({"id": "org/model", "sha": "abc123", "siblings": []}),
                encoding="utf-8",
            )
            payload = build_sglang_manifest(
                model_key="model",
                model_path=model,
                served_model_name="model",
                base_url="http://localhost:30000/v1",
                sglang_version="1",
                sglang_git_revision=None,
                tool_call_parser="parser",
                sampling_backend="pytorch",
                tensor_parallel_size=1,
                context_length=1024,
                dtype="float16",
            )
            weight = payload["model"]["weights"]["files"][0]
            self.assertEqual(weight["digest_source"], "local_sha256")
            self.assertEqual(weight["sha256"], hashlib.sha256(b"weights").hexdigest())

    def test_weight_manifest_rejects_lfs_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self._model(root)
            (model / ".hfd" / "repo_metadata.json").write_text(
                json.dumps(
                    {
                        "id": "org/model",
                        "sha": "abc123",
                        "siblings": [
                            {
                                "rfilename": "weights.safetensors",
                                "lfs": {"sha256": "f" * 64, "size": 8},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "size differs"):
                build_sglang_manifest(
                    model_key="model",
                    model_path=model,
                    served_model_name="model",
                    base_url="http://localhost:30000/v1",
                    sglang_version="1",
                    sglang_git_revision=None,
                    tool_call_parser="parser",
                    sampling_backend="pytorch",
                    tensor_parallel_size=1,
                    context_length=1024,
                    dtype="float16",
                )

    def test_generation_config_is_shared_with_inspect_authority(self):
        kwargs = inspect_generation_kwargs()
        self.assertEqual(
            {
                key: kwargs[key]
                for key in ("temperature", "max_tokens", "seed")
            },
            {"temperature": 0, "max_tokens": 1024, "seed": 0},
        )
        self.assertEqual(
            kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        from inspect_ai.model import GenerateConfig

        resolved = GenerateConfig(**kwargs)
        self.assertEqual(resolved.temperature, 0)
        self.assertEqual(resolved.max_tokens, 1024)
        self.assertEqual(resolved.seed, 0)
        self.assertEqual(resolved.extra_body, kwargs["extra_body"])
        record = resolved_generation_config_record()
        self.assertEqual(record["authority"], "inspect.eval")
        self.assertEqual(record["max_model_output_tokens"], 1024)

    def test_generation_config_manifest_disagreement_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "generation_config"):
                build_sglang_manifest(
                    model_key="model",
                    model_path=self._model(root),
                    served_model_name="model",
                    base_url="http://localhost:30000/v1",
                    sglang_version="1",
                    sglang_git_revision=None,
                    tool_call_parser="parser",
                    sampling_backend="pytorch",
                    tensor_parallel_size=1,
                    context_length=1024,
                    dtype="float16",
                    generation_config={
                        "temperature": 1,
                        "max_tokens": 1024,
                        "seed": 0,
                        "enable_thinking": False,
                    },
                )

    def test_runtime_info_is_embedded_and_signed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_path = root / "serving-runtime.json"
            runtime = {
                "python": "3.11.16",
                "torch": "2.13.0",
                "cuda": "13.0",
                "gpu": "fixture",
                "driver": "fixture-driver",
            }
            runtime_path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            payload = build_vllm_manifest(
                model_key="model",
                model_path=self._model(root),
                served_model_name="model",
                base_url="http://localhost:30000/v1",
                vllm_version="1",
                tool_call_parser="parser",
                reasoning_parser=None,
                language_model_only=True,
                tensor_parallel_size=1,
                context_length=1024,
                dtype="float16",
                max_num_seqs=1,
                gpu_memory_utilization=0.8,
                runtime_info_path=runtime_path,
            )
            self.assertEqual(payload["backend"]["runtime"], runtime)
            output = root / "backend.json"
            write_json_atomic(output, payload)
            self.assertEqual(load_backend_manifest(output)["backend"]["runtime"], runtime)


if __name__ == "__main__":
    unittest.main()
