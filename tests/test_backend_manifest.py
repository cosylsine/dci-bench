import json
import tempfile
import unittest
from pathlib import Path

from dci_bench.backends.manifest import (
    build_sglang_manifest,
    build_vllm_manifest,
    load_backend_manifest,
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


if __name__ == "__main__":
    unittest.main()
