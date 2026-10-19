"""Online code data isolation, per-sample scoring and publication failures."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import SampleScore, Score, Target
from inspect_ai.solver import TaskState

from dci_bench.agents.code_agent import _configuration, run_pi_code
from dci_bench.agents.pi_agent import run_pi_dci
from dci_bench.backends.manifest import build_sglang_manifest, validate_backend_manifest
from dci_bench.data.code_online import CodeSample, load_code_samples, validate_workspace
from dci_bench.protocol.code_runtime import code_runtime_contract, runtime_contract_sha256
from dci_bench.protocol.contracts import CONTRACT_SHA256, protocol_contract
from dci_bench.results.code_runner import execute_code_run
from dci_bench.scoring.code import load_json, score_query, score_run, snapshot_identity
from dci_bench.tasks.code_retrieval import code_eligible_mean, code_pi_solver, code_retrieval, code_retrieval_scorer, write_new_json
from dci_bench.tasks.mteb_llm_retrieval import BridgePortPool
from scripts.run_code_task import main


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/code_retrieval"


class CodeOnlineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        shutil.copytree(EXAMPLE, self.data)
        self.kwargs = dict(queries_path=self.data / "online_queries.json", labels_path=self.data / "labels.json")
        self.samples = load_code_samples(**self.kwargs, query_ids=["demo-trace", "demo-natural"])
        self.sample = self.samples[0]

    def prediction(self, sample=None):
        sample = sample or self.sample
        output = {"ranked_files": ["src/arithmetic.py"], "abstain": False,
                  "spans": [{"path": "src/arithmetic.py", "start_line": 5, "end_line": 6}]}
        if not sample.reference["gold_files"]:
            output = {"ranked_files": [], "abstain": True}
        return {**snapshot_identity(sample.reference), "output": output}

    def raw_score(self, sample, run_dir, prediction):
        directory = run_dir / "samples" / sample.directory_name
        write_new_json(directory / "scorer-prediction.json", prediction)
        write_new_json(directory / "scorer-score.json", score_query(prediction, sample.reference))

    def execute(self, evaluator, *, samples=None, name="run", **kwargs):
        with patch("dci_bench.results.code_runner.code_preflight"):
            return execute_code_run(samples or self.samples, run_dir=self.root / name,
                                    model="mockllm/model", base_url=None, mock_script=EXAMPLE / "mock_trace.json",
                                    inspect_eval_fn=evaluator, **kwargs)

    def test_selection_order_host_labels_and_opaque_query_ids(self):
        selected = load_code_samples(**self.kwargs, query_ids=["demo-natural", "demo-trace"])
        self.assertEqual([s.query_id for s in selected], ["demo-natural", "demo-trace"])
        all_samples = load_code_samples(**self.kwargs, all_queries=True, task_name="code2test")
        self.assertEqual([s.query_id for s in all_samples], ["demo-missing"])
        for selection in ({}, {"query_ids": []}, {"query_ids": ["missing"]},
                          {"query_ids": ["demo-trace"] * 2}, {"all_queries": True, "query_ids": ["demo-trace"]}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                load_code_samples(**self.kwargs, **selection)
        opaque = replace(self.sample, query_id="owner/repo/../q%20")
        self.assertNotIn("/", opaque.directory_name)
        self.assertEqual(len(opaque.directory_name), 64)
        task = code_retrieval([opaque], output_dir=self.root / "task")
        self.assertEqual(task.dataset[0].target, "")
        visible = json.dumps(task.dataset[0].model_dump(), default=str)
        for hidden in ("gold_files", "gold_functions", "arithmetic.divide", "no_gold_kind"):
            self.assertNotIn(hidden, visible)

    def test_completed_arb_export_loads_snapshot_and_keeps_export_immutable(self):
        import zstandard
        from dci_bench.data import expansion_code as adapter
        release, commit = "v2_trace2code", "a" * 40
        row = {"id": "arb-q1", "repo": "owner/repo", "base_commit": commit, "task_type": "trace2code",
               "query": {"failure": "observed failure"}, "gold": {"root_cause_files": ["src/a.py"]}}
        chunk = {"repo": "owner/repo", "base_commit": commit, "path": "src/a.py", "kind": "file",
                 "start_line": 1, "end_line": 2, "text": "def f():\n    return 1\n"}
        raw, export = self.root / "raw", self.root / "export"
        archive = raw / "releases" / release / f"agent_retrieval_bench_{release}.tar.zst"
        archive.parent.mkdir(parents=True)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as handle:
            for name, value in ((f"benchmark/{release}/samples.jsonl", row),
                                (f"corpus/{release}/owner__repo/{commit}.chunks.jsonl", chunk)):
                data = (json.dumps(value) + "\n").encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                handle.addfile(member, io.BytesIO(data))
        archive.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        with patch.dict(adapter.RELEASES, {release: (1, digest)}, clear=True):
            adapter.build_arb(raw, export, adapter.ARB_REVISION, (release,))
        samples = load_code_samples(code_root=export, all_queries=True)
        self.assertEqual(samples[0].query_id, "arb-q1")
        self.assertEqual(samples[0].export_root, export)
        self.assertEqual(samples[0].reference["base_commit"], commit)
        self.assertNotIn("gold_functions", samples[0].reference)
        self.assertEqual(json.loads(samples[0].query), row["query"])
        with self.assertRaisesRegex(ValueError, "immutable code export"):
            self.execute(Mock(), samples=samples, name="export/runs/test")
        adapter.verify_arb(export)
        (export / "_COMPLETE.json").unlink()
        with self.assertRaises((ValueError, OSError)):
            load_code_samples(code_root=export, all_queries=True)

    def test_query_label_sets_and_duplicates_fail_before_running(self):
        path = self.kwargs["queries_path"]
        rows = json.loads(path.read_text())
        for changed in (rows + rows[:1], rows[:-1]):
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_code_samples(**self.kwargs, all_queries=True)

    def test_snapshot_byte_token_and_tree_changes_are_rejected(self):
        candidate = self.data / "corpus/src/arithmetic.py"
        original = candidate.read_bytes()
        for changed in (original + b"# changed\n", original.replace(b"\n", b"\r\n")):
            candidate.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "changed"):
                load_code_samples(**self.kwargs, query_ids=["demo-trace"])
        candidate.write_bytes(original)
        extra = self.data / "corpus/empty-dir"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "unexpected candidate directory"):
            validate_workspace(self.data, self.sample.reference["files"])
        extra.rmdir()
        linked = self.data / "linked.py"
        os.link(candidate, linked)
        with self.assertRaisesRegex(ValueError, "unlinked"):
            validate_workspace(self.data, self.sample.reference["files"])
        linked.unlink()
        candidate.unlink()
        candidate.symlink_to(EXAMPLE / "corpus/src/arithmetic.py")
        with self.assertRaisesRegex(ValueError, "unlinked"):
            validate_workspace(self.data, self.sample.reference["files"])

    def test_code_runtime_is_independent_and_passes_only_public_files(self):
        before = protocol_contract()
        with patch.dict(os.environ, {"DCI_PROTOCOL_PROFILE": "budgeted-temp1"}):
            with _configuration(self.sample.reference["files"], 30) as (contract, files_path, env):
                self.assertEqual(contract["generation_config"]["temperature"], 0)
                self.assertEqual(contract["protocol_profile"], "code-v1")
                self.assertNotIn("budget_control", contract)
                self.assertNotIn("doc_id", json.dumps(contract))
                self.assertEqual(contract["final_answer_schema"]["properties"]["ranked_files"]["maxItems"], 30)
                self.assertEqual(load_json(files_path.read_text()), self.sample.reference["files"])
                self.assertEqual(env["DCI_PROTOCOL_CONTRACT_SHA256"], runtime_contract_sha256(contract))
                self.assertNotIn("gold", files_path.read_text())
        self.assertEqual(protocol_contract(), before)

    def test_inspect_aggregation_preserves_missing_annotation_denominators(self):
        refs = {s.query_id: deepcopy(s.reference) for s in self.samples}
        predictions = {s.query_id: self.prediction(s) for s in self.samples}
        refs["missing-positive"] = deepcopy(refs["demo-trace"])
        for ref in refs.values():
            ref.pop("gold_functionalities", None)
        scores = [SampleScore(score=Score(value=score_query(predictions.get(qid), ref)["metrics"]))
                  for qid, ref in refs.items()]
        preview = code_eligible_mean()(scores)
        expected = score_run(predictions, {"schema_version": "dci-code-labels-v1", "queries": refs})
        for key, values in expected["metrics"].items():
            if values["mean"] is None:
                self.assertNotIn(key, preview)
            else:
                self.assertEqual(preview[key], values["mean"])
        self.assertEqual(preview["file_recall_at_20"], 0.5)
        self.assertEqual(preview["no_gold_abstention"], 1)

    def test_real_inspect_reducer_handles_mixed_strata_and_unavailable_metrics(self):
        from inspect_ai import Task, eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_ai.solver import solver

        refs = {s.query_id: deepcopy(s.reference) for s in self.samples}
        # Every sample lacks this optional annotation; Inspect must not invent a zero.
        for ref in refs.values():
            ref.pop("gold_functionalities", None)
        samples = [replace(s, reference=refs[s.query_id]) for s in self.samples]
        missing = replace(samples[0], query_id="missing-positive")
        samples.append(missing)
        predictions = {s.query_id: self.prediction(s) for s in samples if s != missing}

        @solver
        def answers():
            async def solve(state, _generate):
                state.output.completion = json.dumps(predictions.get(str(state.sample_id)))
                return state
            return solve

        task = Task(name="code-mixed-aggregation",
                    dataset=MemoryDataset([Sample(id=s.query_id, input=s.query) for s in samples]),
                    solver=answers(), scorer=code_retrieval_scorer(
                        samples, self.root / "scores", (5, 10, 20), 20, (4000, 8000, 16000, 32000)))
        logs = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(self.root / "inspect"))
        self.assertEqual(logs[0].status, "success", str(logs[0].error))
        metrics = logs[0].results.scores[0].metrics
        self.assertEqual(metrics["file_recall_at_20"].value, 0.5)
        self.assertEqual(metrics["no_gold_abstention"].value, 1)
        self.assertNotIn("functionality_recall", metrics)
        saved = load_json((self.root / "scores" / samples[0].directory_name / "scorer-score.json").read_text())
        self.assertIsNone(saved["metrics"]["functionality_recall"])

    def test_real_task_state_is_scored_immediately_without_an_error_attribute(self):
        state = TaskState(model="mockllm/model", sample_id=self.sample.query_id, epoch=1,
                          input=self.sample.query, messages=[])
        state.output = ModelOutput.from_content("mockllm/model", json.dumps(self.prediction()))
        scorer = code_retrieval_scorer([self.sample], self.root, (5, 10, 20), 20, (4000, 8000, 16000, 32000))
        result = asyncio.run(scorer(state, Target("")))
        self.assertEqual(result.value["functionality_recall"], 1)
        self.assertTrue((self.root / self.sample.directory_name / "scorer-score.json").is_file())

    def test_solver_resolves_inspect_timeout_on_event_loop_before_worker(self):
        state = SimpleNamespace(sample_id=self.sample.query_id, input_text=self.sample.query,
                                output=SimpleNamespace(completion=""))

        def remaining():
            asyncio.get_running_loop()  # Fails if moved into the worker thread.
            return 17.5

        fake_result = SimpleNamespace(valid_output=True, failure=None, code_output=self.prediction()["output"])
        solve = code_pi_solver([self.sample], self.root, BridgePortPool(13131, 1), 65536, 20, EXAMPLE / "mock_trace.json")
        with patch("dci_bench.tasks.code_retrieval._remaining_pi_timeout_seconds", side_effect=remaining), \
             patch("dci_bench.tasks.code_retrieval.run_pi_code", return_value=fake_result) as run:
            asyncio.run(solve(state, None))
        self.assertEqual(run.call_args.kwargs["timeout_seconds"], 17.5)
        self.assertEqual(run.call_args.kwargs["task_instruction"], "Retrieval task type: trace2code")
        self.assertEqual(json.loads(state.output.completion), self.prediction())

    def test_code_timeout_clears_a_late_valid_answer_and_document_ignores_code_env(self):
        for track in ("code", "document"):
            with self.subTest(track=track):
                directory = self.root / track
                directory.mkdir()
                output = directory / "final.json"
                output.write_text(json.dumps({"valid_output": True, "code_output": self.prediction()["output"]}))
                process = Mock(pid=12345, returncode=-15)
                process.communicate.side_effect = [subprocess.TimeoutExpired("node", 1), ("", "")]
                kwargs = dict(query="q", workspace=self.data, output_path=output, timeout_seconds=1)
                with patch("dci_bench.agents.pi_agent.subprocess.Popen", return_value=process) as popen, \
                     patch("dci_bench.agents.pi_agent.os.killpg"), \
                     patch.dict(os.environ, {"DCI_EVALUATION_TRACK": "code", "DCI_CODE_RUNTIME_CONTRACT": "/untrusted"}):
                    result = run_pi_code(files=self.sample.reference["files"], **kwargs) if track == "code" else run_pi_dci(**kwargs)
                self.assertFalse(result.valid_output)
                self.assertIsNone(result.code_output)
                self.assertEqual(result.failure_kind, "timeout")
                env = popen.call_args.kwargs["env"]
                self.assertEqual(env["DCI_EVALUATION_TRACK"], track)
                if track == "document":
                    self.assertNotIn("DCI_CODE_RUNTIME_CONTRACT", env)
                    self.assertEqual(env["DCI_PROTOCOL_CONTRACT_SHA256"], CONTRACT_SHA256)
                else:
                    self.assertEqual(env["DCI_PROTOCOL_PROFILE"], "code-v1")
                    self.assertIsNone(json.loads(output.read_text())["code_output"])

    def test_terminal_inspect_limits_override_valid_raw_scores_in_both_strata(self):
        for kind, expected_kind in (("token", "token_limit"), ("time", "timeout"), ("context", "context_overflow")):
            with self.subTest(kind=kind):
                run_dir = self.root / kind

                def evaluate(task, **kwargs):
                    self.assertEqual(len(task.dataset), 2)
                    self.assertTrue(kwargs["score_on_error"])
                    for sample in self.samples:
                        self.raw_score(sample, run_dir, self.prediction(sample))
                    return [SimpleNamespace(status="success", samples=[
                        SimpleNamespace(id=s.query_id, error=None, limit={"type": kind, "limit": 10}) for s in self.samples
                    ])]

                report = self.execute(evaluate, name=kind)
                self.assertEqual(report["status"], "complete_with_sample_failures")
                self.assertEqual(report["invalid_outputs"], 2)
                self.assertEqual(report["failure_counts"], {expected_kind: 2})
                self.assertEqual(report["metrics"]["file_recall_at_20"]["mean"], 0)
                self.assertEqual(report["metrics"]["no_gold_abstention"]["mean"], 0)
                self.assertEqual(report["infrastructure_failures"], [])
                raw = json.loads((run_dir / "samples" / self.sample.directory_name / "scorer-score.json").read_text())
                self.assertTrue(raw["valid_output"])

    def test_logged_error_wins_over_valid_scoring_and_secrets_are_redacted(self):
        def evaluate(_task, **_kwargs):
            self.raw_score(self.sample, self.root / "run", self.prediction())
            return [SimpleNamespace(status="success", samples=[SimpleNamespace(
                id=self.sample.query_id, error=SimpleNamespace(message="broken bridge secret-value"), limit=None)])]
        report = self.execute(evaluate, samples=[self.sample], api_key="secret-value")
        self.assertEqual(report["status"], "infrastructure_error")
        self.assertEqual(report["valid_outputs"], 0)
        self.assertNotIn("secret-value", json.dumps(report))

    def test_all_planned_samples_survive_missing_records_and_hashes_verify(self):
        def evaluate(_task, **_kwargs):
            self.raw_score(self.sample, self.root / "run", self.prediction())
            return [SimpleNamespace(status="success", samples=[SimpleNamespace(id=self.sample.query_id, error=None)])]
        report = self.execute(evaluate)
        self.assertEqual((report["num_queries"], report["valid_outputs"], report["invalid_outputs"]), (2, 1, 1))
        self.assertEqual(report["status"], "infrastructure_error")
        complete = json.loads((self.root / "run/_COMPLETE.json").read_text())
        for path, expected in complete["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.root / "run" / path).read_bytes()).hexdigest(), expected)
        self.assertFalse((self.root / "run/task-result.json").exists())
        with self.assertRaises(FileExistsError):
            self.execute(evaluate)

    def test_scorer_mismatch_never_publishes_completion(self):
        def evaluate(_task, **_kwargs):
            directory = self.root / "run/samples" / self.sample.directory_name
            write_new_json(directory / "scorer-prediction.json", self.prediction())
            write_new_json(directory / "scorer-score.json", {"tampered": True})
            return [SimpleNamespace(status="success", samples=[SimpleNamespace(id=self.sample.query_id)])]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.execute(evaluate, samples=[self.sample])
        self.assertFalse((self.root / "run/_COMPLETE.json").exists())

    def test_export_and_candidate_directories_are_never_used_for_results(self):
        sample = replace(self.sample, export_root=self.root)
        with self.assertRaisesRegex(ValueError, "immutable code export"):
            self.execute(Mock(), samples=[sample])
        with self.assertRaisesRegex(ValueError, "candidate workspace"):
            self.execute(Mock(), name="data/run")
        self.assertFalse((self.data / "run").exists())

    def test_cli_preflight_starts_no_evaluator_and_creates_no_output(self):
        argv = ["--queries", str(self.kwargs["queries_path"]), "--labels", str(self.kwargs["labels_path"]),
                "--query-ids", "demo-trace", "--output-dir", str(self.root / "run"), "--preflight-only"]
        with patch("dci_bench.results.code_runner.code_preflight") as preflight, \
             patch("dci_bench.results.code_runner.execute_code_run") as execute, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(argv), 0)
        preflight.assert_called_once()
        execute.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_explicit_code_generation_contract_does_not_change_document_validation(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "weights.safetensors").write_bytes(b"fixture")
        manifest = build_sglang_manifest(
            model_key="test", model_path=model, served_model_name="test", base_url="http://localhost:1234/v1",
            sglang_version="test", sglang_git_revision=None, tool_call_parser="test", sampling_backend="test",
            tensor_parallel_size=1, context_length=65536, dtype="float16",
        )
        with patch("dci_bench.backends.manifest.GENERATION_CONFIG", {**manifest["backend"]["generation_config"], "temperature": 1}):
            validate_backend_manifest(manifest, generation_contract=code_runtime_contract())
            with self.assertRaisesRegex(ValueError, "shared protocol"):
                validate_backend_manifest(manifest)
        changed = deepcopy(manifest)
        changed["backend"]["generation_config"]["temperature"] = 1
        with self.assertRaisesRegex(ValueError, "selected track"):
            validate_backend_manifest(changed, generation_contract=code_runtime_contract())


if __name__ == "__main__":
    unittest.main()
