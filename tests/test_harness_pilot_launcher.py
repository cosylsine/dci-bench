"""CPU-only regression checks for source snapshots and pilot startup ordering."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "Git required for source snapshots")
class HarnessPilotLauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dci-pilot-launcher-test-")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.root = self.temporary / "checkout"
        self.root.mkdir()
        self.global_config = self.temporary / "gitconfig"
        self.global_config.write_text("[user]\n\tname = Pilot Test\n\temail = pilot@example.invalid\n")
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        for key in ("MAX_CONCURRENCY", "VLLM_MAX_NUM_SEQS"):
            self.env.pop(key, None)
        self.env.update({"GIT_CONFIG_GLOBAL": str(self.global_config), "GIT_CONFIG_NOSYSTEM": "1",
                         "PYTHONDONTWRITEBYTECODE": "1"})
        (self.root / "scripts").mkdir()
        shutil.copyfile(REPO / "scripts/run_harness_pilot.py", self.root / "scripts/run_harness_pilot.py")
        (self.root / "README.md").write_text("Original source.\n")
        (self.root / ".gitignore").write_text("pi-dci/\n")
        pi = self.root / "pi-dci"
        (pi / "scripts").mkdir(parents=True)
        (pi / "scripts/dci-run.ts").write_text("export const fixture = 1;\n")
        for root in (self.root, pi):
            self.git(root, "init", "--quiet")
            self.git(root, "add", ".")
            self.git(root, "-c", "user.name=Pilot Test", "-c", "user.email=pilot@example.invalid",
                     "commit", "--quiet", "--no-gpg-sign", "-m", "fixture")
        (self.root / "README.md").write_text("Local source change.\n")
        (pi / "scripts/dci-run.ts").write_text("export const fixture = 2;\n")
        (self.root / "scripts/untracked.json").write_text('{"new_source": true}\n')
        (self.root / "dci_bench/protocol").mkdir(parents=True)
        (self.root / "dci_bench/protocol/stable-rg.conf").write_text("--sort=path\n")
        spec = importlib.util.spec_from_file_location("harness_pilot_launcher_test", REPO / "scripts/run_harness_pilot.py")
        self.pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pilot)
        self.pilot.REPO = self.root

    def git(self, root, *args):
        return subprocess.run(["git", *args], cwd=root, env=self.env, capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()

    def test_shared_mount_ownership_keeps_revisions_patches_and_archive(self):
        expected_revisions = {label: self.git(root, "rev-parse", "HEAD")
                              for label, root in (("dci", self.root), ("pi", self.root / "pi-dci"))}
        original_config = self.global_config.read_bytes()
        self.env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"
        # Reproduce the cluster's bare Git failure before checking the fix.
        for root in (self.root, self.root / "pi-dci"):
            bare = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root,
                                  env=self.env, capture_output=True, timeout=30)
            self.assertNotEqual(bare.returncode, 0)
        output = self.temporary / "snapshot"
        output.mkdir()
        with patch.dict(os.environ, self.env, clear=True):
            hashes = self.pilot.snapshot_sources(output)
        self.assertIn("scripts/untracked.json", hashes)
        self.assertIn("dci_bench/protocol/stable-rg.conf", hashes)
        self.assertEqual(json.loads((output / "source-hashes.json").read_text()), hashes)
        with tarfile.open(output / "sources.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), set(hashes))
            for name, digest in hashes.items():
                self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), digest)
        for label, revision in expected_revisions.items():
            self.assertEqual((output / f"{label}.revision").read_text().strip(), revision)
            self.assertIn("diff --git", (output / f"{label}.patch").read_text())
        audit = json.loads((output / "git-audit.json").read_text())
        self.assertTrue(all(item["available"] for item in audit.values()))
        self.assertEqual(self.global_config.read_bytes(), original_config)

    def test_missing_git_keeps_source_archive_and_records_unavailable_metadata(self):
        output = self.temporary / "snapshot"
        output.mkdir()
        no_binaries = self.temporary / "empty-bin"
        no_binaries.mkdir()
        errors = io.StringIO()
        with patch.dict(os.environ, {**self.env, "PATH": str(no_binaries)}, clear=True), redirect_stderr(errors):
            hashes = self.pilot.snapshot_sources(output)
        self.assertTrue((output / "sources.tar.gz").is_file())
        self.assertEqual(json.loads((output / "source-hashes.json").read_text()), hashes)
        audit = json.loads((output / "git-audit.json").read_text())
        self.assertEqual(set(audit), {"dci", "pi"})
        for label, item in audit.items():
            self.assertFalse(item["available"])
            self.assertIn("Source snapshot Git command failed", item["error"])
            self.assertFalse((output / f"{label}.revision").exists())
            self.assertFalse((output / f"{label}.patch").exists())
        self.assertIn("optional Git metadata unavailable", errors.getvalue())

    def test_linked_worktree_gitdir_marker_is_resolved(self):
        linked = self.temporary / "pi-linked"
        expected = self.git(self.root / "pi-dci", "rev-parse", "HEAD")
        self.git(self.root / "pi-dci", "worktree", "add", "--quiet", "--detach", str(linked), "HEAD")
        self.assertTrue((linked / ".git").is_file())
        (linked / "scripts/dci-run.ts").write_text("export const fixture = 3;\n")
        self.env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"
        with patch.dict(os.environ, self.env, clear=True):
            revision = self.pilot.git_output(linked, "rev-parse", "HEAD").decode().strip()
            diff = self.pilot.git_output(linked, "diff", "--binary", "HEAD", "--").decode()
        self.assertEqual(revision, expected)
        self.assertIn("+export const fixture = 3;", diff)

    def test_source_preflight_needs_no_backend_and_creates_no_run(self):
        self.env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"
        result = subprocess.run([sys.executable, str(self.root / "scripts/run_harness_pilot.py"),
                                 "--preflight-only"], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("source snapshot preflight passed", result.stdout)
        self.assertFalse((self.root / "results").exists())

    def launcher_fixture(self, *, missing=(), uid=0, apt_exit=0, sandbox_exit=0, data_exit=0):
        # Only the isolated copy points at fixture OS tools. Production always
        # requires /usr/bin paths; these tests never install system packages.
        binaries = self.temporary / "bin"
        binaries.mkdir()
        trusted = self.temporary / "usr-bin"
        trusted.mkdir()
        for tool in ("bwrap", "rg"):
            if tool not in missing:
                (trusted / tool).write_text("#!/bin/sh\nexit 0\n")
                (trusted / tool).chmod(0o755)
        launcher = self.root / "scripts/run_harness_pilot.sh"
        launcher.write_text((REPO / "scripts/run_harness_pilot.sh").read_text()
                            .replace("/usr/bin/bwrap", str(trusted / "bwrap"))
                            .replace("/usr/bin/rg", str(trusted / "rg")))
        events = self.temporary / "events"
        event_path = shlex.quote(str(events))
        bench = binaries / "bench-python"
        bench.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = scripts/run_all.py ]; then echo data >> {event_path}; exit {data_exit}; fi\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n'
        )
        bench.chmod(0o755)
        serving = binaries / "serving-python"
        marker = self.temporary / "unexpected-serving-start"
        serving.write_text(f"#!/bin/sh\necho serving >> {event_path}\n"
                           f"printf invoked > {shlex.quote(str(marker))}\nexit 97\n")
        serving.chmod(0o755)
        for name, body in {
            "id": f"echo {uid}\n",
            "npm": f'echo "sandbox $DCI_PROTOCOL_PROFILE" >> {event_path}\nexit {sandbox_exit}\n',
            "apt-get": (
                f'echo "apt $*" >> {event_path}\n'
                f'if [ {apt_exit} != 0 ]; then exit {apt_exit}; fi\n'
                'if [ "$1" = install ]; then\n'
                '  for package in "$@"; do\n'
                '    case "$package" in bubblewrap) tool=bwrap;; ripgrep) tool=rg;; *) continue;; esac\n'
                f'    touch {shlex.quote(str(trusted))}/"$tool"\n'
                f'    chmod +x {shlex.quote(str(trusted))}/"$tool"\n'
                '  done\nfi\n'
            ),
        }.items():
            executable = binaries / name
            executable.write_text("#!/bin/sh\n" + body)
            executable.chmod(0o755)
        output = self.temporary / "pilot-output"
        env = {**self.env, "BENCH_PYTHON": str(bench), "SERVING_PYTHON": str(serving),
               "PILOT_ROOT": str(output), "DRY_RUN": "0"}
        return launcher, env, events, marker, output

    def run_launcher(self, fixture):
        launcher, env, *_ = fixture
        return subprocess.run(["bash", str(launcher), "--profiles", "v3", "budgeted"],
                              env=env, capture_output=True, text=True, timeout=30)

    def test_git_failure_is_reported_but_allows_launcher_to_continue(self):
        fixture = self.launcher_fixture()
        _, env, events, marker, output = fixture
        binaries = self.temporary / "bin"
        git = binaries / "git"
        git.write_text("#!/bin/sh\nprintf '%s\\n' 'snapshot-git-error-fixture' >&2\nexit 129\n")
        git.chmod(0o755)
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 97, result.stdout + result.stderr)
        self.assertIn("source snapshot preflight passed", result.stdout)
        self.assertIn("optional Git metadata available for 0/2 checkouts", result.stdout)
        self.assertIn("exit 129", result.stderr)
        self.assertIn("snapshot-git-error-fixture", result.stderr)
        self.assertIn(str(self.root), result.stderr)
        self.assertTrue(marker.exists(), "Optional Git failure blocked serving startup")
        self.assertTrue(output.exists())
        self.assertEqual(events.read_text().splitlines(), ["data", "sandbox v3", "sandbox budgeted", "serving"])

    def test_missing_packages_are_installed_before_sandbox_and_cuda(self):
        fixture = self.launcher_fixture(missing=("bwrap", "rg"))
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 97, result.stdout + result.stderr)
        self.assertEqual(fixture[2].read_text().splitlines(), [
            "apt update -o Acquire::Retries=3",
            "apt install -y --no-install-recommends bubblewrap ripgrep",
            "data", "sandbox v3", "sandbox budgeted", "serving",
        ])

    def test_only_missing_package_is_installed(self):
        fixture = self.launcher_fixture(missing=("rg",))
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 97, result.stdout + result.stderr)
        self.assertEqual(fixture[2].read_text().splitlines()[1],
                         "apt install -y --no-install-recommends ripgrep")

    def test_installation_failure_stops_before_data_sandbox_and_cuda(self):
        fixture = self.launcher_fixture(missing=("bwrap", "rg"), apt_exit=100)
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("OS package installation failed", result.stderr)
        self.assertEqual(fixture[2].read_text().splitlines(), ["apt update -o Acquire::Retries=3"])
        self.assertFalse(fixture[3].exists())
        self.assertFalse(fixture[4].exists())

    def test_missing_packages_without_root_stop_without_attempting_install(self):
        fixture = self.launcher_fixture(missing=("bwrap",), uid=1000)
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Cannot install missing OS packages", result.stderr)
        self.assertFalse(fixture[2].exists())
        self.assertFalse(fixture[3].exists())
        self.assertFalse(fixture[4].exists())

    def test_sandbox_failure_after_installation_stops_before_cuda(self):
        fixture = self.launcher_fixture(missing=("bwrap", "rg"), sandbox_exit=1)
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Pi sandbox preflight failed", result.stderr)
        self.assertEqual(fixture[2].read_text().splitlines()[-2:], ["data", "sandbox v3"])
        self.assertFalse(fixture[3].exists())
        self.assertFalse(fixture[4].exists())

    def test_data_failure_stops_before_sandbox_and_cuda(self):
        fixture = self.launcher_fixture(data_exit=2)
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(fixture[2].read_text().splitlines(), ["data"])
        self.assertFalse(fixture[3].exists())
        self.assertFalse(fixture[4].exists())

    def test_new_profiles_preflight_before_model_startup(self):
        fixture = self.launcher_fixture()
        launcher, env, events, *_ = fixture
        result = subprocess.run(["bash", str(launcher), "--profiles", "budgeted-stable", "budgeted-selective"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 97, result.stdout + result.stderr)
        self.assertEqual(events.read_text().splitlines(), [
            "data", "sandbox budgeted-stable", "sandbox budgeted-selective", "serving"])

    def test_validation_freeze_failure_stops_before_data_sandbox_and_cuda(self):
        fixture = self.launcher_fixture()
        launcher, env, events, marker, output = fixture
        (self.root / "scripts/harness_pilot_queries.json").write_text('{"TaskA": ["Q1"]}')
        result = subprocess.run(["bash", str(launcher), "--profiles", "budgeted-stable", "budgeted-selective",
                                 "--query-suite", "validation", "--profile-order", "alternating"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("required validation freeze is missing", result.stderr)
        self.assertFalse(events.exists())
        self.assertFalse(marker.exists())
        self.assertFalse(output.exists())

    def test_dry_run_does_not_install_packages_or_probe_sandbox(self):
        fixture = self.launcher_fixture(missing=("bwrap", "rg"))
        fixture[1]["DRY_RUN"] = "1"
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(fixture[2].exists())
        self.assertFalse(fixture[4].exists())

    def test_shell_concurrency_configures_serving_capacity(self):
        fixture = self.launcher_fixture()
        launcher, env, *_ = fixture
        env.update({"DRY_RUN": "1", "MAX_CONCURRENCY": "4"})
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("concurrency=4 max_num_seqs=4", result.stdout)
        for option in (["--max-concurrency", "8"], ["--max-concurrency=8"]):
            result = subprocess.run(["bash", str(launcher), *option], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("concurrency=8 max_num_seqs=8", result.stdout)

    def test_shell_rejects_insufficient_capacity_before_startup(self):
        fixture = self.launcher_fixture()
        fixture[1].update({"MAX_CONCURRENCY": "4", "VLLM_MAX_NUM_SEQS": "1"})
        result = self.run_launcher(fixture)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("must be at least MAX_CONCURRENCY", result.stderr)
        self.assertFalse(fixture[2].exists())
        self.assertFalse(fixture[4].exists())

    def test_required_preflight_failure_stops_before_cuda_or_model_start(self):
        binaries = self.temporary / "bin"
        binaries.mkdir()
        bench = binaries / "bench-python"
        bench.write_text("#!/bin/sh\nprintf '%s\\n' 'source archive write error fixture' >&2\nexit 23\n")
        bench.chmod(0o755)
        serving = binaries / "serving-python"
        marker = self.temporary / "unexpected-serving-start"
        serving.write_text(f"#!/bin/sh\nprintf invoked > {shlex.quote(str(marker))}\nexit 97\n")
        serving.chmod(0o755)
        output = self.temporary / "pilot-output"
        env = {**self.env, "BENCH_PYTHON": str(bench), "SERVING_PYTHON": str(serving),
               "PILOT_ROOT": str(output), "DRY_RUN": "0"}
        result = subprocess.run(["bash", str(REPO / "scripts/run_harness_pilot.sh"),
                                 "--profiles", "v3", "budgeted"], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertIn("source archive write error fixture", result.stderr)
        self.assertFalse(marker.exists(), "Serving interpreter ran before source preflight passed")
        self.assertFalse(output.exists())


class HarnessPilotConcurrencyTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dci-pilot-concurrency-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/harness_pilot_queries.json").write_text(
            json.dumps({"TaskA": ["Q1", "Q2"], "TaskB": ["Q3", "Q4"]}))
        self.backend = self.root / "backend.json"
        self.backend.write_text(json.dumps({"model": {"model_key": "fixture"},
                                           "backend": {"max_num_seqs": 8}, "manifest_sha256": "fixture"}))
        spec = importlib.util.spec_from_file_location("pilot_concurrency_test", REPO / "scripts/run_harness_pilot.py")
        self.pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pilot)
        self.pilot.REPO = self.root
        self.selector = ModuleType("dci_bench.tasks.mteb_llm_retrieval")
        self.selector._select_query_ids = lambda *args, **kwargs: kwargs["query_ids"]

    def invoke(self, *args, environment_concurrency="1"):
        output = io.StringIO()
        with patch.dict(sys.modules, {"dci_bench.tasks.mteb_llm_retrieval": self.selector}), \
             patch.dict(os.environ, {"MAX_CONCURRENCY": environment_concurrency}), \
             patch.object(self.pilot.subprocess, "check_output", return_value=json.dumps(
                 {"sha256": "contract", "contract": {"generation_config": {"temperature": 0}}})), \
             redirect_stdout(output):
            self.pilot.main(["--backend-manifest", str(self.backend), "--profiles", "v3", "budgeted", *args])
        return output.getvalue()

    def test_concurrent_plan_reserves_disjoint_port_pools(self):
        plan = json.loads(self.invoke("--dry-run", "--max-concurrency", "4"))
        self.assertEqual(plan["max_concurrency"], 4)
        self.assertEqual(plan["query_ids"], {"TaskA": ["Q1", "Q2"], "TaskB": ["Q3", "Q4"]})
        self.assertEqual(plan["bridge_ports"], {
            "v3": {"TaskA": 28131, "TaskB": 28135},
            "budgeted": {"TaskA": 28139, "TaskB": 28143},
        })

    def test_new_profiles_share_queries_and_have_disjoint_ports(self):
        plan = json.loads(self.invoke("--dry-run", "--profiles", "budgeted-stable", "budgeted-selective",
                                      "--profile-order", "alternating", "--max-concurrency", "4"))
        self.assertEqual(set(plan["bridge_ports"]), {"budgeted-stable", "budgeted-selective"})
        self.assertEqual(plan["bridge_ports"]["budgeted-selective"]["TaskB"], 28143)
        self.assertEqual(plan["query_ids"], {"TaskA": ["Q1", "Q2"], "TaskB": ["Q3", "Q4"]})

    def test_environment_concurrency_reaches_plan_and_cli_can_override_it(self):
        plan = json.loads(self.invoke("--dry-run", environment_concurrency="4"))
        self.assertEqual(plan["max_concurrency"], 4)
        plan = json.loads(self.invoke("--dry-run", "--max-concurrency", "2", environment_concurrency="4"))
        self.assertEqual(plan["max_concurrency"], 2)

    def test_concurrency_and_port_limits_fail_before_running_tasks(self):
        for args in (("--max-concurrency", "0"), ("--max-concurrency", "9"),
                     ("--max-concurrency", "4", "--bridge-port", "65521")):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke("--dry-run", *args)
                self.assertEqual(error.exception.code, 2)
        plan = json.loads(self.invoke("--dry-run", "--max-concurrency", "4", "--bridge-port", "65520"))
        self.assertEqual(plan["bridge_ports"]["budgeted"]["TaskB"] + 3, 65535)

    def test_concurrency_reaches_each_task_command(self):
        commands = []

        def run_task(command, **kwargs):
            commands.append(command)
            value = lambda flag: command[command.index(flag) + 1]
            run = Path(value("--results-root")) / "fixture" / value("--task") / "runs" / value("--run-id")
            run.mkdir(parents=True)
            (run / "_COMPLETE").write_text("{}")
            (run / "task-result.json").write_text(json.dumps({
                "data": {}, "counts": {"planned": 2, "valid": 2},
                "usage": {"total_tokens": {"sum": 100}},
                "metrics": {group: {metric: {"mean": 1} for metric in
                                     ("recall_at_10", "f1_at_10", "ndcg_at_10")}
                            for group in ("macro", "valid_only")},
            }))
            return subprocess.CompletedProcess(command, 0)

        for order, ports in (("as-listed", [28131, 28135, 28139, 28143]),
                             ("alternating", [28139, 28131, 28135, 28143])):
            with self.subTest(order=order):
                commands.clear()
                output = self.root / f"evaluation-{order}"
                with patch.object(self.pilot, "snapshot_sources", return_value={}), \
                     patch.object(self.pilot.subprocess, "run", side_effect=run_task):
                    self.invoke("--max-concurrency", "4", "--output-root", str(output),
                                "--profile-order", order)
                self.assertEqual(len(commands), 4)
                for command in commands:
                    self.assertEqual(command[command.index("--max-concurrency") + 1], "4")
                self.assertEqual([int(command[command.index("--bridge-port") + 1]) for command in commands], ports)
                plan = json.loads((output / "plan.json").read_text())
                self.assertEqual(plan["max_concurrency"], 4)
                self.assertEqual([(command[command.index("--task") + 1],
                                   command[command.index("--run-id") + 1]) for command in commands],
                                 [(item["task"], f"pilot-{item['profile']}") for item in plan["execution_order"]])


if __name__ == "__main__":
    unittest.main()
