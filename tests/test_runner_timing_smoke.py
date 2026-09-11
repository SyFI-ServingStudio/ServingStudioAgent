import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "runner_timing_smoke",
    Path(__file__).resolve().parents[1] / "scripts/lib/runner-timing-smoke.py",
)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class RunnerTimingSmokeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.presets = self.root / "presets"
        self.presets.mkdir()
        self.preset = self.presets / "predict_llama3_8b_iter.json"
        self.config = {
            "gpu": "NVIDIA H200",
            "arch": {"iter": {"type": "llama3_dense"}},
            "cases_file": "cases.json",
            "log_dir": "logs/original",
        }
        self.preset.write_text(json.dumps(self.config))
        (self.presets / "cases.json").write_text("[]")

    def test_detects_b200_preserves_cases_and_uses_fresh_outputs(self):
        outputs = []

        def invoke(args, **kwargs):
            if args[0] == "nvidia-smi":
                return subprocess.CompletedProcess(
                    args, 0, "NVIDIA B200\nNVIDIA B200\n"
                )
            self.assertEqual(args[1:4], ["-m", "launcher", "timing-predict"])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertTrue(kwargs["check"])
            path = Path(args[4])
            self.assertEqual(path.parent, self.presets)
            config = json.loads(path.read_text())
            self.assertEqual(config["gpu"], "NVIDIA B200")
            self.assertEqual(config["arch"], self.config["arch"])
            self.assertEqual(config["cases_file"], self.config["cases_file"])
            self.assertTrue((path.parent / config["cases_file"]).is_file())
            output = Path(config["log_dir"])
            self.assertEqual(output.parent, self.root / "logs")
            outputs.append(output)
            (output / "reports").mkdir()
            (output / "reports/iter_breakdown.ans").write_text("result")
            return subprocess.CompletedProcess(args, 0)

        with patch.object(timing.subprocess, "run", side_effect=invoke):
            timing.run(self.root)
            timing.run(self.root)
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertEqual(json.loads(self.preset.read_text()), self.config)
        self.assertEqual(list(self.presets.glob(".runner-timing-*")), [])

    def test_empty_malformed_and_mixed_models_fail_before_launcher(self):
        for output in ("", "\n", "GPU,unexpected\n", "NVIDIA B200\nNVIDIA H200\n"):
            with self.subTest(output=output):
                with (
                    patch.object(
                        timing.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0, output),
                    ) as run,
                    self.assertRaises(ValueError),
                ):
                    timing.run(self.root)
                self.assertEqual(run.call_count, 1)
                self.assertFalse((self.root / "logs").exists())

    def test_probe_and_launcher_failures_and_missing_report_are_not_success(self):
        with (
            patch.object(
                timing.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(7, ["nvidia-smi"]),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            timing.run(self.root)
        probe = subprocess.CompletedProcess([], 0, "NVIDIA H200\n")
        with (
            patch.object(
                timing.subprocess,
                "run",
                side_effect=[probe, subprocess.CalledProcessError(8, ["launcher"])],
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            timing.run(self.root)
        with (
            patch.object(
                timing.subprocess,
                "run",
                side_effect=[probe, subprocess.CompletedProcess([], 0)],
            ),
            self.assertRaisesRegex(RuntimeError, "did not produce"),
        ):
            timing.run(self.root)
        self.assertEqual(list(self.presets.glob(".runner-timing-*")), [])
