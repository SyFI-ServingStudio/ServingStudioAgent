"""New explicit rendering preserves the established four instruction contracts."""

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import AgentMode, Role
from vibesim_agent.prompts import render
from vibesim_agent.prompts.render import Prompts


class PromptBundleTests(unittest.TestCase):
    def setUp(self):
        self.golden = json.loads(
            (Path(__file__).parent / "fixtures/legacy_prompts/golden.json").read_text()
        )

    def test_repair_continue_handoff_and_steer_match_baseline(self):
        with TemporaryDirectory() as directory:
            bundle = Prompts.prepare(Path(directory))
            for mode in AgentMode:
                options = {"agent_mode": mode.value, "conversation_id": "c"}
                self.assertEqual(
                    bundle.driver_repair_prompt("unparsed", **options),
                    self.golden["modes"][mode.value]["repair"],
                )
                self.assertEqual(
                    bundle.driver_continue_prompt("milestone", "completed", **options),
                    self.golden["modes"][mode.value]["continuation"],
                )
            self.assertEqual(
                bundle.orchestrator_handoff_prompt(
                    "task", "summary", conversation_id="c"
                ),
                self.golden["handoff"],
            )
            self.assertEqual(
                bundle.implementer_steer_prompt("correction"),
                self.golden["steer"],
            )
            for resume in (False, True):
                self.assertEqual(
                    bundle.implementer_prompt("task", is_resume=resume),
                    self.golden["implementer"][str(resume).lower()],
                )

    def test_rendered_bytes_and_driver_prompt_match_baseline(self):
        with TemporaryDirectory() as directory:
            bundle = Prompts.prepare(Path(directory))
            for name, expected in self.golden["rendered_sha256"].items():
                self.assertEqual(
                    hashlib.sha256((bundle.directory / name).read_bytes()).hexdigest(),
                    expected,
                    name,
                )
            sources = {
                p.name: p
                for p in (Path(render.__file__).parent / "contracts").iterdir()
                if p.is_file()
            }
            self.assertEqual(set(sources), set(self.golden["contracts_sha256"]))
            for name, source in sources.items():
                expected = self.golden["contracts_sha256"][name]
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(), expected, name
                )
                self.assertEqual(
                    hashlib.sha256((bundle.directory / name).read_bytes()).hexdigest(),
                    expected,
                    name,
                )
            for mode in AgentMode:
                self.assertEqual(
                    bundle.driver_prompt("hello", mode=mode, conversation_id="c"),
                    self.golden["modes"][mode.value]["driver"],
                )
            before = {p.name: p.stat().st_mtime_ns for p in bundle.directory.iterdir()}
            Prompts.prepare(bundle.directory)
            self.assertEqual(
                before,
                {p.name: p.stat().st_mtime_ns for p in bundle.directory.iterdir()},
            )

    def test_fingerprint_preserves_baseline_and_ignores_unused_roles(self):
        with TemporaryDirectory() as directory:
            bundle = Prompts.prepare(Path(directory))
            selection = RoleRuntime(
                "gpt", "codex:gpt:v1", "gpt-5.6-sol", "high", "default"
            )
            runtimes = {role: selection for role in Role}
            for mode in AgentMode:
                for autonomous in (False, True):
                    self.assertEqual(
                        bundle.fingerprint(
                            mode=mode, autonomous=autonomous, runtimes=runtimes
                        ),
                        next(
                            case["fingerprint"]
                            for case in self.golden["basic_fingerprints"]
                            if case["mode"] == mode.value
                            and case["autonomous"] == autonomous
                        ),
                    )
            before = bundle.fingerprint(
                mode=AgentMode.SINGLE, autonomous=False, runtimes=runtimes
            )
            runtimes[Role.IMPLEMENTER] = replace(selection, model_id="unused")
            self.assertEqual(
                before,
                bundle.fingerprint(
                    mode=AgentMode.SINGLE, autonomous=False, runtimes=runtimes
                ),
            )

    def test_fingerprint_matches_baseline_for_all_models_and_mixed_roles(self):
        with TemporaryDirectory() as directory:
            bundle = Prompts.prepare(Path(directory))
            self.assertTrue(self.golden["model_ids"])
            self.assertEqual(
                [case["model"] for case in self.golden["mixed"]],
                self.golden["model_ids"],
            )
            for case in self.golden["mixed"]:
                runtimes = {
                    Role(role): RoleRuntime(
                        value["provider"],
                        "explicit-scope",
                        value["model"],
                        value["effort"],
                        value["service_tier"],
                    )
                    for role, value in case["runtimes"].items()
                }
                for mode in AgentMode:
                    for autonomous in (False, True):
                        with self.subTest(
                            model=case["model"], mode=mode, autonomous=autonomous
                        ):
                            self.assertEqual(
                                bundle.fingerprint(
                                    mode=mode, autonomous=autonomous, runtimes=runtimes
                                ),
                                next(
                                    item["fingerprint"]
                                    for item in case["expected"]
                                    if item["mode"] == mode.value
                                    and item["autonomous"] == autonomous
                                ),
                            )
