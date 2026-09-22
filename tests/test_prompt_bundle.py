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
            # `implementer.txt` is rendered now, so that the host can name its
            # real contract path; the container's copy must still be the same
            # bytes the verbatim file was.
            self.assertEqual(
                set(sources) | {"implementer.txt"}, set(self.golden["contracts_sha256"])
            )
            for name, expected in self.golden["contracts_sha256"].items():
                if name in sources:
                    self.assertEqual(
                        hashlib.sha256(sources[name].read_bytes()).hexdigest(),
                        expected,
                        name,
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


class HostRenderingTests(unittest.TestCase):
    def test_a_host_rendering_names_the_repository_and_never_the_mount(self):
        with TemporaryDirectory() as directory:
            bundle = Prompts.prepare(Path(directory), workspace=Path("/repo/wt-x"))
            # On a host `/workspace` is not this repository -- on this machine
            # it exists and belongs to something else -- so one surviving
            # reference sends the agent to read or write the wrong tree.
            for path in bundle.directory.iterdir():
                self.assertNotIn("/workspace", path.read_text(), path.name)
            self.assertIn(
                "`/repo/wt-x/skills/skill-of-skills/SKILL.md`",
                bundle.contract_path(AgentMode.SINGLE, False).read_text(),
            )
            self.assertIn(
                "Conversation plan: `/repo/wt-x/c_plan.md`.",
                bundle.driver_prompt("hi", mode=AgentMode.SINGLE, conversation_id="c"),
            )

    def test_each_role_text_names_the_contract_its_turn_was_given(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = Prompts.prepare(root, workspace=Path("/repo/wt-x"))
            for autonomous in (False, True):
                bound = bundle.bound(autonomous=autonomous)
                for mode in AgentMode:
                    with self.subTest(mode=mode, autonomous=autonomous):
                        self.assertIn(
                            f"Read and follow `{bundle.contract_path(mode, autonomous)}`",
                            bound.role_text(mode.driver),
                        )
                self.assertIn(
                    f"`{bundle.contract_path(AgentMode.ORCHESTRATED, autonomous)}`",
                    bound.implementer_prompt("task", is_resume=False),
                )

    def test_a_container_rendering_ignores_autonomy_for_the_role_texts(self):
        with TemporaryDirectory() as directory:
            bundle = Prompts.prepare(Path(directory))
            self.assertEqual(
                bundle.bound(autonomous=True).role_text(Role.ORCHESTRATOR),
                bundle.role_text(Role.ORCHESTRATOR),
            )
            self.assertFalse((bundle.directory / "orchestrator.autonomous.txt").exists())
