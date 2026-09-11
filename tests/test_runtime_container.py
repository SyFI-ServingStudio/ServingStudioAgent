import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from tests.runtime_fixtures import execution_environment
from vibesim_agent.runtime.container import (
    CONFIG_LABEL,
    CREATION_LABEL,
    OWNER_LABEL,
    ContainerManager,
    ContainerSpec,
)
from vibesim_agent.runtime.mounts import Mount

IMAGE = "sha256:" + "a" * 64
CREATED = "b" * 64
EXISTING = "c" * 64


class Docker:
    def __init__(self):
        self.calls = []
        self.options = []
        self.current = None
        self.image = IMAGE
        self.fail = None
        self.fail_readiness = False
        self.inspect_error = None
        self.create_error = None
        self.creation_matches = True

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        self.options.append(kwargs)
        code, stdout, stderr = 0, "", ""
        if command[1:3] == ["image", "inspect"]:
            stdout = self.image
        elif command[1:3] == ["container", "inspect"]:
            if self.inspect_error:
                code, stderr = 1, self.inspect_error
            elif self.current is None:
                code, stderr = 1, "Error: No such container: test"
            else:
                stdout = json.dumps([self.current])
        elif command[1] == "create":
            stdout = CREATED
            if self.create_error:
                labels = dict(
                    command[i + 1].split("=", 1)
                    for i, value in enumerate(command)
                    if value == "--label"
                )
                if not self.creation_matches:
                    labels[CREATION_LABEL] = "another-invocation"
                self.current = {"Id": CREATED, "Config": {"Labels": labels}}
                if self.create_error == "timeout":
                    raise subprocess.TimeoutExpired(command, 180)
                if self.create_error == "invalid":
                    stdout = ""
                else:
                    code = 1
        elif command[1] == "exec" and self.fail_readiness:
            code = 1
            self.fail_readiness = False
        if command[1] == self.fail:
            code, stderr = 1, "private upstream output"
        return subprocess.CompletedProcess(command, code, stdout, stderr)


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.docker = Docker()
        self.manager = ContainerManager(
            execution_environment(), run=self.docker
        )
        self.spec = ContainerSpec(
            "test",
            "w/c",
            (Mount(self.root, PurePosixPath("/workspace")),),
            ("/roles/assistant/scope",),
            ("claude",),
            "/opt/vibesim/prompts/AGENTS.single.md",
            (("assistant", "scope"),),
        )

    def existing(self, **overrides):
        self.docker.current = {
            "Id": EXISTING,
            "Image": IMAGE,
            "State": {"Running": True},
            "Config": {
                "Labels": {
                    OWNER_LABEL: self.spec.owner,
                    CONFIG_LABEL: self.manager._fingerprint(self.spec, IMAGE),
                }
            },
            **overrides,
        }

    def test_create_uses_immutable_image_mounts_and_injected_runtime(self):
        self.assertEqual(self.manager.ensure(self.spec), CREATED)
        commands = self.docker.calls
        self.assertEqual(
            [c[1] for c in commands], ["image", "container", "create", "start", "exec"]
        )
        create = commands[2]
        self.assertEqual(create[-3:], [IMAGE, "sleep", "infinity"])
        self.assertIn(f"{OWNER_LABEL}=w/c", create)
        self.assertIn("--mount", create)
        self.assertIn("--init", create)
        self.assertEqual(commands[3], ["docker", "start", CREATED])
        self.assertIn(CREATED, commands[-1])
        self.assertNotIn("private upstream output", " ".join(create))

    def test_matching_running_container_is_reused_after_readiness(self):
        self.existing()
        self.assertEqual(self.manager.ensure(self.spec), EXISTING)
        self.assertEqual(
            [c[1] for c in self.docker.calls], ["image", "container", "exec"]
        )

    def test_management_lifecycle_never_inherits_terminal_stdin(self):
        self.manager.ensure(self.spec)
        self.existing()
        self.manager.ensure(self.spec)
        self.manager.remove(self.spec.name, owner=self.spec.owner)
        self.docker.current = None
        self.docker.fail = "exec"
        with self.assertRaisesRegex(RuntimeError, "Docker exec failed"):
            self.manager.ensure(self.spec)
        self.assertEqual(
            {command[1] for command in self.docker.calls},
            {"image", "container", "create", "start", "exec", "rm"},
        )
        self.assertTrue(
            all("-i" in command for command in self.docker.calls if command[1] == "exec")
        )
        for command, options in zip(self.docker.calls, self.docker.options, strict=True):
            with self.subTest(command=command[1]):
                self.assertEqual(options.get("stdin"), subprocess.DEVNULL)

    def test_create_and_exec_use_only_new_gpu_environment_including_empty(self):
        for gpus in ("", "device=2"):
            with self.subTest(gpus=gpus):
                self.docker.calls.clear()
                self.manager.environment = replace(
                    self.manager.environment,
                    container=self.manager.environment.container.model_copy(
                        update={"gpus": gpus}
                    ),
                )
                self.manager.ensure(self.spec)
                for command in self.docker.calls:
                    if command[1] not in ("create", "exec"):
                        continue
                    environment = [
                        command[index + 1]
                        for index, part in enumerate(command)
                        if part == "-e"
                    ]
                    self.assertEqual(
                        environment.count("VIBESIM_RUNNER_GPUS=" + gpus), 1
                    )
                    self.assertFalse(
                        any(
                            item.startswith("CODEX_DOCKER_GPUS=")
                            for item in environment
                        )
                    )
                    if command[1] == "create":
                        self.assertEqual("--gpus" in command, bool(gpus))
                        if gpus:
                            self.assertEqual(command[command.index("--gpus") + 1], gpus)

    def test_image_scope_mount_and_readiness_changes_recreate_owned_container(self):
        for variation in ("image", "scope", "mount", "readiness", "stopped"):
            with self.subTest(variation=variation):
                self.docker = Docker()
                self.manager.run = self.docker
                self.existing()
                spec = self.spec
                if variation == "image":
                    self.docker.image = "sha256:" + "d" * 64
                elif variation == "scope":
                    spec = replace(spec, session_scopes=(("assistant", "new"),))
                elif variation == "mount":
                    spec = replace(
                        spec, mounts=(replace(spec.mounts[0], read_only=True),)
                    )
                elif variation == "readiness":
                    self.docker.fail_readiness = True
                else:
                    self.docker.current["State"]["Running"] = False
                self.assertEqual(self.manager.ensure(spec), CREATED)
                self.assertIn(["docker", "rm", "-f", EXISTING], self.docker.calls)

    def test_foreign_owner_and_daemon_failure_do_not_remove_or_create(self):
        self.existing(Config={"Labels": {OWNER_LABEL: "someone-else"}})
        with self.assertRaisesRegex(RuntimeError, "another owner"):
            self.manager.ensure(self.spec)
        self.assertEqual([c[1] for c in self.docker.calls], ["image", "container"])
        self.docker.calls.clear()
        self.docker.inspect_error = "Cannot connect to the Docker daemon"
        with self.assertRaisesRegex(RuntimeError, "inspection failed"):
            self.manager.ensure(self.spec)
        self.assertEqual([c[1] for c in self.docker.calls], ["image", "container"])

    def test_failed_start_or_bootstrap_removes_only_new_id_and_hides_stderr(self):
        for command in ("start", "exec"):
            with self.subTest(command=command):
                self.docker = Docker()
                self.docker.fail = command
                self.manager.run = self.docker
                with self.assertRaises(RuntimeError) as raised:
                    self.manager.ensure(self.spec)
                self.assertNotIn("private", str(raised.exception))
                self.assertEqual(self.docker.calls[-1], ["docker", "rm", "-f", CREATED])

    def test_missing_source_rejected_before_existing_container_is_removed(self):
        source = self.root / "source"
        source.mkdir()
        spec = replace(self.spec, mounts=(Mount(source, PurePosixPath("/workspace")),))
        self.existing()
        source.rmdir()
        with self.assertRaises(ValueError):
            self.manager.ensure(spec)
        self.assertFalse(any(c[1] == "rm" for c in self.docker.calls))

    def test_cleanup_failure_keeps_original_error_with_a_diagnostic_note(self):
        self.docker.fail = "exec"

        def run(command, **kwargs):
            result = self.docker(command, **kwargs)
            if command[1] == "rm":
                return subprocess.CompletedProcess(
                    command, 1, "", "private cleanup detail"
                )
            return result

        self.manager.run = run
        with self.assertRaisesRegex(RuntimeError, "Docker exec failed") as caught:
            self.manager.ensure(self.spec)
        self.assertEqual(
            caught.exception.__notes__, ["Container cleanup also failed (RuntimeError)"]
        )

    def test_failed_removal_does_not_create_replacement(self):
        self.existing(State={"Running": False})
        self.docker.fail = "rm"
        with self.assertRaisesRegex(RuntimeError, "Docker rm failed"):
            self.manager.ensure(self.spec)
        self.assertFalse(any(command[1] == "create" for command in self.docker.calls))

    def test_create_failure_recovers_only_this_invocations_container(self):
        for error in ("timeout", "invalid", "failure"):
            for matches in (True, False):
                with self.subTest(error=error, matches=matches):
                    self.docker = Docker()
                    self.manager.run = self.docker
                    self.docker.create_error = error
                    self.docker.creation_matches = matches
                    with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                        self.manager.ensure(self.spec)
                    removed = [c for c in self.docker.calls if c[1] == "rm"]
                    self.assertEqual(
                        removed, [["docker", "rm", "-f", CREATED]] if matches else []
                    )

    def test_remove_checks_owner_and_uses_inspected_id(self):
        self.existing()
        with self.assertRaisesRegex(RuntimeError, "another owner"):
            self.manager.remove("test", owner="other")
        self.manager.remove("test", owner="w/c")
        self.assertEqual(self.docker.calls[-1], ["docker", "rm", "-f", EXISTING])

    def test_invalid_names_and_duplicate_mount_targets_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.spec, name="--all")
        with self.assertRaises(ValueError):
            replace(self.spec, mounts=self.spec.mounts * 2)
