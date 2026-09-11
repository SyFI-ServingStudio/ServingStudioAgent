import copy
import json
import os
import signal
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from tools import startup_selection
from tools import tmux_deployment as deployment
from tools.migrate_v1_database import MigrationError


class TmuxDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.actual_containers = deployment._containers
        self.actual_panes = deployment._panes
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "state"
        self.source.mkdir()
        self.receipt = self.root / "stopped.json"
        self.target = self.root / "migrated"
        self.descriptors = {}
        self.script = self.root / "serve.sh"
        self.script.write_text("exec server\n")
        self.environment = {"DOCKER_HOST": "unix:///var/run/docker.sock"}
        self.socket = {"path": str(self.root / "tmux.sock"), "device": 1, "inode": 2}
        self.live = True
        self.processes = {
            pid: {
                "pid": pid,
                "uid": os.getuid(),
                "session": pid,
                "start": pid + 1,
                "parent": 1,
                "state": "S",
            }
            for pid in (100, 200)
        }
        self.panes = [
            {
                "name": "backend",
                "session": "$1",
                "pane": "%1",
                "process": deployment._identity(self.processes[200]),
                "command": f"bash {self.script}",
            }
        ]
        self.container_id = "a" * 64
        self.containers = {
            self.container_id: {
                "name": "/fixture",
                "image": "sha256:fixture",
                "mounts": [{"Type": "bind", "Source": str(self.source)}],
                "restart": {"Name": "always"},
                "running": True,
                "restarting": False,
                "paused": True,
            }
        }
        self.commands = []
        self.timeouts = []
        self.timeline = []
        self.after_kill = None
        stack = self.enterContext(ExitStack())
        stack.enter_context(
            patch.object(
                deployment, "_descriptors", side_effect=lambda _: self.descriptors
            )
        )
        stack.enter_context(
            patch.object(
                startup_selection,
                "_descriptors",
                side_effect=lambda _: self.descriptors,
            )
        )
        stack.enter_context(
            patch.object(
                deployment,
                "_socket",
                side_effect=lambda _: copy.deepcopy(self.socket) if self.live else None,
            )
        )
        stack.enter_context(
            patch.object(
                deployment, "_panes", side_effect=lambda *_: copy.deepcopy(self.panes)
            )
        )
        stack.enter_context(
            patch.object(
                deployment,
                "_process",
                side_effect=lambda pid: copy.deepcopy(self.processes.get(pid)),
            )
        )
        stack.enter_context(
            patch.object(
                deployment,
                "_processes",
                side_effect=lambda: copy.deepcopy(self.processes),
            )
        )
        stack.enter_context(
            patch.object(deployment, "_containers", side_effect=self.read_containers)
        )
        stack.enter_context(
            patch.object(deployment, "_run", side_effect=self.run_command)
        )
        self.report = deployment.capture_deployment(
            self.socket["path"],
            self.source,
            {"backend": self.script},
            environment=self.environment,
        )
        self.commands.clear()
        self.timeline.clear()

    def read_containers(self, *_):
        self.timeline.append("containers")
        return copy.deepcopy(self.containers)

    def run_command(self, arguments, _environment, *, timeout=30):
        if "display-message" in arguments:
            return "100\n"
        self.commands.append(list(arguments))
        self.timeouts.append(timeout)
        if "kill-server" in arguments:
            self.timeline.append("kill-host")
            self.live = False
            self.processes.clear()
            if self.after_kill:
                self.after_kill()
        elif arguments[:3] == ["docker", "container", "update"]:
            self.containers[arguments[-1]]["restart"] = {"Name": "no"}
        elif arguments[:3] == ["docker", "container", "unpause"]:
            self.containers[arguments[-1]]["paused"] = False
        elif arguments[:3] == ["docker", "container", "stop"]:
            self.containers[arguments[-1]]["running"] = False
            self.containers[arguments[-1]]["restarting"] = False
        else:
            raise AssertionError(arguments)
        return ""

    def owner(self):
        return deployment.TmuxDeployment(
            self.report, environment=self.environment, grace_seconds=0.1
        )

    def test_pane_format_forces_utf8_in_a_locale_free_environment(self):
        output = f"backend\t$1\t%1\t200\tbash {self.script}\n"
        with patch.object(deployment, "_run", return_value=output) as command:
            panes = self.actual_panes(self.socket["path"], self.environment)
        self.assertEqual(panes, self.panes)
        self.assertEqual(command.call_args.args[0][:2], ["tmux", "-u"])

    def receipt_owner(self, **overrides):
        options = {
            "environment": self.environment,
            "grace_seconds": 0.1,
            "receipt": self.receipt,
            "target": self.target,
        }
        options.update(overrides)
        return deployment.TmuxDeployment(self.report, **options)

    def test_receipt_published_private_and_new_owner_ignores_reused_process_ids(self):
        with self.receipt_owner().quiesce(self.source):
            self.assertTrue(self.receipt.is_file())
            self.assertEqual(self.receipt.stat().st_mode & 0o777, 0o600)
        raw = self.receipt.read_bytes()
        record = json.loads(raw)
        self.assertEqual(record["target"], str(self.target))
        self.assertEqual(
            record["boot_id"],
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        )
        self.assertEqual(len(record["deployment_sha256"]), 64)
        self.processes[200] = {
            "pid": 200,
            "uid": os.getuid(),
            "session": 200,
            "start": 999999,
            "parent": 1,
            "state": "S",
        }
        self.commands.clear()
        owner = self.receipt_owner()
        with (
            patch.object(
                owner,
                "_members",
                side_effect=AssertionError("retired process identities"),
            ) as members,
            patch.object(owner, "_signal") as sent,
            patch.object(
                owner, "_stop_host", side_effect=AssertionError("must not stop again")
            ) as stop,
            owner.quiesce(self.source),
        ):
            pass
        members.assert_not_called()
        sent.assert_not_called()
        stop.assert_not_called()
        self.assertEqual(self.commands, [])
        self.assertEqual(self.receipt.read_bytes(), raw)

    def test_invalid_receipt_never_falls_back_to_stopping_again(self):
        with self.receipt_owner().quiesce(self.source):
            pass
        valid = json.loads(self.receipt.read_bytes())
        self.commands.clear()
        for field in ("corrupt", "boot_id", "target", "deployment_sha256"):
            record = valid | {field: "different"}
            self.receipt.write_text(
                "not json" if field == "corrupt" else json.dumps(record)
            )
            owner = self.receipt_owner()
            with (
                self.subTest(field=field),
                patch.object(owner, "_stop_host") as stop,
                patch.object(owner, "_members") as members,
                self.assertRaises(MigrationError),
                owner.quiesce(self.source),
            ):
                self.fail("invalid receipt yielded")
            stop.assert_not_called()
            members.assert_not_called()
            self.assertEqual(self.commands, [])

    def test_receipt_binding_rejects_changed_configured_target_and_report(self):
        with self.receipt_owner().quiesce(self.source):
            pass
        self.commands.clear()
        original = copy.deepcopy(self.report)
        for changed in ("target", "report"):
            self.report = copy.deepcopy(original)
            options = {}
            if changed == "target":
                options["target"] = self.root / "other-target"
            else:
                self.report["server"]["start"] += 1
            with (
                self.subTest(changed=changed),
                self.assertRaises(MigrationError),
                self.receipt_owner(**options).quiesce(self.source),
            ):
                self.fail("binding mismatch yielded")
            self.assertEqual(self.commands, [])

    def test_receipt_reuse_refuses_restored_socket_or_container_without_mutations(self):
        with self.receipt_owner().quiesce(self.source):
            pass
        stopped = copy.deepcopy(self.containers)
        self.commands.clear()
        for restored in (
            "socket",
            "running",
            "paused",
            "restarting",
            "policy",
            "new-container",
        ):
            self.containers = copy.deepcopy(stopped)
            self.live = restored == "socket"
            if restored in ("running", "paused", "restarting"):
                self.containers[self.container_id][restored] = True
            elif restored == "policy":
                self.containers[self.container_id]["restart"]["Name"] = "always"
            elif restored == "new-container":
                self.containers["b" * 64] = copy.deepcopy(stopped[self.container_id])
            with (
                self.subTest(restored=restored),
                self.assertRaises(MigrationError),
                self.receipt_owner().quiesce(self.source),
            ):
                self.fail("restored writer yielded")
            self.assertEqual(self.commands, [])

    def test_receipt_publication_failure_does_not_yield_or_restart_stopped_services(
        self,
    ):
        with (
            patch.object(deployment.os, "link", side_effect=OSError("publish failed")),
            self.assertRaisesRegex(OSError, "publish failed"),
            self.receipt_owner().quiesce(self.source),
        ):
            self.fail("unpublished receipt yielded")
        self.assertFalse(self.receipt.exists())
        self.assertFalse(self.live)
        self.assertFalse(self.containers[self.container_id]["running"])
        self.assertEqual(self.containers[self.container_id]["restart"]["Name"], "no")
        self.assertEqual(list(self.root.glob(".agent-shutdown-*")), [])
        self.assertEqual(
            [command[2] for command in self.commands[1:]], ["update", "unpause", "stop"]
        )

    def test_receipt_path_rejects_state_roots_external_repository_and_symlink(self):
        self.target.mkdir()
        external = self.root / "external"
        external.mkdir()
        self.descriptors = {
            "w_main": {
                "storage_kind": "external",
                "repo_path": str(external),
                "logs_path": str(external),
            }
        }
        alias = self.root / "alias.json"
        alias.symlink_to(self.root / "missing")
        for path in (
            self.source / "receipt.json",
            self.target / "receipt.json",
            external / "receipt.json",
            alias,
            Path("relative.json"),
        ):
            with self.subTest(path=path), self.assertRaises(MigrationError):
                self.receipt_owner(receipt=path)
        for target in (
            self.source,
            self.source / "nested",
            self.root,
            Path("relative"),
        ):
            with self.subTest(target=target), self.assertRaises(MigrationError):
                self.receipt_owner(target=target)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.receipt.exists())

    def test_receipt_fdopen_failure_closes_raw_descriptor_and_cleans_temporary(self):
        descriptors = []
        original = tempfile.mkstemp

        def tracked_mkstemp(*args, **kwargs):
            descriptor, path = original(*args, **kwargs)
            descriptors.append(descriptor)
            return descriptor, path

        with (
            patch.object(deployment.tempfile, "mkstemp", side_effect=tracked_mkstemp),
            patch.object(deployment.os, "fdopen", side_effect=OSError("fdopen failed")),
            self.assertRaisesRegex(OSError, "fdopen failed"),
            self.receipt_owner().quiesce(self.source),
        ):
            self.fail("failed publication yielded")
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError) as error:
            os.fstat(descriptors[0])
        self.assertEqual(error.exception.errno, 9)
        self.assertFalse(self.receipt.exists())
        self.assertEqual(list(self.root.glob(".agent-shutdown-*")), [])
        self.assertFalse(self.live)
        self.assertFalse(self.containers[self.container_id]["running"])

    def test_only_explicit_local_daemon_is_accepted_before_any_commands(self):
        for environment in (
            {},
            {"DOCKER_HOST": "tcp://elsewhere:2375"},
            self.environment | {"DOCKER_CONTEXT": "other"},
            self.environment | {"DOCKER_TLS_VERIFY": "1"},
            self.environment | {"DOCKER_CERT_PATH": "/other"},
        ):
            with (
                self.subTest(environment=environment),
                self.assertRaises(MigrationError),
            ):
                deployment.capture_deployment(
                    self.socket["path"],
                    self.source,
                    {"backend": self.script},
                    environment=environment,
                )
        self.assertEqual(self.commands, [])

    def test_capture_rejects_undeclared_pane_or_wrong_script_without_mutations(self):
        original = copy.deepcopy(self.panes)
        for panes in (
            original + [original[0] | {"name": "unknown"}],
            [original[0] | {"command": "bash /another/script"}],
        ):
            self.panes = panes
            with self.assertRaises(MigrationError):
                deployment.capture_deployment(
                    self.socket["path"],
                    self.source,
                    {"backend": self.script},
                    environment=self.environment,
                )
        self.assertEqual(self.commands, [])

    def test_source_script_socket_and_process_identity_refuse_before_mutation(self):
        cases = ("source", "script", "socket", "pane", "server", "process")
        for case in cases:
            with self.subTest(case=case):
                report = copy.deepcopy(self.report)
                if case == "source":
                    report["source"]["inode"] += 1
                elif case == "script":
                    report["scripts"][str(self.script)] = "incorrect"
                elif case == "socket":
                    report["socket"]["inode"] += 1
                elif case == "pane":
                    report["panes"][0]["command"] = "different"
                elif case == "server":
                    report["server"]["start"] += 1
                else:
                    report["panes"][0]["process"]["start"] += 1
                owner = deployment.TmuxDeployment(report, environment=self.environment)
                with self.assertRaises(MigrationError), owner.quiesce(self.source):
                    self.fail("invalid identity yielded")
                self.assertEqual(self.commands, [])

    def test_container_addition_or_mount_change_refuses_before_stop(self):
        original = copy.deepcopy(self.containers)
        for kind in ("new", "changed"):
            self.containers = copy.deepcopy(original)
            if kind == "new":
                self.containers["b" * 64] = copy.deepcopy(original[self.container_id])
            else:
                self.containers[self.container_id]["mounts"] = []
            with (
                self.subTest(kind=kind),
                self.assertRaises(MigrationError),
                self.owner().quiesce(self.source),
            ):
                self.fail("changed container yielded")
            self.assertEqual(self.commands, [])

    def test_container_created_during_host_shutdown_is_detected_after_host_stop(self):
        self.after_kill = lambda: self.containers.update(
            {"b" * 64: copy.deepcopy(self.containers[self.container_id])}
        )
        with (
            self.assertRaisesRegex(MigrationError, "containers changed"),
            self.owner().quiesce(self.source),
        ):
            self.fail("new writer cannot be silently stopped or ignored")
        self.assertEqual(self.timeline, ["containers", "kill-host", "containers"])
        self.assertEqual(len(self.commands), 1)
        self.assertIn("kill-server", self.commands[0])

    def test_stop_order_retains_container_and_body_failure_never_restarts(self):
        with (
            self.assertRaisesRegex(RuntimeError, "application failed"),
            self.owner().quiesce(self.source),
        ):
            self.assertFalse(self.live)
            item = self.containers[self.container_id]
            self.assertFalse(item["running"])
            self.assertFalse(item["paused"])
            self.assertEqual(item["restart"]["Name"], "no")
            raise RuntimeError("application failed")
        self.assertEqual(
            [command[2] for command in self.commands[1:]], ["update", "unpause", "stop"]
        )
        self.assertIn("kill-server", self.commands[0])
        self.assertEqual(set(self.containers), {self.container_id})
        self.assertFalse(self.live)

    def test_foreign_uid_or_daemonized_child_refuses_before_mutations(self):
        for case in ("uid", "session"):
            self.processes[201] = self.processes[200] | {
                "pid": 201,
                "parent": 200,
                "uid": os.getuid() + 1 if case == "uid" else os.getuid(),
                "session": 201 if case == "session" else 200,
            }
            with (
                self.subTest(case=case),
                self.assertRaises(MigrationError),
                self.owner().quiesce(self.source),
            ):
                self.fail("foreign or detached process yielded")
            self.assertEqual(self.commands, [])

    def test_pidfd_revalidation_never_signals_a_reused_pid_and_always_closes(self):
        expected = copy.deepcopy(self.processes[200])
        self.processes[200]["start"] += 1
        with (
            patch.object(os, "pidfd_open", return_value=9876) as opened,
            patch.object(os, "close") as closed,
            patch.object(signal, "pidfd_send_signal") as sent,
            self.assertRaisesRegex(MigrationError, "identity changed"),
        ):
            deployment.TmuxDeployment._signal(expected, signal.SIGTERM)
        opened.assert_called_once_with(200)
        sent.assert_not_called()
        closed.assert_called_once_with(9876)

    def test_valid_process_is_signaled_through_pidfd_not_numeric_kill(self):
        with (
            patch.object(os, "pidfd_open", return_value=9876),
            patch.object(os, "close") as closed,
            patch.object(os, "kill") as numeric_kill,
            patch.object(signal, "pidfd_send_signal") as sent,
        ):
            deployment.TmuxDeployment._signal(self.processes[200], signal.SIGTERM)
        sent.assert_called_once_with(9876, signal.SIGTERM)
        numeric_kill.assert_not_called()
        closed.assert_called_once_with(9876)

    def test_each_signal_phase_signals_once_and_finds_late_members(self):
        owner = self.owner()
        owner.grace_seconds = 1
        self.live = False
        first = {200: self.processes[200]}
        both = first | {201: self.processes[200] | {"pid": 201, "start": 202}}
        with (
            patch.object(
                owner,
                "_members",
                side_effect=[first, first, both, both, both, {}, {}, {}],
            ),
            patch.object(owner, "_signal") as sent,
            patch.object(
                deployment.time, "monotonic", side_effect=[0, 0.2, 0.4, 2, 3, 3.2]
            ),
            patch.object(deployment.time, "sleep"),
        ):
            owner._stop_host()
        self.assertEqual(
            [(call.args[0]["pid"], call.args[1]) for call in sent.call_args_list],
            [
                (200, signal.SIGTERM),
                (201, signal.SIGTERM),
                (200, signal.SIGKILL),
                (201, signal.SIGKILL),
            ],
        )

    def test_docker_stop_timeout_includes_the_full_grace_period(self):
        owner = deployment.TmuxDeployment(
            self.report, environment=self.environment, grace_seconds=60
        )
        with owner.quiesce(self.source):
            pass
        stop_index = next(
            index for index, command in enumerate(self.commands) if "stop" in command
        )
        self.assertGreater(self.timeouts[stop_index], 60)
        self.assertEqual(self.commands[stop_index][-2], "60")

    def test_broader_writable_source_ancestor_is_not_adopted_as_owned_container(self):
        cases = [
            ("running", True, False, False, "no", False),
            ("paused", False, True, False, "no", False),
            ("restarting", False, False, True, "no", False),
            ("stopped-no", False, False, False, "no", True),
            ("stopped-unless", False, False, False, "unless-stopped", True),
            ("stopped-always", False, False, False, "always", False),
            ("stopped-on-failure", False, False, False, "on-failure", False),
            ("stopped-unknown", False, False, False, "future-policy", False),
            ("stopped-missing-policy", False, False, False, None, False),
            ("missing-running", False, False, False, "no", False),
            ("missing-paused", False, False, False, "no", False),
            ("missing-restarting", False, False, False, "no", False),
        ]
        for label, running, paused, restarting, policy, allowed in cases:
            container = {
                "Id": "b" * 64,
                "Mounts": [{"Type": "bind", "Source": str(self.root), "RW": True}],
                "State": {
                    "Running": running,
                    "Paused": paused,
                    "Restarting": restarting,
                },
                "HostConfig": {"RestartPolicy": {"Name": policy}},
            }
            if label.startswith("missing-"):
                container["State"].pop(label.removeprefix("missing-").capitalize())
            with (
                self.subTest(label=label),
                patch.object(
                    deployment, "_run", side_effect=["b" * 64, json.dumps([container])]
                ) as commands,
            ):
                if allowed:
                    self.assertEqual(
                        self.actual_containers(self.source, self.environment), {}
                    )
                else:
                    with self.assertRaisesRegex(
                        MigrationError,
                        "state is incomplete"
                        if label.startswith("missing-")
                        else "broader writable access",
                    ):
                        self.actual_containers(self.source, self.environment)
            self.assertEqual(
                [call.args[0][2] for call in commands.call_args_list], ["ls", "inspect"]
            )

    @contextmanager
    def external_fixture(self):
        external = {
            "Id": "b" * 64,
            "Name": "/external-service",
            "Image": "sha256:" + "e" * 64,
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": str(self.root),
                    "Destination": "/broad",
                    "RW": True,
                }
            ],
            "State": {"Running": True, "Paused": False, "Restarting": False},
            "HostConfig": {"RestartPolicy": {"Name": "always"}},
        }
        calls = []

        def run(arguments, environment, *, timeout=30):
            calls.append(arguments)
            owned = self.containers[self.container_id]
            items = [
                external,
                {
                    "Id": self.container_id,
                    "Name": owned["name"],
                    "Image": owned["image"],
                    "Mounts": [
                        {"Type": "bind", "Source": str(self.source), "RW": True}
                    ],
                    "State": {
                        "Running": owned["running"],
                        "Paused": owned["paused"],
                        "Restarting": owned["restarting"],
                    },
                    "HostConfig": {"RestartPolicy": owned["restart"]},
                },
            ]
            if arguments[:3] == ["docker", "container", "ls"]:
                return "\n".join(item["Id"] for item in items)
            if arguments[:3] == ["docker", "container", "inspect"]:
                return json.dumps(
                    [item for item in items if item["Id"] in arguments[3:]]
                )
            return self.run_command(arguments, environment, timeout=timeout)

        with (
            patch.object(deployment, "_containers", self.actual_containers),
            patch.object(deployment, "_run", run),
        ):
            self.report = deployment.capture_deployment(
                self.socket["path"],
                self.source,
                {"backend": self.script},
                environment=self.environment,
                external_containers=(external["Id"],),
            )
            yield external, calls

    def test_explicit_external_container_is_audited_but_never_stopped(self):
        with self.external_fixture() as (external, calls):
            self.assertEqual(
                self.report["external_containers"],
                {external["Id"]: deployment._external_identity(external)},
            )
            self.assertNotIn(external["Id"], self.report["containers"])
            with self.receipt_owner().quiesce(self.source):
                self.assertTrue(external["State"]["Running"])
            before = list(self.commands)
            with self.receipt_owner().quiesce(self.source):
                pass
            self.assertEqual(self.commands, before)
            self.assertTrue(external["State"]["Running"])
            self.assertEqual(external["HostConfig"]["RestartPolicy"]["Name"], "always")
            self.assertFalse(
                any(external["Id"] in command for command in self.commands)
            )
            self.assertGreaterEqual(
                sum(c[:3] == ["docker", "container", "ls"] for c in calls), 5
            )
            external["Image"] = "sha256:" + "f" * 64
            with self.assertRaisesRegex(MigrationError, "external container identity"):
                with self.receipt_owner().quiesce(self.source):
                    self.fail("changed external image reached yield")
            self.assertEqual(self.commands, before)

    def test_external_change_during_host_shutdown_blocks_container_mutations(self):
        with self.external_fixture() as (external, _):
            self.after_kill = lambda: external.update(Name="/changed-external")
            with self.assertRaisesRegex(MigrationError, "external container identity"):
                with self.owner().quiesce(self.source):
                    self.fail("changed external identity reached yield")
            self.assertFalse(self.live)
            self.assertFalse(any(command[0] == "docker" for command in self.commands))

    def test_external_mount_order_is_ignored_but_all_mount_fields_are_preserved(self):
        with self.external_fixture() as (external, _):
            external["Mounts"].append(
                {
                    "Type": "bind",
                    "Source": str(self.root / "data"),
                    "Destination": "/data",
                    "RW": False,
                }
            )
            original = deployment._external_identity(external)
            original["mounts"].reverse()
            evidence = {external["Id"]: original}
            before = copy.deepcopy(evidence)
            external["Mounts"].reverse()
            self.actual_containers(self.source, self.environment, evidence)
            self.assertEqual(evidence, before)
            for change in ("source", "destination", "rw", "duplicate"):
                altered = copy.deepcopy(external)
                if change == "duplicate":
                    altered["Mounts"].append(copy.deepcopy(altered["Mounts"][0]))
                else:
                    key, value = {
                        "source": ("Source", str(self.root / "other")),
                        "destination": ("Destination", "/changed"),
                        "rw": ("RW", True),
                    }[change]
                    altered["Mounts"][0][key] = value
                with (
                    self.subTest(change=change),
                    patch.object(
                        deployment,
                        "_run",
                        side_effect=[altered["Id"], json.dumps([altered])],
                    ),
                ):
                    with self.assertRaisesRegex(
                        MigrationError, "external container identity"
                    ):
                        self.actual_containers(self.source, self.environment, evidence)

    def test_owned_mount_order_is_ignored_without_mutating_existing_report(self):
        mounts = [
            {"Type": "bind", "Source": str(self.source), "RW": True},
            {"Type": "bind", "Source": str(self.root / "data"), "RW": False},
        ]
        self.report["containers"][self.container_id]["mounts"] = copy.deepcopy(mounts)
        self.containers[self.container_id]["mounts"] = list(reversed(mounts))
        before = copy.deepcopy(self.report)
        self.owner()._check_containers()
        self.assertEqual(self.report, before)
        self.containers[self.container_id]["mounts"][0]["RW"] = True
        with self.assertRaisesRegex(MigrationError, "container identity or mounts"):
            self.owner()._check_containers()

    def test_external_exceptions_require_unchanged_broad_only_mounts_and_exact_ids(
        self,
    ):
        with self.external_fixture() as (external, _):
            original = copy.deepcopy(external)
            evidence = self.report["external_containers"]
        for variation in (
            "name",
            "image",
            "mount",
            "direct",
            "nested",
            "missing",
            "unknown",
        ):
            items = [copy.deepcopy(original)]
            if variation == "name":
                items[0]["Name"] = "/renamed"
            elif variation == "image":
                items[0]["Image"] = "sha256:" + "f" * 64
            elif variation == "mount":
                items[0]["Mounts"][0]["Destination"] = "/different"
            elif variation in {"direct", "nested"}:
                items[0]["Mounts"].append(
                    {
                        "Type": "bind",
                        "RW": False,
                        "Source": str(
                            self.source
                            if variation == "direct"
                            else self.source / "child"
                        ),
                    }
                )
            elif variation == "missing":
                items = []
            else:
                extra = copy.deepcopy(original)
                extra["Id"] = "c" * 64
                items.append(extra)
            outputs = ["\n".join(item["Id"] for item in items)]
            if items:
                outputs.append(json.dumps(items))
            with (
                self.subTest(variation=variation),
                patch.object(deployment, "_run", side_effect=outputs) as commands,
            ):
                with self.assertRaises(MigrationError):
                    self.actual_containers(self.source, self.environment, evidence)
                self.assertTrue(
                    all(
                        call.args[0][2] in {"ls", "inspect"}
                        for call in commands.call_args_list
                    )
                )
        for identities in ("b" * 64, ["short"], ["b" * 64, "b" * 64], [None]):
            with (
                self.subTest(identities=identities),
                patch.object(deployment, "_run") as commands,
            ):
                with self.assertRaisesRegex(MigrationError, "unique full"):
                    deployment._capture_external(identities, self.environment)
                commands.assert_not_called()
