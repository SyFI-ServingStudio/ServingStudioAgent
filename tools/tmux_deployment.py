"""Stop one explicitly audited, non-daemonizing legacy tmux deployment.

This adapter owns a dedicated tmux socket and its declared service sessions.
It assumes no external supervisor or operator restarts those services. It does
not discover authority from a port or container name, or stop arbitrary writers.
"""

import hashlib
import json
import os
import re
import shlex
import signal
import socket as sockets
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from tools.migrate_v1_database import MigrationError
from tools.migrate_v1_workspaces import _descriptors
from tools.startup_selection import _location, _outside_repositories, _read, _root


def _run(arguments, environment, *, timeout=30):
    return subprocess.run(
        arguments,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    ).stdout


def _process(pid):
    directory = Path("/proc") / str(pid)
    try:
        fields = (directory / "stat").read_text().rpartition(")")[2].split()
        return {
            "pid": pid,
            "uid": directory.stat().st_uid,
            "parent": int(fields[1]),
            "session": int(fields[3]),
            "start": int(fields[19]),
            "state": fields[0],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def _processes():
    return {
        int(path.name): process
        for path in Path("/proc").iterdir()
        if path.name.isdigit() and (process := _process(int(path.name))) is not None
    }


def _identity(process):
    return {key: process[key] for key in ("pid", "uid", "session", "start")}


def _socket(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise MigrationError("deployment socket must be an absolute real socket")
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise MigrationError("deployment socket is not owned by this user")
    with sockets.socket(sockets.AF_UNIX) as connection:
        connection.settimeout(2)
        try:
            connection.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            return None
    return {"path": str(path), "device": info.st_dev, "inode": info.st_ino}


def _panes(socket, environment):
    output = _run(
        [
            "tmux",
            "-u",
            "-S",
            str(socket),
            "list-panes",
            "-a",
            "-F",
            "#{session_name}\t#{session_id}\t#{pane_id}\t#{pane_pid}\t#{pane_start_command}",
        ],
        environment,
    )
    result = []
    for line in output.splitlines():
        name, session, pane, pid, command = line.split("\t", 4)
        process = _process(int(pid))
        if (
            process is None
            or process["uid"] != os.getuid()
            or process["session"] != int(pid)
        ):
            raise MigrationError("tmux pane lacks a live owned process session")
        result.append(
            {
                "name": name,
                "session": session,
                "pane": pane,
                "process": _identity(process),
                "command": command,
            }
        )
    return sorted(result, key=lambda item: item["name"])


def _containers(source, environment):
    identities = _run(
        ["docker", "container", "ls", "-aq", "--no-trunc"], environment
    ).split()
    result = {}
    for offset in range(0, len(identities), 32):
        items = json.loads(
            _run(
                ["docker", "container", "inspect", *identities[offset : offset + 32]],
                environment,
            )
        )
        for item in items:
            mounts = item["Mounts"]
            if any(
                type(item["State"].get(key)) is not bool
                for key in ("Running", "Paused", "Restarting")
            ):
                raise MigrationError("container state is incomplete; audit again")
            may_write = (
                item["State"]["Running"]
                or item["State"]["Paused"]
                or item["State"]["Restarting"]
                or item["HostConfig"].get("RestartPolicy", {}).get("Name")
                not in {"no", "unless-stopped"}
            )
            if may_write and any(
                mount["Type"] == "bind"
                and mount["RW"]
                and source != Path(mount["Source"]).resolve()
                and source.is_relative_to(Path(mount["Source"]).resolve())
                for mount in mounts
            ):
                raise MigrationError(
                    "container has broader writable access to source; audit its owner separately"
                )
            if not any(
                mount["Type"] == "bind"
                and Path(mount["Source"]).resolve().is_relative_to(source)
                for mount in mounts
            ):
                continue
            result[item["Id"]] = {
                "name": item["Name"],
                "image": item["Image"],
                "mounts": mounts,
                "restart": item["HostConfig"]["RestartPolicy"],
                "running": item["State"]["Running"],
                "paused": item["State"]["Paused"],
                "restarting": item["State"]["Restarting"],
            }
    return result


def _environment(environment):
    result = dict(os.environ if environment is None else environment)
    if result.get("DOCKER_HOST") != "unix:///var/run/docker.sock" or any(
        result.get(key)
        for key in ("DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
    ):
        raise MigrationError("deployment requires the explicit local Docker daemon")
    return result


def capture_deployment(socket, source, scripts, *, environment=None):
    """Read-only evidence for review; does not authorize stopping this deployment."""
    environment = _environment(environment)
    source_identity = _root(source)
    source = Path(source_identity["path"])
    socket_identity = _socket(socket)
    if socket_identity is None:
        raise MigrationError("audited tmux deployment must be running during capture")
    panes = _panes(socket, environment)
    if len(panes) != len(scripts) or {pane["name"] for pane in panes} != set(scripts):
        raise MigrationError("dedicated tmux server contains undeclared services")
    files = {}
    for pane in panes:
        path = Path(scripts[pane["name"]])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise MigrationError("deployment script must be an absolute real file")
        words = shlex.split(pane["command"])
        if len(words) == 1:
            words = shlex.split(words[0])
        if words != ["bash", str(path)]:
            raise MigrationError(
                "tmux command differs from the declared service script"
            )
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    server = _process(
        int(
            _run(
                ["tmux", "-S", str(socket), "display-message", "-p", "#{pid}"],
                environment,
            )
        )
    )
    if server is None or server["uid"] != os.getuid():
        raise MigrationError("tmux server identity unavailable")
    report = {
        "format": 1,
        "source": source_identity,
        "socket": socket_identity,
        "server": _identity(server),
        "panes": panes,
        "scripts": files,
        "containers": _containers(source, environment),
    }
    if _socket(socket) != socket_identity or _panes(socket, environment) != panes:
        raise MigrationError("deployment changed while its evidence was captured")
    return report


class TmuxDeployment:
    """Explicit cancellation of the audited legacy stack; never restarts it.

    Service scripts must retain their descendants in each pane's process session
    and have no external restart supervisor. The source-state owner must also
    exclude independent host writers before using this as a migration callback.
    Docker containers are stopped and retained, including their writable layers.
    """

    def __init__(
        self, report, *, environment=None, grace_seconds=10, receipt=None, target=None
    ):
        self.report = json.loads(json.dumps(report))
        self.environment = _environment(environment)
        self.grace_seconds = grace_seconds
        self.receipt = None
        self.target = None
        if (
            report.get("format") != 1
            or not report.get("panes")
            or not 0 < grace_seconds <= 60
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in report["containers"]
            )
        ):
            raise MigrationError("invalid audited deployment evidence")
        if receipt is not None:
            path = Path(receipt)
            source = Path(report["source"]["path"])
            if (
                target is None
                or not Path(target).is_absolute()
                or Path(target).is_symlink()
            ):
                raise MigrationError(
                    "shutdown receipt requires an absolute migration target"
                )
            target = Path(target)
            self.target = target.parent.resolve(strict=True) / target.name
            if not path.is_absolute() or path.is_symlink():
                raise MigrationError("shutdown receipt requires an absolute real file")
            path = _location(path, source, self.target)
            _outside_repositories(path, source, {"descriptors": _descriptors(source)})
            self.receipt = path

    def _receipt_record(self):
        return {
            "format": 1,
            "target": str(self.target),
            "deployment_sha256": hashlib.sha256(
                json.dumps(self.report, sort_keys=True).encode()
            ).hexdigest(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }

    def _has_receipt(self):
        if self.receipt is None:
            return False
        try:
            _, record = _read(self.receipt)
        except FileNotFoundError:
            return False
        if record != self._receipt_record() or type(record.get("format")) is not int:
            raise MigrationError("shutdown receipt or host boot changed; audit again")
        return True

    def _publish_receipt(self):
        if self.receipt is None:
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=".agent-shutdown-", dir=self.receipt.parent
        )
        try:
            try:
                stream = os.fdopen(descriptor, "w")
            except BaseException:
                os.close(descriptor)
                raise
            with stream:
                json.dump(self._receipt_record(), stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, self.receipt)
            parent = os.open(self.receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _check_stopped(self):
        current = self._check_containers()
        if any(
            item["running"]
            or item["paused"]
            or item["restarting"]
            or item["restart"]["Name"] != "no"
            for item in current.values()
        ):
            raise MigrationError("legacy containers did not stop with restart disabled")
        if _socket(self.report["socket"]["path"]) is not None:
            raise MigrationError("legacy deployment restarted after shutdown")

    def _check_containers(self):
        current = _containers(Path(self.report["source"]["path"]), self.environment)
        if set(current) != set(self.report["containers"]):
            raise MigrationError("source-mounted containers changed; audit again")
        for identity, item in current.items():
            original = self.report["containers"][identity]
            if any(item[key] != original[key] for key in ("name", "image", "mounts")):
                raise MigrationError("audited container identity or mounts changed")
        return current

    def _members(self):
        processes = _processes()
        sessions = {pane["process"]["session"] for pane in self.report["panes"]}
        members = {
            pid: process
            for pid, process in processes.items()
            if process["session"] in sessions and process["state"] != "Z"
        }
        for pane in self.report["panes"]:
            expected = pane["process"]
            leader = processes.get(expected["pid"])
            if leader is not None and _identity(leader) != expected:
                raise MigrationError("audited process session identity changed")
        if any(process["uid"] != os.getuid() for process in members.values()):
            raise MigrationError(
                "deployment process session contains another user's process"
            )
        if any(
            process["parent"] in members and process["session"] not in sessions
            for process in processes.values()
        ):
            raise MigrationError(
                "daemonized service children require a different deployment owner"
            )
        return members

    @staticmethod
    def _signal(process, value):
        try:
            descriptor = os.pidfd_open(process["pid"])
        except ProcessLookupError:
            return
        try:
            current = _process(process["pid"])
            if current is None:
                return
            if _identity(current) != _identity(process):
                raise MigrationError("process identity changed before cancellation")
            try:
                signal.pidfd_send_signal(descriptor, value)
            except ProcessLookupError:
                pass
        finally:
            os.close(descriptor)

    def _stop_host(self):
        path = self.report["socket"]["path"]
        current = _socket(path)
        if current is not None:
            if (
                current != self.report["socket"]
                or _panes(path, self.environment) != self.report["panes"]
            ):
                raise MigrationError("tmux deployment identity changed")
            server = _process(self.report["server"]["pid"])
            if server is None or _identity(server) != self.report["server"]:
                raise MigrationError("tmux server process identity changed")
            _run(["tmux", "-S", path, "kill-server"], self.environment)
        for value in (signal.SIGTERM, signal.SIGKILL):
            deadline = time.monotonic() + self.grace_seconds
            sent = set()
            while members := self._members():
                for process in members.values():
                    identity = (process["pid"], process["start"])
                    if identity not in sent:
                        self._signal(process, value)
                        sent.add(identity)
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            if not self._members():
                break
        if self._members() or _socket(path) is not None:
            raise MigrationError("legacy host processes did not stop")

    @contextmanager
    def quiesce(self, source):
        if _root(source) != self.report["source"]:
            raise MigrationError("deployment source identity changed")
        for name, digest in self.report["scripts"].items():
            path = Path(name)
            if (
                path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise MigrationError("audited service script changed")
        if self._has_receipt():
            # Completed shutdown retires the old process identities. Reusing a
            # numeric PID later does not make its new owner part of this stack.
            self._check_stopped()
            yield
            return
        self._members()
        self._check_containers()
        self._stop_host()
        current = self._check_containers()
        for identity, item in current.items():
            _run(
                ["docker", "container", "update", "--restart=no", identity],
                self.environment,
            )
            if item["paused"]:
                _run(["docker", "container", "unpause", identity], self.environment)
            if item["running"] or item["restarting"]:
                _run(
                    [
                        "docker",
                        "container",
                        "stop",
                        "--time",
                        str(int(self.grace_seconds)),
                        identity,
                    ],
                    self.environment,
                    timeout=self.grace_seconds + 30,
                )
        self._check_stopped()
        if self._members() or _socket(self.report["socket"]["path"]) is not None:
            raise MigrationError("legacy deployment restarted during shutdown")
        self._publish_receipt()
        yield
