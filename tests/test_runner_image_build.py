"""Image build orchestration with real Git copying and a fake Docker boundary."""

import hashlib
import os
import shlex
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.provider_fixture import write_minimal_providers
from tools import runner_image
from tools.runner_image import BUILD_OPTIONS
from vibesim_agent.bootstrap import configuration


class RunnerImageBuildTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.agent = self.root / "agent source"
        source = Path(__file__).resolve().parents[1]
        for name in (
            "docker/runner.Dockerfile",
            "scripts/lib/main-tree-copy.sh",
            "scripts/test-runner-image.sh",
        ):
            target = self.agent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        write_minimal_providers(self.agent)
        self.main = self.root / "custom main source"
        self.main.mkdir()
        self.environment = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root / "host home"),
            "USER": "builduser",
            "TMPDIR": str(self.root),
            "VIBESIM_AGENT_MAIN_DIR": str(self.main),
        }
        self.git("init", "--template=", "--initial-branch=main")
        for name, value in (
            ("uv.lock", "version = 1\n"),
            ("tracked", "initial\n"),
            ("deleted", "remove\n"),
        ):
            (self.main / name).write_text(value)
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "initial",
        )
        (self.main / "tracked").write_text("dirty current bytes\n")
        (self.main / "deleted").unlink()
        (self.main / "untracked").write_text("excluded\n")
        self.calls = []
        self.contexts = []
        self.failure = None

    def git(self, *args):
        environment = {
            **self.environment,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        return subprocess.run(
            ["git", "-C", str(self.main), *args],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def run_process(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        is_smoke = any(str(arg).endswith("test-runner-image.sh") for arg in args)
        is_docker = args[0] == "docker"
        is_copy = args[0] == "bash" and not is_smoke
        if is_docker:
            self.assertEqual(args[1], "build")
            context = Path(args[-1])
            self.contexts.append(context)
            self.assertTrue(context.is_dir())
            self.assertEqual(
                (context / "vibesim/tracked").read_text(), "dirty current bytes\n"
            )
            self.assertFalse((context / "vibesim/untracked").exists())
            self.assertFalse((context / "vibesim/deleted").exists())
        if self.failure and (
            (self.failure == "docker" and is_docker)
            or (self.failure == "smoke" and is_smoke)
            or (self.failure == "copy" and is_copy)
        ):
            raise subprocess.CalledProcessError(17, args)
        if is_docker or is_smoke:
            return subprocess.CompletedProcess(args, 0, "", "")
        self.assertIn(args[0], ("git", "bash"), "unrecognized real subprocess")
        self.assertTrue(kwargs.pop("check"))
        return subprocess.run(args, check=True, **kwargs)

    def build(self):
        return runner_image.build(
            environment=self.environment, repo_root=self.agent, run=self.run_process
        )

    def source_bytes(self):
        return {
            str(path.relative_to(self.main)): path.read_bytes()
            for path in self.main.rglob("*")
            if path.is_file()
        }

    def test_default_settings_build_arguments_and_smoke_share_identity_and_source(self):
        expected = configuration(
            environment=self.environment, repo_root=self.agent
        ).container
        before = self.source_bytes()
        host_environment = dict(os.environ)
        self.build()
        docker, _ = next(call for call in self.calls if call[0][0] == "docker")
        arguments = {
            docker[index + 1].split("=", 1)[0]: docker[index + 1].split("=", 1)[1]
            for index, item in enumerate(docker)
            if item == "--build-arg"
        }
        self.assertEqual(docker[docker.index("-t") + 1], expected.image)
        for key, value in {
            "APP_UID": str(expected.uid),
            "APP_GID": str(expected.gid),
            "APP_USER": expected.user,
            "RUNNER_VERSION": expected.version,
            "VIBESIM_LOCK_SHA": hashlib.sha256(
                (self.main / "uv.lock").read_bytes()
            ).hexdigest(),
        }.items():
            self.assertEqual(arguments[key], value)
        smoke, options = next(
            call
            for call in self.calls
            if any(str(arg).endswith("test-runner-image.sh") for arg in call[0])
        )
        self.assertIn("build", smoke)
        self.assertEqual(smoke[smoke.index("--main-dir") + 1], str(self.main))
        self.assertFalse(any(key.startswith("CODEX_DOCKER_") for key in options["env"]))
        self.assertEqual(
            configuration(environment=options["env"], repo_root=self.agent).container,
            expected,
        )
        self.assertEqual(self.source_bytes(), before)
        self.assertEqual(dict(os.environ), host_environment)
        self.assertTrue(all(not path.exists() for path in self.contexts))

    def test_nondefault_build_options_and_skip_are_explicit(self):
        self.environment.update(
            {
                "VIBESIM_RUNNER_IMAGE": "fixture:custom",
                "VIBESIM_RUNNER_UID": "1234",
                "VIBESIM_RUNNER_GID": "2345",
                "VIBESIM_RUNNER_SKIP_IMAGE_TEST": "1",
            }
        )
        options = {
            "CUDA_IMAGE": "cuda:fixture",
            "UV_IMAGE": "uv:fixture",
            "CODEX_NPM_PACKAGE": "codex@fixture",
            "CLAUDE_NPM_PACKAGE": "claude@fixture",
            "NODE_VERSION": "v20.fixture",
            "NODE_ARCH": "linux-arm64",
            "RUST_TOOLCHAIN": "fixture-toolchain",
        }
        self.environment.update(
            {"VIBESIM_RUNNER_" + key: value for key, value in options.items()}
        )
        self.build()
        docker = next(args for args, _ in self.calls if args[0] == "docker")
        for key, value in options.items():
            self.assertIn(key + "=" + value, docker)
        self.assertIn("APP_UID=1234", docker)
        self.assertIn("APP_GID=2345", docker)
        self.assertFalse(
            any(
                any(str(arg).endswith("test-runner-image.sh") for arg in args)
                for args, _ in self.calls
            )
        )

    def test_incompatible_baked_paths_and_retired_keys_refuse_before_subprocess(self):
        for key, value in (
            ("VIBESIM_RUNNER_HOME", "/other/home"),
            ("VIBESIM_RUNNER_UV_PROJECT_ENVIRONMENT", "/other/venv"),
            ("VIBESIM_RUNNER_UV_CACHE_DIR", "/other/cache"),
            ("VIBESIM_RUNNER_DG_USE_LOCAL_VERSION", "true"),
            ("CODEX_DOCKER_IMAGE", "old-secret-value"),
        ):
            with self.subTest(key=key):
                self.environment[key] = value
                with self.assertRaises(ValueError) as caught:
                    self.build()
                self.assertNotIn("old-secret-value", str(caught.exception))
                self.assertEqual(self.calls, [])
                del self.environment[key]

    def test_hostile_git_environment_cannot_redirect_copy(self):
        foreign = self.root / "foreign-index"
        foreign.write_bytes(b"preserve")
        self.environment.update(
            {
                "GIT_DIR": str(self.root / "missing"),
                "GIT_WORK_TREE": str(self.root),
                "GIT_INDEX_FILE": str(foreign),
                "GIT_CONFIG_GLOBAL": str(self.root / "bad-config"),
            }
        )
        self.build()
        self.assertEqual(foreign.read_bytes(), b"preserve")
        for _, options in self.calls:
            environment = options["env"]
            self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
            self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
            self.assertNotIn("GIT_DIR", environment)
            self.assertNotIn("GIT_INDEX_FILE", environment)

    def test_copy_docker_and_smoke_failure_propagate_and_remove_temporary_context(self):
        before = self.source_bytes()
        children = set(self.root.iterdir())
        for failure in ("copy", "docker", "smoke"):
            with self.subTest(failure=failure):
                self.failure = failure
                self.calls.clear()
                with self.assertRaises(subprocess.CalledProcessError):
                    self.build()
                self.assertEqual(set(self.root.iterdir()), children)
                self.assertEqual(self.source_bytes(), before)
                if failure == "copy":
                    self.assertFalse(any(args[0] == "docker" for args, _ in self.calls))
                if failure == "docker":
                    self.assertFalse(
                        any(
                            any(
                                str(arg).endswith("test-runner-image.sh")
                                for arg in args
                            )
                            for args, _ in self.calls
                        )
                    )

    def test_source_must_be_git_top_level_with_lock(self):
        original = self.environment["VIBESIM_AGENT_MAIN_DIR"]
        for path in (self.root, self.main / "nested"):
            path.mkdir(exist_ok=True)
            self.environment["VIBESIM_AGENT_MAIN_DIR"] = str(path)
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                self.build()
            self.assertFalse(any(args[0] == "docker" for args, _ in self.calls))
        self.environment["VIBESIM_AGENT_MAIN_DIR"] = original
        (self.main / "uv.lock").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.build()
        self.assertFalse(any(args[0] == "docker" for args, _ in self.calls))

    def test_real_copy_helper_git_failures_never_reach_docker(self):
        module = self.main / "vendor/module"
        module.mkdir(parents=True)
        (module / ".git").mkdir()
        (self.main / ".gitmodules").write_text(
            '[submodule "fixture"]\n path = vendor/module\n url = /fixture\n'
        )
        self.git("add", ".gitmodules")
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        marker = self.root / "git-failed"
        actual_git = shutil.which("git")
        fake_git = fake_bin / "git"
        fake_git.write_text(
            "#!/bin/bash\n"
            'original=("$@"); repo=""\n'
            'while [ "$#" -gt 0 ]; do case "$1" in -c) shift 2;; -C) repo="$2"; shift 2;; *) break;; esac; done\n'
            'if { [ "$1" = "config" ] && [ "${COPY_FAILURE}" = "config" ]; } || '
            '{ [ "$1" = "ls-files" ] && [ "$repo" = "${COPY_MODULE}" ] && [ "${COPY_FAILURE}" = "submodule" ]; }; then\n'
            '  printf injected > "$COPY_MARKER"\n  exit 9\nfi\n'
            f'exec {shlex.quote(actual_git)} "${{original[@]}}"\n'
        )
        fake_git.chmod(0o755)
        self.environment.update(
            PATH=str(fake_bin) + os.pathsep + self.environment["PATH"],
            COPY_MODULE=str(module),
            COPY_MARKER=str(marker),
        )
        for failure in ("config", "submodule"):
            with self.subTest(failure=failure):
                self.calls.clear()
                self.environment["COPY_FAILURE"] = failure
                marker.unlink(missing_ok=True)
                before = self.source_bytes()
                with self.assertRaises(subprocess.CalledProcessError):
                    self.build()
                self.assertEqual(marker.read_text(), "injected")
                self.assertFalse(any(args[0] == "docker" for args, _ in self.calls))
                self.assertEqual(self.source_bytes(), before)
                self.assertEqual(list(self.root.glob("vibesim-runner-build-*")), [])

    def test_initialized_submodule_with_spaces_copies_current_tracked_bytes(self):
        self.git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(self.main),
            "vendor/space module",
        )
        nested = self.main / "vendor/space module"
        (nested / "tracked").write_text("submodule dirty bytes\n")
        (nested / "untracked").write_text("skip nested untracked\n")
        before = self.source_bytes()
        copied = []

        def inspect(args, **kwargs):
            if args[0] == "docker":
                target = Path(args[-1]) / "vibesim/vendor/space module"
                self.assertEqual(
                    (target / "tracked").read_text(), "submodule dirty bytes\n"
                )
                self.assertFalse((target / "untracked").exists())
                copied.append(target)
            return self.run_process(args, **kwargs)

        runner_image.build(
            environment=self.environment, repo_root=self.agent, run=inspect
        )
        self.assertEqual(len(copied), 1)
        self.assertEqual(self.source_bytes(), before)


class RunnerImagePinTests(unittest.TestCase):
    def test_build_defaults_match_the_dockerfile_args(self):
        """`BUILD_OPTIONS` is passed as `--build-arg`, so it wins over the ARG.

        The two therefore have to agree, and nothing else would notice if they
        stopped: the image would simply be built with a different CLI than the
        Dockerfile documents.
        """
        dockerfile = (
            Path(__file__).parents[1] / "docker/runner.Dockerfile"
        ).read_text()
        declared = dict(
            line.removeprefix("ARG ").split("=", 1)
            for line in dockerfile.splitlines()
            if line.startswith("ARG ") and "=" in line
        )
        for name, default in BUILD_OPTIONS.items():
            with self.subTest(option=name):
                self.assertEqual(declared.get(name), default)
