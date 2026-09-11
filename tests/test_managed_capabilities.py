"""Capability expiry, concurrent authority and private atomic context publication."""

import json
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import AgentRequest, Model, Selection
from vibesim_agent.services.capabilities import (
    CapabilityRegistry,
    ManagedContext,
    bearer_token,
)


class ManagedCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.now = 1000.0
        self.registry = CapabilityRegistry(clock=lambda: self.now)
        self.path = self.root / "control" / "managed-run.json"
        self.context = ManagedContext(
            self.registry, "http://backend:8765/", lambda w, c: self.path
        )
        model = Model("model", "Model", ("high",), "high")
        self.request = AgentRequest(
            "w",
            "c",
            "t",
            Role.ASSISTANT,
            "prompt",
            "container",
            Selection("provider", model, "high", "default", "scope"),
        )

    def issue(self, **overrides):
        arguments = {
            "workspace_id": "w",
            "conversation_id": "c",
            "turn_id": "t",
            "role": "assistant",
        }
        arguments.update(overrides)
        return self.registry.issue(**arguments)

    def test_issue_authorize_expiry_boundary_and_restart_invalidation(self):
        capability = self.issue(ttl_seconds=10)
        self.assertEqual(capability.expires_at, 1010)
        self.assertEqual(self.registry.authorize(capability.token), capability)
        self.assertIsNone(self.registry.authorize("unknown"))
        self.assertNotIn(capability.token, repr(capability))
        self.now = 1009.99
        self.assertIsNotNone(self.registry.authorize(capability.token))
        self.now = 1010
        self.assertIsNone(self.registry.authorize(capability.token))
        next_token = self.issue()
        restarted = CapabilityRegistry(clock=lambda: self.now)
        self.assertIsNone(restarted.authorize(next_token.token))

    def test_turn_revocation_is_workspace_scoped_and_includes_all_roles(self):
        first = self.issue()
        second = self.issue(role="implementer")
        other_workspace = self.issue(workspace_id="other")
        other_turn = self.issue(turn_id="other")
        self.registry.revoke_turn("w", "t")
        self.registry.revoke_turn("w", "t")
        self.assertIsNone(self.registry.authorize(first.token))
        self.assertIsNone(self.registry.authorize(second.token))
        self.assertEqual(
            self.registry.authorize(other_workspace.token), other_workspace
        )
        self.assertEqual(self.registry.authorize(other_turn.token), other_turn)

    def test_invalid_lifetime_cannot_create_nonexpiring_capability(self):
        for ttl in (0, -1, float("inf"), float("nan")):
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                self.issue(ttl_seconds=ttl)
        self.now = float("nan")
        with self.assertRaises(ValueError):
            self.issue()

    def test_concurrent_issue_authorize_and_revocation_keep_tokens_isolated(self):
        def worker(index):
            capability = self.issue(turn_id=str(index % 2))
            self.assertEqual(self.registry.authorize(capability.token), capability)
            return capability

        with ThreadPoolExecutor(max_workers=8) as pool:
            capabilities = list(pool.map(worker, range(100)))
            self.assertEqual(len({c.token for c in capabilities}), 100)
            list(pool.map(lambda _: self.registry.revoke_turn("w", "0"), range(8)))
            authorized = list(
                pool.map(lambda c: self.registry.authorize(c.token), capabilities)
            )
        for capability, result in zip(capabilities, authorized, strict=True):
            self.assertEqual(result, None if capability.turn_id == "0" else capability)

    def test_context_wire_shape_permissions_and_turn_lifetime(self):
        capability = self.context.write(self.request)
        self.assertEqual(
            json.loads(self.path.read_text()),
            {
                "schema_version": 1,
                "managed_jobs_api": "agent-v1",
                "backend_url": "http://backend:8765",
                "capability_token": capability.token,
                "expires_at": 8200,
            },
        )
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(self.registry.authorize(capability.token), capability)
        later = self.context.write(replace(self.request, role=Role.IMPLEMENTER))
        self.assertEqual(self.registry.authorize(capability.token), capability)
        self.context.remove("w", "c")
        self.context.remove("w", "c")
        self.assertFalse(self.path.exists())
        self.assertIsNotNone(self.registry.authorize(later.token))
        self.registry.revoke_turn("w", "t")
        self.assertIsNone(self.registry.authorize(later.token))

    def test_replace_failure_revokes_new_token_preserves_old_file_and_cleans_private_temp(
        self,
    ):
        old = self.context.write(self.request)
        before = self.path.read_bytes()
        new = self.issue()

        def failed_replace(temporary, target):
            self.assertEqual(stat.S_IMODE(temporary.stat().st_mode), 0o600)
            self.assertEqual(self.path.read_bytes(), before)
            raise OSError("replace failed")

        with (
            patch.object(self.registry, "issue", return_value=new),
            patch.object(Path, "replace", failed_replace),
            self.assertRaisesRegex(OSError, "replace failed"),
        ):
            self.context.write(self.request)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertIsNone(self.registry.authorize(new.token))
        self.assertEqual(self.registry.authorize(old.token), old)

    def test_write_and_path_failure_revoke_new_token_without_publishing(self):
        for method in ("write", "path"):
            new = self.issue()
            target = "json.dump" if method == "write" else None
            manager = (
                self.context
                if target
                else ManagedContext(
                    self.registry, "http://backend", lambda w, c: Path("relative")
                )
            )
            with patch.object(self.registry, "issue", return_value=new):
                if target:
                    with (
                        patch(target, side_effect=OSError("disk full")),
                        self.assertRaises(OSError),
                    ):
                        manager.write(self.request)
                else:
                    with self.assertRaises(ValueError):
                        manager.write(self.request)
            self.assertIsNone(self.registry.authorize(new.token))
            self.assertFalse(self.path.exists())
            self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_write_and_remove_serialize_atomic_publication(self):
        entered, release = threading.Event(), threading.Event()
        original = Path.replace

        def blocked_replace(temporary, target):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test publication was not released")
            return original(temporary, target)

        with (
            ThreadPoolExecutor(max_workers=2) as pool,
            patch.object(Path, "replace", blocked_replace),
        ):
            writer = pool.submit(self.context.write, self.request)
            try:
                self.assertTrue(entered.wait(2))
                remover = pool.submit(self.context.remove, "w", "c")
                self.assertFalse(remover.done())
            finally:
                release.set()
            writer.result(timeout=2)
            remover.result(timeout=2)
        self.assertFalse(self.path.exists())

    def test_bearer_parsing_matches_legacy_case_and_whitespace_rules(self):
        for header in (None, "", "bearer token", "Bearer\ttoken", "Basic token"):
            self.assertEqual(bearer_token(header), "")
        self.assertEqual(bearer_token("Bearer  token \t"), "token")
        self.assertEqual(bearer_token("Bearer "), "")

    def test_stream_open_failure_closes_raw_descriptor_and_revokes_token(self):
        original = tempfile.mkstemp
        descriptors = []

        def tracked_tempfile(**kwargs):
            descriptor, name = original(**kwargs)
            descriptors.append(descriptor)
            return descriptor, name

        capability = self.issue()
        with (
            patch.object(self.registry, "issue", return_value=capability),
            patch("tempfile.mkstemp", tracked_tempfile),
            patch("os.fdopen", side_effect=OSError("open failed")),
            self.assertRaisesRegex(OSError, "open failed"),
        ):
            self.context.write(self.request)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.assertIsNone(self.registry.authorize(capability.token))
        self.assertEqual(list(self.root.rglob("*.tmp")), [])
