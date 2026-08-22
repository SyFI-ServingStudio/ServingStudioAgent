from __future__ import annotations

import unittest

from backend.app import list_codex_backends
from backend.codex_runtime.config import ALL_CODEX_ROLES


class CodexBackendCatalogTest(unittest.TestCase):
    def test_defaults_cover_every_role_not_just_the_default_mode(self) -> None:
        """The picker offers `agent_mode` before the first message, so it needs a
        default for `assistant` too — a role the default mode never runs."""
        defaults = list_codex_backends()["defaults"]

        self.assertEqual(sorted(defaults), sorted(ALL_CODEX_ROLES))
        for role, runtime in defaults.items():
            self.assertEqual(
                sorted(runtime), ["effort", "model", "serviceTier"], msg=role
            )


if __name__ == "__main__":
    unittest.main()
