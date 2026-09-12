"""Exercise client token selection without starting a server or provider."""

import json
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class ApiSmokeAuthTests(unittest.TestCase):
    def test_new_token_precedence_and_legacy_fallback(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        cases = (
            ({"VIBESIM_AGENT_API_TOKEN": "new"}, "new"),
            ({"VIBESIM_API_TOKEN": "old"}, "old"),
            ({"VIBESIM_AGENT_API_TOKEN": "new", "VIBESIM_API_TOKEN": "old"}, "new"),
            ({"VIBESIM_AGENT_API_TOKEN": "", "VIBESIM_API_TOKEN": "old"}, ""),
            ({}, ""),
        )
        for script in ("agent_api_smoke.sh", "agent_conversation_smoke.sh"):
            for tokens, expected in cases:
                with (
                    self.subTest(script=script, keys=tuple(tokens)),
                    TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    log = root / "requests.jsonl"
                    curl = root / "curl"
                    curl.write_text(
                        f"#!{sys.executable}\n"
                        "import json, os, sys\n"
                        "args = sys.argv[1:]\n"
                        "with open(os.environ['SMOKE_REQUESTS'], 'a') as out:\n"
                        "    out.write(json.dumps(args) + '\\n')\n"
                        "if args[-1].endswith('/tools/skill'):\n"
                        "    print('ServingStudio Agent API')\n"
                        "elif '%{http_code}' in args:\n"
                        "    print('401')\n"
                        "else:\n"
                        "    sys.exit(22)\n"
                    )
                    curl.chmod(0o700)
                    uv = root / "uv"
                    uv.write_text(
                        "#!/bin/sh\nshift 2\nexec "
                        + shlex.quote(sys.executable)
                        + ' "$@"\n'
                    )
                    uv.chmod(0o700)
                    result = subprocess.run(
                        ["bash", str(scripts / script), "http://smoke.invalid"],
                        env={
                            "PATH": str(root) + os.pathsep + os.defpath,
                            "SMOKE_REQUESTS": str(log),
                            **tokens,
                        },
                        text=True,
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 22, result.stderr)
                    requests = [
                        json.loads(line) for line in log.read_text().splitlines()
                    ]
                    self.assertEqual(len(requests), 3 if expected else 2)
                    self.assertNotIn("Authorization:", " ".join(requests[0]))
                    if expected:
                        self.assertNotIn("Authorization:", " ".join(requests[1]))
                        self.assertIn(f"Authorization: Bearer {expected}", requests[-1])
                    else:
                        self.assertNotIn("Authorization:", " ".join(requests[-1]))


if __name__ == "__main__":
    unittest.main()
