"""Launch a backend with explicit credentials or a simple local Claude wrapper.

Parse literal shell definitions only: never source startup files or execute a
wrapper while discovering credentials. No discovered values are printed.
"""

import os
import re
import shlex
import sys
from pathlib import Path

AUTH = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
ALLOWED = (*AUTH, "ANTHROPIC_BASE_URL")
VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z_0-9]*)\}|([A-Za-z_][A-Za-z_0-9]*))")


def wrapper_environment(body: str, environment: dict[str, str]) -> dict[str, str]:
    try:
        words = shlex.split(body.replace("\\\n", ""), comments=True)
    except ValueError:
        return {}
    found = {}
    if words and words[0] == "env":
        words.pop(0)
    while words and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", words[0]):
        name, value = words.pop(0).split("=", 1)
        # Restrict discovery to literal assignments and inherited variables.
        if "`" in value or "$(" in value:
            return {}
        unresolved = False

        def expand(match):
            nonlocal unresolved
            key = match[1] or match[2]
            if key not in environment:
                unresolved = True
            return environment.get(key, "")

        value = VARIABLE.sub(expand, value)
        if unresolved or "$" in value:
            return {}
        if name in ALLOWED:
            found[name] = value
    if words and words[0] == "command":
        words.pop(0)
    if not words or Path(words.pop(0)).name != "claude":
        return {}
    if words not in ([], ["$@"], ["$@", ";"]):
        return {}
    return found


def discover(environment: dict[str, str], home: Path) -> dict[str, str]:
    # A configured provider is a unit: never mix its credentials with a shell
    # wrapper's endpoint or an alternative authentication mechanism.
    result = dict(environment)
    if any(environment.get(key) for key in AUTH):
        return result
    candidates = []
    for name in (".bash_profile", ".profile", ".bashrc", ".bash_aliases", ".zshrc"):
        try:
            source = (home / name).read_text()
        except (OSError, UnicodeError):
            continue
        source = source.replace("\\\n", "")
        definitions = []
        for match in re.finditer(
            r"(?m)^\s*(?:function\s+claude(?:\s*\(\s*\))?|claude\s*\(\s*\))\s*\{([^{}]*(?:\$\{[^{}]*\}[^{}]*)*)\}",
            source,
        ):
            definitions.append((match.start(), match[1].strip()))
        for match in re.finditer(r"(?m)^\s*alias\s+claude=(.+)$", source):
            try:
                words = shlex.split(match[1], comments=True)
            except ValueError:
                continue
            if len(words) == 1:
                definitions.append((match.start(), words[0]))
        for _, body in sorted(definitions):
            candidates.append(wrapper_environment(body, environment))
    for candidate in reversed(candidates):
        if not any(candidate.get(key) for key in AUTH):
            continue
        if environment.get("ANTHROPIC_BASE_URL") and environment[
            "ANTHROPIC_BASE_URL"
        ] != candidate.get("ANTHROPIC_BASE_URL"):
            continue
        for key, value in candidate.items():
            if not result.get(key):
                result[key] = value
        return result
    return result


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: with_claude_env.py COMMAND [ARG ...]")
    environment = dict(os.environ)
    if environment.get("CLAUDE_DISCOVER_SHELL_ENV", "1") != "0":
        environment = discover(environment, Path.home())
    os.execvpe(sys.argv[1], sys.argv[1:], environment)


if __name__ == "__main__":
    main()
