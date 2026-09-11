"""Transport fixtures shared by old and new HTTP contract tests."""

import asyncio
import json


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        kind = "message"
        data = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            events.append((kind, json.loads("\n".join(data))))
    return events


class LiveRequest:
    """An ASGI connection with explicit disconnect and unbuffered response frames."""

    def __init__(self, app, path: str, *, method="GET", body=None):
        self.messages = asyncio.Queue()
        self.disconnected = asyncio.Event()
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {
                    "type": "http.request",
                    "body": json.dumps(body or {}).encode(),
                    "more_body": False,
                }
            await self.disconnected.wait()
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "server": ("test", 80),
            "client": ("test", 1234),
        }
        self.task = asyncio.create_task(app(scope, receive, self.messages.put))

    async def next(self):
        return await asyncio.wait_for(self.messages.get(), timeout=5)

    async def close(self):
        self.disconnected.set()
        await asyncio.wait_for(self.task, timeout=5)
