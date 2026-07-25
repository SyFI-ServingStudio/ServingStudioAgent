"""Forward a localhost TCP listener to the Docker-bridge VibeSim UI endpoint.

The production UI binds to the bridge address so conversation containers can
reach it.  This process adds a loopback-only browser endpoint without starting
a second FastAPI process or duplicating conversation state.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress


async def _copy_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    while data := await reader.read(64 * 1024):
        writer.write(data)
        await writer.drain()


async def _forward_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            upstream_host, upstream_port
        )
    except OSError:
        client_writer.close()
        await client_writer.wait_closed()
        return

    copy_tasks = {
        asyncio.create_task(_copy_stream(client_reader, upstream_writer)),
        asyncio.create_task(_copy_stream(upstream_reader, client_writer)),
    }
    done_tasks, pending_tasks = await asyncio.wait(
        copy_tasks, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending_tasks:
        task.cancel()
    for task in done_tasks | pending_tasks:
        with suppress(asyncio.CancelledError, ConnectionError, OSError):
            await task

    upstream_writer.close()
    client_writer.close()
    with suppress(ConnectionError, OSError):
        await upstream_writer.wait_closed()
    with suppress(ConnectionError, OSError):
        await client_writer.wait_closed()


async def _serve(arguments: argparse.Namespace) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: _forward_connection(
            reader,
            writer,
            arguments.upstream_host,
            arguments.upstream_port,
        ),
        arguments.listen_host,
        arguments.listen_port,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8765)
    parser.add_argument("--upstream-host", default="172.19.0.1")
    parser.add_argument("--upstream-port", type=int, default=8765)
    asyncio.run(_serve(parser.parse_args()))


if __name__ == "__main__":
    main()
