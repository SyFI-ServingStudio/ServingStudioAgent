"""Bounded pipe streaming with adapter-owned remote process interruption."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessOutput:
    stream: str
    data: bytes = b""


StopProcess = Callable[[asyncio.subprocess.Process], Awaitable[None]]


class ProcessStream:
    """Use as an async context manager and iterate once inside its lifetime.

    Tick events let an adapter poll its durable log. Only actual output (or an
    explicit touch after finding durable activity) resets the idle deadline.
    Stderr warnings do not extend the deadline.
    The stop callback must stop the remote call when running through Docker.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        input_data: bytes = b"",
        idle_timeout: float,
        stop: StopProcess,
        poll_interval: float = 5,
        stop_timeout: float = 20,
        environment: Mapping[str, str] | None = None,
    ):
        for value in (idle_timeout, poll_interval, stop_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("process timeouts must be finite and positive")
        if not command:
            raise ValueError("process command is required")
        self.command = tuple(command)
        self.input_data = input_data
        self.idle_timeout = idle_timeout
        self.poll_interval = poll_interval
        self.stop_timeout = stop_timeout
        self.stop = stop
        self.environment = dict(environment) if environment is not None else None
        self.process: asyncio.subprocess.Process | None = None
        self.timed_out = False
        self.stderr_tail = bytearray()
        self._tasks: set[asyncio.Task] = set()
        self._entered = False
        self._iterated = False
        self._deadline = 0.0

    def touch(self) -> None:
        self._deadline = asyncio.get_running_loop().time() + self.idle_timeout

    async def __aenter__(self) -> ProcessStream:
        if self._entered:
            raise RuntimeError("process stream cannot be reused")
        self._entered = True
        startup = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.environment,
            )
        )
        try:
            self.process = await asyncio.shield(startup)
        except asyncio.CancelledError:
            # Cancellation must not lose the process handle while spawn finishes.
            async def finish_startup():
                self.process = await startup
                await self._close()

            await self._shield_cleanup(finish_startup())
            raise
        self.touch()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._shield_cleanup(self._close())

    @staticmethod
    async def _shield_cleanup(operation: Awaitable[None]) -> None:
        task = asyncio.create_task(operation)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _write(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(self.input_data)
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.process.stdin.close()

    async def _close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self.process is None:
            return

        async def drain(pipe):
            if pipe is not None:
                while await pipe.read(8192):
                    pass

        # A paused full pipe prevents asyncio's process wait from completing.
        # Cleanup owns both readers after the streaming tasks have stopped.
        drains = [
            asyncio.create_task(drain(pipe))
            for pipe in (self.process.stdout, self.process.stderr)
        ]
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(self.stop(self.process), self.stop_timeout)
                finally:
                    if self.process.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            self.process.kill()
                    try:
                        await asyncio.wait_for(self.process.wait(), self.stop_timeout)
                    except asyncio.TimeoutError:
                        self._close_local_transport()
                        await asyncio.wait_for(self.process.wait(), self.stop_timeout)
        finally:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*drains), self.stop_timeout
                )
            except asyncio.TimeoutError:
                self._close_local_transport()
            finally:
                for task in drains:
                    task.cancel()
                await asyncio.gather(*drains, return_exceptions=True)

    def _close_local_transport(self) -> None:
        assert self.process is not None
        # asyncio Process has no public close. A descendant can retain pipe write
        # ends after returncode is set, so bounded cleanup must close local FDs.
        self.process._transport.close()

    async def events(self) -> AsyncIterator[ProcessOutput]:
        if self.process is None or self._iterated:
            raise RuntimeError("enter process context and consume its stream once")
        self._iterated = True
        assert self.process.stdout is not None and self.process.stderr is not None
        writer = asyncio.create_task(self._write())
        waiter = asyncio.create_task(self.process.wait())
        readers = {
            asyncio.create_task(self.process.stdout.read(8192)): "stdout",
            asyncio.create_task(self.process.stderr.read(8192)): "stderr",
        }
        self._tasks.update((writer, waiter, *readers))
        next_tick = asyncio.get_running_loop().time() + self.poll_interval
        while readers or not waiter.done() or not writer.done():
            if asyncio.get_running_loop().time() >= next_tick:
                yield ProcessOutput("tick")
                next_tick = asyncio.get_running_loop().time() + self.poll_interval
            remaining = self._deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self.timed_out = True
                return
            pending = set(readers)
            pending.update(task for task in (writer, waiter) if not task.done())
            done, _ = await asyncio.wait(
                pending,
                timeout=min(
                    remaining, max(0, next_tick - asyncio.get_running_loop().time())
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task not in readers:
                    task.result()
                    continue
                stream = readers.pop(task)
                self._tasks.discard(task)
                chunk = task.result()
                if not chunk:
                    continue
                if stream == "stdout":
                    self.touch()
                if stream == "stderr":
                    self.stderr_tail.extend(chunk)
                    del self.stderr_tail[:-8192]
                yield ProcessOutput(stream, chunk)
                pipe = (
                    self.process.stdout if stream == "stdout" else self.process.stderr
                )
                next_read = asyncio.create_task(pipe.read(8192))
                readers[next_read] = stream
                self._tasks.add(next_read)
        writer.result()
        waiter.result()
