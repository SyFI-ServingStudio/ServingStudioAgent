import asyncio
import os
import sys
import unittest
from unittest.mock import patch

from vibesim_agent.runtime.process import ProcessStream


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def stop(self, process):
        self.stopped.append(process.pid)
        process.terminate()
        await process.wait()

    def setUp(self):
        self.stopped = []

    def call(self, code, **options):
        return ProcessStream(
            [sys.executable, "-u", "-c", code],
            stop=self.stop,
            idle_timeout=options.pop("idle_timeout", 2),
            poll_interval=0.02,
            **options,
        )

    async def test_simultaneous_pipes_and_large_input(self):
        payload = b"p" * 200000
        call = self.call(
            "import sys; sys.stderr.write('e'*100000); "
            "data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)",
            input_data=payload,
        )
        stdout = bytearray()
        async with call:
            async for event in call.events():
                if event.stream == "stdout":
                    stdout.extend(event.data)
        self.assertEqual(stdout, payload)
        self.assertEqual(call.stderr_tail, b"e" * 8192)
        self.assertEqual(call.process.returncode, 0)
        self.assertFalse(self.stopped)

    async def test_timeout_after_stdout_eof_stops_live_process(self):
        call = self.call(
            "import os,time; os.close(1); os.close(2); time.sleep(30)",
            idle_timeout=0.15,
        )
        async with call:
            async for _ in call.events():
                pass
        self.assertTrue(call.timed_out)
        self.assertEqual(self.stopped, [call.process.pid])
        self.assertIsNotNone(call.process.returncode)

    async def test_early_consumer_exit_cleans_up(self):
        call = self.call("import time; print('ready'); time.sleep(30)")
        async with call:
            async for event in call.events():
                if event.stream == "stdout":
                    break
        self.assertEqual(self.stopped, [call.process.pid])
        self.assertFalse(call._tasks)

    async def test_cancel_during_spawn_recovers_and_stops_handle(self):
        spawned = asyncio.Event()
        release = asyncio.Event()
        original = asyncio.create_subprocess_exec

        async def delayed(*args, **kwargs):
            process = await original(*args, **kwargs)
            spawned.set()
            await release.wait()
            return process

        call = self.call("import time; time.sleep(30)")

        async def consume():
            async with call:
                async for _ in call.events():
                    pass

        with patch("asyncio.create_subprocess_exec", delayed):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(spawned.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        self.assertEqual(self.stopped, [call.process.pid])
        self.assertIsNotNone(call.process.returncode)

    async def test_stop_failure_still_reaps_process(self):
        async def broken_stop(process):
            raise RuntimeError("remote stop failed")

        call = self.call("import time; time.sleep(30)")
        call.stop = broken_stop
        with self.assertRaisesRegex(RuntimeError, "remote stop failed"):
            async with call:
                pass
        self.assertIsNotNone(call.process.returncode)

    async def test_stderr_chatter_does_not_extend_idle_deadline(self):
        call = self.call(
            "import os,time; "
            "exec('while True:\\n os.write(2, b\"warning\")\\n time.sleep(.002)')",
            idle_timeout=0.15,
        )
        ticks = 0
        async with asyncio.timeout(3):
            async with call:
                async for event in call.events():
                    ticks += event.stream == "tick"
        self.assertTrue(call.timed_out)
        self.assertGreater(ticks, 0)

    async def test_cleanup_drains_full_pipes_without_consumer(self):
        call = self.call(
            "import os; "
            'exec(\'while True:\\n os.write(1, b"x"*65536)\\n os.write(2, b"e"*65536)\')'
        )
        async with asyncio.timeout(3):
            async with call:
                await asyncio.sleep(0.1)
        self.assertIsNotNone(call.process.returncode)
        self.assertEqual(self.stopped, [call.process.pid])

    async def test_exited_process_still_drains_buffered_output(self):
        call = self.call(
            "import os; os.write(1, b'x'*65536); os.write(2, b'e'*65536)",
            stop_timeout=0.3,
        )
        async with asyncio.timeout(3):
            async with call:
                while call.process.returncode is None:
                    await asyncio.sleep(0.01)
                self.assertFalse(call.process.stdout.at_eof())
                self.assertFalse(call.process.stderr.at_eof())
        self.assertTrue(call.process.stdout.at_eof())
        self.assertTrue(call.process.stderr.at_eof())
        self.assertFalse(self.stopped)

    @unittest.skipUnless(sys.platform == "linux", "requires Linux /proc descriptors")
    async def test_exited_process_with_retained_write_ends_closes_local_pipes(self):
        call = self.call(
            "import sys; print('ready'); sys.stdin.buffer.read()", stop_timeout=0.1
        )
        retained = []
        try:
            async with asyncio.timeout(3):
                async with call:
                    await call.process.stdout.readline()
                    # Duplicate the write ends as an inherited descendant would,
                    # but keep ownership in this test so every FD is closed.
                    for descriptor in (1, 2):
                        retained.append(os.open(
                            f"/proc/{call.process.pid}/fd/{descriptor}", os.O_WRONLY
                        ))
                    pipes = [call.process._transport.get_pipe_transport(fd) for fd in (1, 2)]
                    call.process.stdin.close()
                    while call.process.returncode is None:
                        await asyncio.sleep(0.01)
                    self.assertTrue(all(not pipe.is_closing() for pipe in pipes))
                    started = asyncio.get_running_loop().time()
                self.assertLess(asyncio.get_running_loop().time() - started, 1)
                self.assertTrue(all(pipe.is_closing() for pipe in pipes))
                self.assertFalse(call._tasks)
                self.assertFalse(self.stopped)
        finally:
            for descriptor in retained:
                os.close(descriptor)

    @unittest.skipUnless(sys.platform == "linux", "requires Linux /proc descriptors")
    async def test_stop_error_survives_retained_pipe_cleanup_timeout(self):
        async def broken_stop(process):
            raise RuntimeError("remote stop failed")

        call = self.call("import sys; print('ready'); sys.stdin.buffer.read()", stop_timeout=0.1)
        call.stop = broken_stop
        retained = []
        try:
            async with asyncio.timeout(3):
                with self.assertRaisesRegex(RuntimeError, "remote stop failed"):
                    async with call:
                        await call.process.stdout.readline()
                        for descriptor in (1, 2):
                            retained.append(os.open(
                                f"/proc/{call.process.pid}/fd/{descriptor}", os.O_WRONLY
                            ))
                self.assertIsNotNone(call.process.returncode)
                self.assertTrue(call.process._transport.is_closing())
        finally:
            for descriptor in retained:
                os.close(descriptor)
