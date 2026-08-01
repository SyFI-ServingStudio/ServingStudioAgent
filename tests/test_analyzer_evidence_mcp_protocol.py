from __future__ import annotations

import asyncio

import anyio
from mcp import ClientSession

from backend.analyzer_evidence_mcp.server import mcp


def test_mcp_session_exposes_and_dispatches_the_analyzer_tool() -> None:
    """Exercise the real MCP session and dispatcher without calling the tool directly."""

    async def run_protocol_smoke() -> None:
        client_write, server_read = anyio.create_memory_object_stream(0)
        server_write, client_read = anyio.create_memory_object_stream(0)
        initialization_options = mcp._mcp_server.create_initialization_options()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                mcp._mcp_server.run,
                server_read,
                server_write,
                initialization_options,
            )
            async with ClientSession(client_read, client_write) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                tools = await asyncio.wait_for(session.list_tools(), timeout=5)
                analyzer_tool = next(
                    tool for tool in tools.tools if tool.name == "read_analyzer_resource"
                )
                assert analyzer_tool.inputSchema["required"] == ["path"]
                assert "/predictions" in (analyzer_tool.description or "")

                # An invalid path is rejected before HTTP access. Reaching this
                # error proves the MCP dispatcher invoked the registered tool.
                result = await asyncio.wait_for(
                    session.call_tool(
                        "read_analyzer_resource",
                        {"path": "/outside-analyzer", "source": "host"},
                    ),
                    timeout=5,
                )
                assert result.isError is True
                assert "path must start with /api/v1/" in str(result.content)
            task_group.cancel_scope.cancel()

    asyncio.run(run_protocol_smoke())


if __name__ == "__main__":
    test_mcp_session_exposes_and_dispatches_the_analyzer_tool()
