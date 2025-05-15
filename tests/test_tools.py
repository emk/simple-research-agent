"""Test cases for the tools  subsystem."""

from __future__ import annotations

import pytest

from simple_research.tools import McpManager


@pytest.mark.asyncio
async def test_mcp_manager(dotenv: None):
    """Test the McpManager class."""

    manager = await McpManager.from_config()
    tools = manager.get_tools()
    tool_names = [tool.name for tool in tools]
    assert "calculate" in tool_names

    result = await manager.call_tool("calculate", {"expression": "2 + 3"})
    assert result.strip() == "5"
