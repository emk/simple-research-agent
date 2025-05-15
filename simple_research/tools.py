"""Interface to MCP tools."""

from __future__ import annotations
from contextlib import AbstractAsyncContextManager
from os import environ

from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from typing import List
from mcp import ClientSession, StdioServerParameters, Tool, stdio_client
from mcp.types import TextContent


def get_mcp_params() -> List[StdioServerParameters]:
    """MCP parameters for the tools we support.

    No need to run `uvx` for these tools, because they are installed by our top-level
    `uv` setup.
    """
    return [
        StdioServerParameters(
            command="mcp-server-calculator",
            args=[],
        ),
        StdioServerParameters(
            command="mcp-server-fetch",
            args=[],
        ),
        StdioServerParameters(
            command="mcp-google-cse",
            args=[],
            env={
                "API_KEY": environ["GOOGLE_CSE_API_KEY"],
                "ENGINE_ID": environ["GOOGLE_CSE_ID"],
            },
        ),
    ]


class McpClient:
    """Client for an MCP server."""

    context_managers: List[AbstractAsyncContextManager]

    session: ClientSession
    """The MCP client session."""

    tools: List[Tool]
    """The list of tools available on the server."""

    def __init__(
        self,
        managers: List[AbstractAsyncContextManager],
        session: ClientSession,
        tools: List[Tool],
    ) -> None:
        """Initializes the client with the given session."""
        self.context_managers = managers
        self.session = session
        self.tools = tools

    @classmethod
    async def from_params(
        cls,
        params: StdioServerParameters,
    ) -> McpClient:
        """Creates a new client from the given parameters."""
        # Here is what this normally looks like:
        #
        # async with stdio_client(params) as (read, write):
        #     print(f"Started MCP client for {params.command}")
        #     async with ClientSession(read, write) as session:
        #         print(f"Connected to MCP server {params.command}")
        #         await session.initialize()
        #         print(f"Initialized MCP server {params.command}")
        #         tools_result = await session.list_tools()
        #         print(f"Got tools: {tools_result.tools}")
        #         raise "All good!"

        managers = []

        stdio_manager = stdio_client(params)
        # print(f"Started MCP client for {params.command}")
        read, write = await stdio_manager.__aenter__()
        managers.append(stdio_manager)

        session_manager = ClientSession(read, write)
        session = await session_manager.__aenter__()
        managers.append(session_manager)

        # print(f"Connected to MCP server {params.command}")
        await session.initialize()
        # print(f"Initialized MCP server {params.command}")
        tools_result = await session.list_tools()
        # print(f"Got tools: {tools_result.tools}")
        return cls(managers, session, tools_result.tools)

    def list_tools(self) -> List[Tool]:
        """Lists the available tools. Cached."""
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Calls the given tool with the given arguments."""
        result = await self.session.call_tool(tool_name, arguments=arguments)
        if result.isError:
            raise RuntimeError(f"Error calling tool {tool_name}: {result}")
        else:
            text = []
            for content in result.content:
                if isinstance(content, TextContent):
                    text.append(content.text)
                else:
                    raise RuntimeError(
                        f"Unexpected content from tool {tool_name}: {content}"
                    )
            return "\n\n".join(text)

    async def close(self) -> None:
        """Closes the client."""
        for manager in reversed(self.context_managers):
            await manager.__aexit__(None, None, None)


class McpManager:
    """Manager for our MCP servers."""

    clients: List[McpClient]
    """The list of MCP clients."""

    tool_map: dict[str, McpClient]
    """The map of tool names to clients."""

    def __init__(self) -> None:
        """Initializes the manager with no MCP servers."""
        self.clients = []
        self.tool_map = {}

    @classmethod
    async def from_config(cls) -> McpManager:
        """Creates a new manager from the configuration."""
        manager = cls()
        for params in get_mcp_params():
            await manager.add_client(params)
        return manager

    async def add_client(self, params: StdioServerParameters) -> None:
        """Adds a new MCP client to the manager."""
        # print(f"Adding MCP client for {params.command}")
        client: McpClient = await McpClient.from_params(params)
        self.clients.append(client)
        for tool in client.list_tools():
            self.tool_map[tool.name] = client

    def get_tools(self) -> List[Tool]:
        """Returns the list of tools available on all clients."""
        tools = []
        for client in self.clients:
            tools.extend(client.list_tools())
        return tools

    def show_tools(self) -> None:
        """Prints the list of tools available on all clients."""
        tool_info: List[str] = []
        for client in self.clients:
            for tool in client.list_tools():
                description = tool.description.strip().replace("\n", "\n  ")
                tool_info.append(f"- `{tool.name}`: {description}\n")
        markdown = Markdown("".join(tool_info))
        print(
            Panel(
                markdown,
                title="Available tools",
                title_align="left",
                border_style="yellow",
                style="italic white",
            )
        )

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Calls the given tool with the given arguments."""
        client = self.tool_map[tool_name]
        return await client.call_tool(tool_name, arguments=arguments)

    async def close(self) -> None:
        """Closes all clients."""
        for client in reversed(self.clients):
            await client.close()
        self.clients = []
        self.tool_map = {}
