"""llm-research-agent: A research agent for LLMs.

Usage:
    uv run research

Connects to an OpenAI-compatible API, typically a local Ollama server, and runs
a research agent.
"""

from __future__ import annotations

import asyncio
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich import print

from .client import Client
from .memory import Memory
from .agents import Agent
from .agents.planning import PlanningAgent
from .tools import McpManager


def main():
    """Main entry point. Called by `uv run research`."""
    print("[bold magenta]Starting research agent![/bold magenta]")
    load_dotenv()
    asyncio.run(run())


async def run():
    """Run the research agent."""

    # Ask the user for a question.
    prompt_session = PromptSession()
    with patch_stdout():
        question = await prompt_session.prompt_async(
            "What is your question?\n>>> ",
            prompt_continuation="... ",
            default='What are some commonly-advocated advantages of "city block" designs in the game Factorio?',
            wrap_lines=True,
        )

    client = Client()
    memory = Memory(original_user_question=question)

    # Set up our tool manager.
    tools = await McpManager.from_config()
    tools.show_tools()

    # result = await tools.call_tool("calculate", {"expression": "2 + 2"})
    # print("Tool result:", result)

    agent: Agent = PlanningAgent()
    while agent is not None:
        next_agent_type = await agent.run(client, memory, tools)
        if next_agent_type is None:
            break
        agent = Agent.lookup(next_agent_type)

    print("[bold magenta]Done![/bold magenta]")

    await tools.close()
