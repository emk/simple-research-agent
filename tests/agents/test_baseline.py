"""Baseline agent tests.

All of these tests are marked as flaky, because the underlying LLM is
non-deterministic. We try to make the "correct" course of action as clear
as possible, but the LLM may still choose to do something else."""

from __future__ import annotations

from flaky import flaky
import pytest

from simple_research.agents.baseline import BaselineAgent
from simple_research.client import Client
from simple_research.memory import Memory
from simple_research.tools import McpManager


QUESTION = """\
According to the the newest data you can find, what are the 5 most populous cities in Vermont?"""


@pytest.mark.asyncio
@flaky
async def test_planning_agent_search(client: Client):
    """Test the BaselineAgent class."""

    memory = Memory(original_user_question=QUESTION)
    tools = McpManager()

    # Run the agent and check the result.
    agent = BaselineAgent()
    next_agent_type = await agent.run(client, memory, tools)
    assert next_agent_type is None
