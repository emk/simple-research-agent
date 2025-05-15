"""Planning agent tests.

All of these tests are marked as flaky, because the underlying LLM is
non-deterministic. We try to make the "correct" course of action as clear
as possible, but the LLM may still choose to do something else."""

from __future__ import annotations

from flaky import flaky
import pytest

from simple_research.agents import AgentType
from simple_research.agents.planning import PlanningAgent
from simple_research.client import Client
from simple_research.memory import (
    FetchResult,
    Memory,
    RelevantInformation,
    SearchResult,
)
from simple_research.tools import McpManager


QUESTION = """\
According to the the newest data you can find, what are the 5 most populous cities in Vermont?"""


@pytest.mark.asyncio
@flaky
async def test_planning_agent_search(client: Client):
    """Test the PlanningAgent class."""

    memory = Memory(original_user_question=QUESTION)
    tools = McpManager()

    # Run the agent and check the result.
    agent = PlanningAgent()
    next_agent_type = await agent.run(client, memory, tools)
    assert next_agent_type is AgentType.SEARCH

    # Make sure our memory was updated.
    query = memory.current_search_query()
    assert query is not None
    assert "Vermont" in query


@pytest.mark.asyncio
@flaky
async def test_planning_agent_fetch(client: Client):
    """Test the PlanningAgent class."""

    memory = Memory(original_user_question=QUESTION)
    memory.search_query_history.append("most populous cities in Vermont")
    memory.search_results = [
        SearchResult(
            title="Most populous cities in Vermont (updated weekly!)",
            url="https://data.vermont.gov/cities",
            snippet="The most populous cities in Vermont are ...",
        ),
    ]
    tools = McpManager()

    # Run the agent and check the result.
    agent = PlanningAgent()
    next_agent_type = await agent.run(client, memory, tools)
    assert next_agent_type is AgentType.FETCH
    assert memory.current_fetch_url == "https://data.vermont.gov/cities"


@pytest.mark.asyncio
@flaky
async def test_planning_agent_output(client: Client):
    """Test the PlanningAgent class."""

    memory = Memory(original_user_question=QUESTION)
    memory.search_query_history.append("most populous cities in Vermont")
    memory.fetch_results = [
        FetchResult(
            url="https://data.vermont.gov/cities",
            fetch_result=RelevantInformation(
                result_type="relevant",
                summary="""
This is a government page. It lists the most populous cities in Vermont as:

1. Burlington       44,528
2. South Burlington 21,043
3. Colchester       17,588
4. Rutland city     15,630
5. Bennington       15,200

This page appears to have been updated this month!
""",
            ),
        ),
    ]
    tools = McpManager()

    # Run the agent and check the result.
    agent = PlanningAgent()
    next_agent_type = await agent.run(client, memory, tools)
    assert next_agent_type is AgentType.OUTPUT
