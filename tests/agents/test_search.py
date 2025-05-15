"""Search agent tests.

All of these tests are marked as flaky, because the underlying LLM is
non-deterministic. We try to make the "correct" course of action as clear
as possible, but the LLM may still choose to do something else."""

from __future__ import annotations

from flaky import flaky
import pytest

from simple_research.agents import AgentType
from simple_research.agents.search import SearchAgent
from simple_research.client import Client
from simple_research.memory import Memory
from simple_research.tools import McpManager


QUESTION = """\
According to the the newest data you can find, what are the 5 most populous cities in Vermont?"""

SEARCH_RESULTS = """\
Title: Most populous cities in California
URL: https://www.geodata.com/us/ca/
Snippet: The most populous cities in California are...

Title: Most populous cities in Vermont
URL: https://data.vermont.gov/cities
Snippet: The most populous cities in Vermont are...

Title: Most populous cities in New York
URL: https://www.geodata.com/us/ny/
Snippet: The most populous cities in New York are...
"""


@pytest.mark.asyncio
@flaky
async def test_search_agent(client: Client):
    """Test the SearchAgent class."""

    memory = Memory(original_user_question=QUESTION)
    memory.search_query_history.append("most populous cities in Vermont")
    tools = McpManager()
    tools.add_mock("google_search", SEARCH_RESULTS)

    # Run the agent and check the result.
    agent = SearchAgent()
    next_agent_type = await agent.run(client, memory, tools)
    assert next_agent_type is AgentType.PLANNING

    # Make sure our memory was updated with only the relevant search result.
    assert len(memory.search_results) == 1
    assert memory.search_results[0].title == "Most populous cities in Vermont"
    assert memory.search_results[0].url == "https://data.vermont.gov/cities"
    assert (
        memory.search_results[0].snippet == "The most populous cities in Vermont are..."
    )
