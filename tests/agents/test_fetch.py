"""Fetch agent tests.

All of these tests are marked as flaky, because the underlying LLM is
non-deterministic. We try to make the "correct" course of action as clear
as possible, but the LLM may still choose to do something else."""

from __future__ import annotations

from flaky import flaky
import pytest

from simple_research.agents import AgentType
from simple_research.agents.fetch import FetchAgent
from simple_research.client import Client
from simple_research.memory import Memory, RelevantInformation
from simple_research.tools import McpManager


QUESTION = """\
According to the the newest data you can find, what are the 5 most populous cities in Vermont?"""

FETCH_RESULTS = """\
Vermont.gov: Most Populous Cities

As of 2025, the most populous cities in Vermont are:

1. Burlington       44,528
2. South Burlington 21,043
3. Colchester       17,588
4. Rutland city     15,630
5. Bennington       15,200
6. Brattleboro      12,110
7. Essex            11,462
8. Essex Junction   10,817
9. Hartford         10,743
10. Milton          10,735
"""


@pytest.mark.asyncio
@flaky
async def test_fetch_agent(client: Client):
    """Test the FetchAgent class."""

    memory = Memory(original_user_question=QUESTION)
    memory.search_query_history.append("most populous cities in Vermont")
    memory.current_fetch_url = "https://data.vermont.gov/cities"
    tools = McpManager()
    tools.add_mock("fetch", FETCH_RESULTS)

    # Run the agent and check the result.
    agent = FetchAgent()
    next_agent_type = await agent.run(client, memory, tools)
    assert next_agent_type is AgentType.PLANNING

    # Make sure our memory was updated with only the relevant search result.
    assert len(memory.fetch_results) == 1
    assert memory.fetch_results[0].url == "https://data.vermont.gov/cities"
    fetch_result = memory.fetch_results[0].fetch_result
    assert isinstance(fetch_result, RelevantInformation)
    summary = fetch_result.summary
    for city in [
        "Burlington",
        "South Burlington",
        "Colchester",
        "Rutland city",
        "Bennington",
    ]:
        assert city in summary
