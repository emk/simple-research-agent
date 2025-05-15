"""Fetch agent tests.

All of these tests are marked as flaky, because the underlying LLM is
non-deterministic. We try to make the "correct" course of action as clear
as possible, but the LLM may still choose to do something else."""

from __future__ import annotations

from flaky import flaky
import pytest

from simple_research.agents.output import OutputAgent
from simple_research.client import Client
from simple_research.memory import FetchResult, Memory, RelevantInformation
from simple_research.tools import McpManager


QUESTION = """\
According to the the newest data you can find, what are the 5 most populous cities in Vermont?"""

FETCH_SUMMARY = """\
As of 2025, the most populous cities in Vermont are:

1. Burlington       44,528
2. South Burlington 21,043
3. Colchester       17,588
4. Rutland City     15,630
5. Bennington       15,200
"""


@pytest.mark.asyncio
@flaky
async def test_output_agent(client: Client):
    """Test the FetchAgent class."""

    memory = Memory(original_user_question=QUESTION)
    memory.search_query_history.append("most populous cities in Vermont")
    memory.fetch_results = [
        FetchResult(
            url="https://data.vermont.gov/cities",
            fetch_result=RelevantInformation(
                result_type="relevant",
                summary=FETCH_SUMMARY,
            ),
        ),
    ]
    tools = McpManager()
    # Run the agent and check the result.
    agent = OutputAgent()
    next_agent_type = await agent.run(client, memory, tools)
    assert next_agent_type is None

    # Make sure we prepared a final report.
    final_report = memory.final_report
    assert final_report is not None
    for city in [
        "Burlington",
        "South Burlington",
        "Colchester",
        "Rutland City",
        "Bennington",
    ]:
        assert city in final_report

    # Also make sure we cited our source!
    assert "https://data.vermont.gov/cities" in final_report
