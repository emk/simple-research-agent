"""Planning agent."""

from __future__ import annotations
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from ..client import Client
from ..memory import Memory
from ..tools import McpManager
from .base import Agent, AgentType


class PlanningAgent(Agent):
    """Planning agent."""

    async def run(
        self, client: Client, memory: Memory, tools: McpManager
    ) -> Optional[AgentType]:
        """Run the planning agent."""

        if memory.contains_data():
            notebook = f"""
Your research notebook currently contains the following.

{str(memory)}
"""
        else:
            notebook = """
Your research notebook is currently empty.
"""

        # Build our instructions.
        full_question = f"""You are a research coordinator!
{notebook}
## Your job

As a research coordinator, your job is to manage a number of research agents who
can perform different tasks. 

These agents are:

1. "Search", who can search the web for information.
2. "Fetch", who can fetch information from a URL and summarize it for you.
2. "Output", who can output information to the user.

You use your knowledge to examine the user's question, and to ask agents to
perform their assigned tasks. If your research notebook is empty, you should
normally start by thinking of a web query that might answer the user's question,
and delegating it to the "Search" agent. The search agent will return a list of
URLs that might be relevant to the user's question. You can then ask the "Fetch"
agent to fetch any URL that seems relevant, and it will summarize the content of
the page for you. If you need more information, you can ask "Fetch" or "Search"
to help you find more.

Once you have collected (and confimed!) enough information to confidently answer
the question, you can delegate to the output agent to summarize your findings.

The user has asked the following question:

> {memory.original_user_question}
"""

        # Ask the question.
        planning_result = client.chat_structured(full_question, model=PlanningResult)
        planning_result.print()

        # Update our memory and return the next step.
        next_step = planning_result.response.next_step
        next_step.update_memory(memory)
        return next_step.agent_type()


class SearchStep(BaseModel):
    """Search for information, and summarize the results."""

    model_config = ConfigDict(extra="forbid")

    step_type: Literal["search"] = "search"
    """The type of step."""

    query: str
    """The search query."""

    def agent_type(self) -> AgentType:
        """Return the agent type for this step."""
        return AgentType.SEARCH

    def update_memory(self, memory: Memory) -> None:
        """Update the memory with the search query."""
        memory.search_query_history.append(self.query)


class FetchStep(BaseModel):
    """Fetch information from a URL, and summarize what we found."""

    model_config = ConfigDict(extra="forbid")

    step_type: Literal["fetch"] = "fetch"
    """The type of step."""

    url: str
    """The URL to fetch."""

    def agent_type(self) -> AgentType:
        """Return the agent type for this step."""
        return AgentType.FETCH

    def update_memory(self, memory: Memory) -> None:
        """Update our memory with a URL to fetch."""
        memory.current_fetch_url = self.url


class OutputStep(BaseModel):
    """Output our findings to the user.

    Do not include any extra information, because the output agent will
    automatically use the information in the research notebook!"""

    model_config = ConfigDict(extra="forbid")

    step_type: Literal["output"] = "output"
    """The type of step."""

    # Since we can't currently constrain Qwen3 to generate a specific
    # JSON Schema in <think>...</think> mode, we need to include this
    # extra field so that it doesn't try to invent some custom field
    # for passing the data to the OutputAgent. 🤦
    data_source: Literal["research_notebook"] = "research_notebook"
    """The data source to use. Should always be "research_notebook"."""

    def agent_type(self) -> AgentType:
        """Return the agent type for this step."""
        return AgentType.OUTPUT

    def update_memory(self, memory: Memory) -> None:
        """Update the memory."""
        pass


class PlanningResult(BaseModel):
    """The result of the planning step."""

    model_config = ConfigDict(extra="forbid")

    next_step: SearchStep | FetchStep | OutputStep
    """The next step to take."""

    def agent_type(self) -> AgentType:
        """Return the agent type for this step."""
        return self.next_step.agent_type()

    def update_memory(self, memory: Memory) -> None:
        """Update the memory with the search query."""
        self.next_step.update_memory(memory)
