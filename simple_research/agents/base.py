"""Base type for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


from ..client import Client
from ..memory import Memory
from ..tools import McpManager


class AgentType(str, Enum):
    """Types of agents we support."""

    PLANNING = "planning"
    """Planning agent. Makes all the big decisions."""

    SEARCH = "search"
    """Search agent."""

    FETCH = "fetch"
    """Fetch agent."""

    OUTPUT = "output"
    """Output agent."""


class Agent(ABC):
    """Base type for all agents."""

    @abstractmethod
    async def run(
        self, client: Client, memory: Memory, tools: McpManager
    ) -> Optional[AgentType]:
        """Run the agent with the given client and memory, and
        return the type of agent to run next."""
        pass

    @staticmethod
    def lookup(agent_type: AgentType) -> Agent:
        """Lookup the agent class for the given agent type."""
        if agent_type == AgentType.PLANNING:
            from .planning import PlanningAgent

            return PlanningAgent()
        elif agent_type == AgentType.SEARCH:
            from .search import SearchAgent

            return SearchAgent()
        elif agent_type == AgentType.FETCH:
            from .fetch import FetchAgent

            return FetchAgent()
        elif agent_type == AgentType.OUTPUT:
            from .output import OutputAgent

            return OutputAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
