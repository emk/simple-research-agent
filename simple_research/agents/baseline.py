"""Baseline agent that answers without researching.

We use this agent to establish a baseline, showing what the model can
do without any research.
"""

from __future__ import annotations
from typing import Optional

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

from ..client import Client
from ..memory import Memory
from ..tools import McpManager
from .base import Agent, AgentType


class BaselineAgent(Agent):
    """Responds to questions with no research, memory or tools. For comparison
    purposes."""

    async def run(
        self, client: Client, memory: Memory, _tools: McpManager
    ) -> Optional[AgentType]:
        """Run a simple query, with no memory or tools."""

        # Ask the question.
        output = client.chat(memory.original_user_question)

        # Format the output specially.
        thinking = output.thinking
        if thinking is not None:
            print(
                Panel(
                    Markdown(thinking),
                    style="italic white",
                    border_style="blue",
                    title="Baseline thinking (no search)",
                    title_align="left",
                )
            )
        print(
            Panel(
                Markdown(output.response),
                style="white",
                border_style="yellow",
                title="Baseline output (no search)",
                title_align="left",
            )
        )

        # Once we print the output, we're done.
        return None
