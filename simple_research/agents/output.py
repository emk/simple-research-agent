"""Output agent.

This agent is responsible for writing up the final output for the user."""

from __future__ import annotations
from typing import Optional

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

from ..client import Client
from ..memory import Memory
from ..tools import McpManager
from .base import Agent, AgentType


class OutputAgent(Agent):
    """Research report output agent."""

    async def run(
        self, client: Client, memory: Memory, tools: McpManager
    ) -> Optional[AgentType]:
        """Run the output agent."""

        # Clear any unfetched search results.
        memory.search_results = []

        # Let's hope we don't make it here without data.
        if not memory.contains_data():
            raise Exception("No data to output.")
        notebook = str(memory)

        markdown = Markdown(notebook, style="italic white")
        print(
            Panel(
                markdown,
                border_style="yellow",
                title="Research notebook",
                title_align="left",
            )
        )

        # Build our instructions.
        full_question = f"""You are writer who summarizes a research notebook!

Here is the content of the notebook:

{notebook}

## Your job

The user has asked the following question:

> {memory.original_user_question}

Your job is to use the information in the research notebook to write a clear,
accurate, and concise answer to the user's question.

If you are unsure about the answer, it is important to say so!

### IMPORTANT!

You should **always** support your claims by citing the sources in the notebook.
At the end of your answer, include a list of the full URLs to the pages you
used.
"""

        # Ask the question.
        output = client.chat(full_question)
        output.print()

        # Record our final report for testing, etc.
        memory.final_report = output.response

        # Once we print the output, we're done.
        return None
