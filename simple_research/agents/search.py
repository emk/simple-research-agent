"""Search agent."""

from __future__ import annotations
from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

from ..client import Client
from ..memory import Memory, SearchResult
from ..tools import McpManager
from .base import Agent, AgentType

SKIP_SITES: List[str] = [
    # robots.txt forbids downloads from reddit.
    "reddit.com",
    # We can't use videos.
    "youtube.com",
    # Generally useless, especially if not logged in.
    "twitter.com",
    "facebook.com",
]
"""Sites we shouldn't search."""


class SearchAgent(Agent):
    """Search agent."""

    async def run(
        self, client: Client, memory: Memory, tools: McpManager
    ) -> Optional[AgentType]:
        """Run the search agent."""

        exclusions = " ".join([f"-site:{site}" for site in SKIP_SITES])
        query = f"{memory.current_search_query()} {exclusions}"
        result = await tools.call_tool("google_search", {"search_term": query})

        markdown = Markdown(result, style="italic white")
        print(
            Panel(
                markdown,
                border_style="yellow",
                title="Search results",
                title_align="left",
            )
        )

        # Build our instructions.
        full_question = f"""You are part of a research team! Your job is to
examine search results and decide which might be relevant to the user's
question.

Here are the search results:

=== BEGIN SEARCH RESULTS ===
{result}
=== END SEARCH RESULTS ===

The user originally asked the following question:

> {memory.original_user_question}

To help answer that question, the research coordinator asked you to search for:

> {memory.current_search_query()}

Please look at the search results, and decide which are most relevant to the
user's question and to the coordinator's search query. Exclude any results which
seem unlikely to be relevant, or which are likely to be videos.
"""

        # Ask the question.
        output = client.chat_structured(full_question, model=SearchResults)
        output.print()

        for result in output.response.results:
            memory.search_results.append(result)

        # Always return to the planning agent.
        return AgentType.PLANNING


class SearchResults(BaseModel):
    """Search results."""

    model_config = ConfigDict(extra="forbid")

    results: List[SearchResult]
    """Search results."""
