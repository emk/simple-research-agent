"""Fetch agent for fetching web pages."""

from __future__ import annotations
from typing import Optional

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

from ..client import Client
from ..memory import FetchResult, Memory
from ..tools import McpManager
from .base import Agent, AgentType


class FetchAgent(Agent):
    """URL Fetching agent."""

    async def run(
        self, client: Client, memory: Memory, tools: McpManager
    ) -> Optional[AgentType]:
        """Run the fetch agent."""

        url = memory.current_fetch_url
        if url is None:
            raise Exception("No URL to fetch.")
        result = await tools.call_tool("fetch", {"url": url, "max_length": 5000})

        markdown = Markdown(result, style="italic white")
        print(
            Panel(markdown, border_style="yellow", title="Web page", title_align="left")
        )

        # Build our instructions.
        full_question = f"""You are part of a research team! Your job is to
examine web pages and summarize any relevant content.

Here is the web page URL you fetched: {url}

Here is the content of the web page:

=== BEGIN WEB PAGE ===
{result}
=== END WEB PAGE ===

The user originally asked the following question:

> {memory.original_user_question}

To help answer that question, the research coordinator was most recently
searching for:

> {memory.current_search_query()}

Please look at the web page, and decide what content is relevant to the
user's question or to the coordinator's search query. You can return
one of three types of results:

- `relevant`: The page contains relevant information. Please summarize it for the
  research coordinator, including key points, important details, and the source
  of the information.
- `irrelevant`: The page does not contain relevant information. We can ignore it.
- `error`: An error occurred while fetching the page. Please summarize the error.
"""

        # Ask the question.
        output = client.chat_structured(full_question, model=FetchResult)
        output.print()

        memory.remove_search_result(url)
        memory.fetch_results.append(output.response)
        memory.current_fetch_url = None

        # Always return to the planning agent.
        return AgentType.PLANNING
