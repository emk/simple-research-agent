"""Memory state and management for the research agents."""

from __future__ import annotations
from io import StringIO
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Memory:
    """Memory state and management for the research agents."""

    original_user_question: str
    """The original question asked by the user."""

    search_query_history: list[str]
    """The current search query being processed, if any."""

    current_fetch_url: str | None
    """The current URL being fetched, if any."""

    search_results: list[SearchResult]
    """Search results we have found."""

    fetch_results: list[FetchResult]
    """Summaries of fetched documents."""

    final_report: str | None = None
    """The final report we printed to the user. Used for testing."""

    def __init__(self, original_user_question: str) -> None:
        """Initializes the memory with the original question."""
        self.original_user_question = original_user_question
        self.search_query_history = []
        self.current_fetch_url = None
        self.search_results = []
        self.fetch_results = []
        self.final_report = None

    def contains_data(self) -> bool:
        """Return True if we have any data in memory."""
        return (
            len(self.search_query_history) > 0
            or len(self.search_results) > 0
            or len(self.fetch_results) > 0
        )

    def current_search_query(self) -> str | None:
        """Return the current search query, if any."""
        if len(self.search_query_history) > 0:
            return self.search_query_history[-1]
        return None

    def add_search_result(self, result: SearchResult) -> None:
        """Add a search result to memory.

        If we've already seen this result, don't add it again.
        """
        if all(result.url != r.url for r in self.search_results) and all(
            fetched.url != result.url for fetched in self.fetch_results
        ):
            self.search_results.append(result)

    def remove_search_result(self, url: str) -> None:
        """Remove a search result from memory."""
        self.search_results = [
            result for result in self.search_results if result.url != url
        ]

    def __str__(self) -> str:
        """Summarize the memory as text, for use in LLM prompts."""
        wtr = StringIO()
        wtr.write(f"""## Original user question

> {self.original_user_question}

""")

        # Queries.
        if len(self.search_query_history) > 0:
            wtr.write("## Search queries your research team has tried\n")
            for query in self.search_query_history:
                wtr.write(f"- {query}\n")
            wtr.write("\n")

        # Fetch results.
        relevant = [result for result in self.fetch_results if result.is_relevant()]
        if len(relevant) > 0:
            wtr.write("## Pages your research team already fetched (summarized)\n\n")
            wtr.write(f"Total pages fetched: {len(self.fetch_results)}\n\n")
            for result in relevant:
                wtr.write(f"{result}\n")

        # Search results.
        if len(self.search_results) > 0:
            wtr.write(
                "## Search results your research team could fetch if you need more information\n"
            )
            for result in self.search_results:
                wtr.write(f"{result}\n")

        return wtr.getvalue()


class SearchResult(BaseModel):
    """A single search result."""

    model_config = ConfigDict(extra="forbid")

    title: str
    """The title of the result."""

    url: str
    """The URL of the result."""

    snippet: str
    """A short snippet returned by the search engine."""

    def __str__(self) -> str:
        """Return a string representation of the search result."""
        snippet = re.sub("\\s+", " ", self.snippet).strip()
        return f"Title: {self.title}\nURL: {self.url}\nSnippet: {snippet}\n"


class FetchResult(BaseModel):
    """Search results."""

    model_config = ConfigDict(extra="forbid")

    url: str
    """The URL of the page we tried to fetch."""

    fetch_result: RelevantInformation | IrrevelevantInformation | FetchError
    """The result of the fetch."""

    def is_relevant(self) -> bool:
        """Return True if the result is relevant."""
        return self.fetch_result.result_type == "relevant"

    def __str__(self) -> str:
        """Return a string representation of the fetch result."""

        return f"""### Data fetched from: {self.url}

{self.fetch_result}

"""


class RelevantInformation(BaseModel):
    """Relevant information from the web page."""

    model_config = ConfigDict(extra="forbid")

    result_type: Literal["relevant"] = "relevant"
    """The type of result."""

    summary: str
    """The summary of the relevant information."""

    def __str__(self) -> str:
        """Return a string representation of the relevant information."""
        return "> " + re.sub("\n", "\n> ", self.summary).strip()


class IrrevelevantInformation(BaseModel):
    """The web page was not relevant."""

    model_config = ConfigDict(extra="forbid")

    result_type: Literal["irrelevant"] = "irrelevant"
    """The type of result."""

    summary: str
    """A summary of why this page was not relevant."""

    def __str__(self) -> str:
        """Return a string representation of the irrelevant information."""
        return "This page was not relevant to the user's question."


class FetchError(BaseModel):
    """An error occurred while fetching the web page."""

    model_config = ConfigDict(extra="forbid")

    result_type: Literal["error"] = "error"
    """The type of result."""

    error_summary: str
    """A summary of the error which occurred."""

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return f"An error occurred while fetching the page: {self.error_summary}"
