from __future__ import annotations

from simple_research.memory import (
    FetchResult,
    Memory,
    RelevantInformation,
    SearchResult,
)


def test_memory():
    memory = Memory(original_user_question="What is the capital of France?")
    assert memory.original_user_question == "What is the capital of France?"
    assert memory.search_query_history == []
    assert memory.current_fetch_url is None
    assert memory.search_results == []
    assert memory.fetch_results == []
    assert memory.current_search_query() is None

    memory.search_query_history.append("capital of France")
    assert memory.search_query_history == ["capital of France"]
    assert memory.current_search_query() == "capital of France"

    sample_search_result = SearchResult(
        title="Paris - Wikipedia",
        url="https://en.wikipedia.org/wiki/Paris",
        snippet="Paris is the capital of France.",
    )

    sample_fetch_result = FetchResult(
        url="https://en.wikipedia.org/wiki/France",
        fetch_result=RelevantInformation(
            result_type="relevant",
            summary="France is a country in Europe.",
        ),
    )

    memory.search_results.append(sample_search_result)
    assert len(memory.search_results) == 1
    memory.remove_search_result(sample_search_result.url)
    assert len(memory.search_results) == 0

    memory.search_results.append(sample_search_result)
    memory.fetch_results.append(sample_fetch_result)
    summary = str(memory)
    assert "What is the capital of France?" in summary
    assert "capital of France" in summary
    assert sample_search_result.title in summary
    assert sample_search_result.url in summary
    assert sample_search_result.snippet in summary
    assert sample_fetch_result.url in summary
    actual_fetch_result = sample_fetch_result.fetch_result
    assert isinstance(actual_fetch_result, RelevantInformation)
    assert actual_fetch_result.summary in summary
