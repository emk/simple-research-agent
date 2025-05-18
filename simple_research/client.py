"""LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import environ
import re
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

from .ui import spinner


class Client:
    """A client for the LLM."""

    client: OpenAI
    """The OpenAI client."""
    model: str

    def __init__(self) -> None:
        """Initializes the client."""
        self.client = OpenAI(
            base_url=environ.get("OPENAI_API_BASE"),
            api_key=environ.get("OPENAI_API_KEY"),
        )
        self.model = environ.get("LLM_MODEL")

    def chat(self, question: str) -> TextResponse:
        """Runs the model with the given question and returns a response."""
        with spinner("Thinking..."):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                max_completion_tokens=6 * 1024,
            )
        return TextResponse.from_response(completion)

    def chat_structured[T: BaseModel](
        self,
        question: str,
        *,
        model: type[T],
    ) -> StructuredResponse[T]:
        """Runs the model with the given question and returns a structured response."""
        full_question = f"""{question}\n\nPlease respond with a JSON value that matches the following schema:\n{model.model_json_schema()}"""
        response = self.chat(full_question)
        return response.parse(model)


class Response(ABC):
    """A response from the LLM."""

    thinking: str | None
    """The thinking content of the response."""

    def __init__(self, thinking: str | None) -> None:
        """Initializes the response."""
        self.thinking = thinking

    @abstractmethod
    def print(self) -> None:
        """Prints the response to the console."""
        if self.thinking:
            markdown = Markdown(self.thinking, style="italic white")
            print(
                Panel(
                    markdown, border_style="blue", title="Thinking", title_align="left"
                )
            )


class TextResponse(Response):
    """A text response from the LLM."""

    response: str
    """The response content."""

    def __init__(self, thinking: str | None, response: str) -> None:
        """Initializes the response."""
        super().__init__(thinking)
        self.response = response

    @classmethod
    def from_response(cls, message: ChatCompletion) -> TextResponse:
        """Creates a Response from a ChatCompletion message."""
        content = message.choices[0].message.content

        # Thinking content is inside leading <think>...</think> tags, followed by the response.
        matches = re.match(
            r"^\s*<think>(.*?)</think>\s*(.*)$",
            content,
            re.DOTALL,
        )
        if matches:
            thinking, response = matches.groups()
        else:
            thinking = None
            response = content

        return cls(thinking=thinking, response=response)

    def print(self) -> None:
        """Prints the response to the console."""
        super().print()
        print(Markdown(self.response))

    def parse[T: BaseModel](self, model: type[T]) -> StructuredResponse[T]:
        """Parses the response as a Pydantic model."""
        try:
            parsed = model.model_validate_json(self.response)
            return StructuredResponse(thinking=self.thinking, response=parsed)
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}\n{self.response}") from e


class StructuredResponse[T: BaseModel](Response):
    """A structured response from the LLM."""

    response: T
    """The parsed response content."""

    def __init__(self, thinking: str | None, response: T) -> None:
        """Initializes the response."""
        super().__init__(thinking)
        self.response = response

    def print(self) -> None:
        """Prints the response to the console."""
        super().print()
        print(self.response)
        # print(self.response.model_dump_json(indent=2))
