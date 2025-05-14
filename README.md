# `simple_research`: Experimental Research Agent

**WORK IN PROGRESS:** This repository is an attempt to mimic the various "Deep Research" tools from the big AI labs, but using Qwen 30B A3B. Think of this more as a benchmark and proof-of-concept than an actual useful tool.

It's not yet complete enough to run.

## Configuring

Copy `.env.template` to `.env` and fill in the missing credentials.

## Running

You will need to [install `uv`](https://docs.astral.sh/uv/getting-started/installation/). From there, you can run any of the following:

```sh
# Run type checks (experimental).
uv run ty check

# Format code.
uv run ruff format

# Run tests.
uv run pytest

# Run research agent. Ask it questions interactively!
uv run research
```
