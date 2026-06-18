# ppagent Source Code Structure

This file provides an overview of the core components in the `src/ppagent` directory. This acts as a reference for agents modifying or interacting with the codebase.

## Directory Structure & Modules

- **`cli.py`**: Command Line Interface definitions using `typer`. Entry point for terminal commands.
- **`tui.py`**: Terminal User Interface and formatting using `rich`. Handles all stylized console outputs.
- **`models.py`**: Contains all Pydantic data models. Defines structured outputs used by the LLM via `instructor`.
- **`pipeline.py`**: The multi-agent orchestrator. Manages parallel execution, dependencies, and flow control across agents (Searcher, Classifier, Writer, Finder, Criticizer, Vision, Assembler).
- **`llm.py`**: Interacts with the different LLM providers (OpenAI, Anthropic, Google GenAI, etc.). Handles streaming and structured outputs.
- **`providers.py`**: Single source of truth for LLM provider knowledge — endpoints, default models, URL-based detection, and per-provider "thinking/reasoning" parameters. To support a new provider, add one `ProviderSpec` entry here; `llm.py` and `tui.py` consume it automatically.
- **`config.py`**: Application configuration, reading and writing user preferences from profile/settings.
- **`hf.py`**: Integration with Hugging Face Hub (fetching papers, metrics, and metadata).
- **`pdf.py`**: Core PDF downloading and text/metadata extraction.
- **`figures.py`**: Specialized handling for PDF figure extraction (via `PyMuPDF`) and selecting the best figure for the final report.
- **`storage.py`**: Caching and saving mechanisms for reports, generated HTML/Markdown files, and images.
- **`scheduler.py`**: Cron-like scheduling utilities for automated paper discovery tasks.
- **`agents/`**: Directory meant for implementations of individual or custom specialized agents.
- **`publishers/`**: Handlers for exporting and publishing the finalized reports to external platforms (e.g., Notion, WeChat, Webhooks).

## Developer Guidelines
When working within this directory:
- Modify `pipeline.py` only when the flow of agents needs to be updated.
- All new data schemas must be registered as Pydantic models in `models.py`.
- Any external API integration should be encapsulated within its own module (e.g., `llm.py`, `hf.py`, `publishers/`).
