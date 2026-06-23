# ppagent Project Guidelines

This file contains rules, style guidelines, behavioral constraints, and source code structure specific to the `ppagent` workspace.

## Technology Stack & Conventions
- **Language:** Python 3.12+
  - Always use modern Python features (e.g., `|` for union types, `list[str]`, `dict[str, Any]` for collections).
  - Use `from __future__ import annotations` at the top of files to enable forward evaluation of type hints.
- **Data Models:** Use `pydantic` v2.
  - Rely on `Field` with `default_factory` for mutable defaults.
  - Use `model_post_init` for post-validation initialization.
- **CLI & TUI:**
  - Build command-line interfaces using `typer`.
  - Format terminal output using `rich` for consistency and styling.
- **LLM Integration:**
  - Structured output parsing is handled via `instructor` and `pydantic`. Define clear schemas with descriptions in Pydantic models for reliable extraction.
  - All provider knowledge — endpoints, default models, URL-based detection, and per-provider "thinking/reasoning" parameters — lives in a **single registry** in `providers.py`. To support a new LLM provider, add one `ProviderSpec` entry there; both `llm.py` and `tui.py` pick it up automatically. Do not hand-maintain parallel vendor lists or detection logic elsewhere.
- **PDF & Markdown:**
  - PDF processing uses `PyMuPDF`.
  - Templating and report generation rely on `Jinja2` and `markdown`.
- **Package & Environment Management:**
  - Use `uv` to handle things (e.g., dependencies, environment management, and script execution).

## Code Style & Linting
- **Formatter/Linter:** The project uses `ruff`. Ensure code is compliant with `ruff` checks.
- **Type Hints:** Strict typing is expected. All function signatures and class definitions must be fully typed.

## Testing
- **Framework:** `pytest`.
- Write tests in the `tests/` directory.

## File Organization & Modules

Production code lives in the `src/ppagent/` directory:
- **`cli.py`**: Command Line Interface definitions using `typer`. Entry point for terminal commands.
- **`tui.py`**: Terminal User Interface and formatting using `rich`. Handles all stylized console outputs.
- **`models.py`**: Contains all Pydantic data models. Defines structured outputs used by the LLM via `instructor`.
- **`pipeline.py`**: The multi-agent orchestrator. Manages parallel execution, dependencies, and flow control across agents (Searcher, Classifier, Writer, Finder, Criticizer, Assembler).
- **`llm.py`**: Interacts with the different LLM providers (OpenAI, Anthropic, Google GenAI, etc.). Handles streaming and structured outputs.
- **`providers.py`**: Single source of truth for LLM provider knowledge — endpoints, default models, URL-based detection, and per-provider "thinking/reasoning" parameters. To support a new provider, add one `ProviderSpec` entry here; `llm.py` and `tui.py` consume it automatically.
- **`config.py`**: Application configuration, reading and writing user preferences from profile/settings.
- **`hf.py`**: Integration with Hugging Face Hub (fetching papers, metrics, and metadata).
- **`pdf.py`**: Core PDF downloading and text/metadata extraction.
- **`arxiv_html.py`**: Specialized handling for arXiv HTML fetch and parse, figure extraction, and deterministic section mapping.
- **`storage.py`**: Caching and saving mechanisms for reports, generated HTML/Markdown files, and images.
- **`scheduler.py`**: Cron-like scheduling utilities for automated paper discovery tasks.
- **`agents/`**: Directory meant for implementations of individual or custom specialized agents.
- **`publishers/`**: Handlers for exporting and publishing the finalized reports to external platforms (e.g., Notion, WeChat, Webhooks).

## Developer Guidelines & Agent Behaviors
- When modifying structured LLM schemas, ensure that the fields have descriptive docstrings or `Field(description=...)` to help the language model understand the intended output.
- Prioritize making clean, self-contained Markdown/HTML reports as requested by the original application intent.
- Follow existing patterns for multi-agent workflows (Searcher, Classifier, Writer, Finder, Criticizer, Assembler).
- Modify `pipeline.py` only when the flow of agents needs to be updated.
- All new data schemas must be registered as Pydantic models in `models.py`.
- Any external API integration should be encapsulated within its own module (e.g., `llm.py`, `hf.py`, `publishers/`).
