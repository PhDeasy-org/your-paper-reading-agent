# ppagent Project Guidelines

This file contains rules, style guidelines, and behavioral constraints specific to the `ppagent` workspace.

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

## File Organization
- Production code lives in the `src/ppagent/` directory.
- `config.py` handles configurations.
- `models.py` contains all Pydantic models and structured LLM output definitions.
- `pipeline.py` orchestrates the multi-agent execution.

## General Agent Behaviors
- When modifying structured LLM schemas, ensure that the fields have descriptive docstrings or `Field(description=...)` to help the language model understand the intended output.
- Prioritize making clean, self-contained Markdown/HTML reports as requested by the original application intent.
- Follow existing patterns for multi-agent workflows (Searcher, Classifier, Writer, Finder, Criticizer, Vision, Assembler).
