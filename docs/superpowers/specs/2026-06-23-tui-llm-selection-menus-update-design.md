# Design Spec: TUI LLM Selection Menus Update

Update the TUI settings pages for LLM selection to remove `Model Name` inputs and consolidate model selection under a single `Models` menu item.

## Problem Description
Currently, when configuring an LLM provider in the TUI:
1. The user sees a text input field called `Model Name` (e.g., `gpt-4o`).
2. If the provider has pre-defined stable aliases, they also see a submenu entry called `Latest Models` where they can choose from a list or choose "Custom Model (type your own)...".
3. This is redundant and confusing, as a user can configure custom models in the picker page anyway.

## Proposed Changes

### TUI Settings Submenus
We will modify the LLM settings submenu builder in `src/ppagent/tui.py`:
1. Remove the `Model Name` `MenuItem`.
2. Add a `Models` `MenuItem` for every provider (removing the `spec.latest_models` check).
3. The `Models` menu item will:
   - Target `f"llm_{role}_{vendor_key}_models"`
   - Reference key `f"llms.{role}.model"`
   - Render the current value of the model next to the label (e.g. `Models : gpt-4o`).

### Menu Formatting
Modify `format_menu_item` to render values for navigation items (items with a `target`) that have a `key`, as long as the target does not end with `_custom_model`.

### Routing and Regex Updates
In `src/ppagent/tui.py`, update:
- The menu match regex from `r"^llm_(text|searcher)_([a-z0-9_]+)_latest$"` to `r"^llm_(text|searcher)_([a-z0-9_]+)_models$"`.
- Suffix check and handling in the main keypress loop from `_latest` to `_models`.

### Test Suite Updates
In `tests/test_tui_config.py`, update all test assertions to:
1. Expect the `Models` nav item instead of `Latest Models`.
2. Assert that `Models` is displayed for all providers.
3. Assert that `Model Name` is not present in the menus.
