# TUI LLM Selection Menus Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamline TUI LLM settings pages by removing the redundant `Model Name` text input and replacing `Latest Models` with a single universal `Models` picker menu for all providers.

**Architecture:** Modify `src/ppagent/tui.py` to adjust `_llm_submenu_items` (removing the text input and exposing `Models` universally), update `format_menu_item` to render the active model value on the submenu, and update the routing regex matching from `_latest` to `_models`. Update all associated tests in `tests/test_tui_config.py`.

**Tech Stack:** Python, Typer, Rich.

## Global Constraints
- **Language:** Python 3.12+ (modern Python features: `|` for union types, etc.).
- **Linting:** Compliant with `ruff` checks.
- **Testing:** Pass all unit tests in `pytest`.

---

### Task 1: Update TUI Submenus, Formatting, and Routing

**Files:**
- Modify: [tui.py](file:///Users/sn/ws/daily-paper-reading/src/ppagent/tui.py)

**Interfaces:**
- Consumes: None
- Produces: Updated TUI menu definitions and routing.

- [ ] **Step 1: Modify `_llm_submenu_items`**
  Remove "Model Name" MenuItem, and replace the conditional "Latest Models" check with a universal "Models" menu item targeting `f"llm_{role}_{vendor_key}_models"`.
  
  ```python
  # Target content to replace in src/ppagent/tui.py (lines 189-209):
      items.append(
          MenuItem(
              "Model Name",
              key=f"{prefix}.model",
              val_type=str,
              description="Target model (e.g. gpt-4o, deepseek-chat).",
          )
      )

      # Providers with stable "-latest" aliases (Grok, Gemini) get a picker so
      # the user can select one instead of typing the model name by hand.
      if spec is not None and spec.latest_models:
          items.append(
              MenuItem(
                  "Latest Models",
                  target=f"llm_{role}_{vendor_key}_latest",
                  description="Pick a '-latest' model alias that auto-routes to "
                  "the newest version of each model family.",
              )
          )
  ```
  
  Replace with:
  
  ```python
      items.append(
          MenuItem(
              "Models",
              key=f"{prefix}.model",
              target=f"llm_{role}_{vendor_key}_models",
              description="Select or customize the model to use.",
          )
      )
  ```

- [ ] **Step 2: Update regex match in `get_menu_definition`**
  Update the picker menu routing check in `get_menu_definition` (around line 543) from `_latest` to `_models`.
  
  ```python
  # Target content in src/ppagent/tui.py (lines 543-546):
      picker_match = re.match(r"^llm_(text|searcher)_([a-z0-9_]+)_latest$", menu_id)
      if picker_match:
          role, vendor_key = picker_match.groups()
  ```
  
  Replace with:
  
  ```python
      picker_match = re.match(r"^llm_(text|searcher)_([a-z0-9_]+)_models$", menu_id)
      if picker_match:
          role, vendor_key = picker_match.groups()
  ```

- [ ] **Step 3: Update `format_menu_item` to render the value for the `Models` entry**
  Update `format_menu_item` (around line 651) so that navigation items show their config value, except for those that target `_custom_model`.
  
  ```python
  # Target content in src/ppagent/tui.py (lines 651-652):
          if item.set_value is not None or item.target is not None:
              return f"{marker}{label_part}"
  ```
  
  Replace with:
  
  ```python
          if item.set_value is not None or (item.target is not None and item.target.endswith("_custom_model")):
              return f"{marker}{label_part}"
  ```

- [ ] **Step 4: Update `run_config_tui` keypress handler routing**
  Update the keypress handler in `run_config_tui` (around line 875) from `_latest` to `_models`.
  
  ```python
  # Target content in src/ppagent/tui.py (lines 875-879):
                      elif re.match(
                          r"^llm_(text|searcher)_[a-z0-9_]+_latest$",
                          item.target,
                      ):
                          menu_stack.append((item.target, 0))
  ```
  
  Replace with:
  
  ```python
                      elif re.match(
                          r"^llm_(text|searcher)_[a-z0-9_]+_models$",
                          item.target,
                      ):
                          menu_stack.append((item.target, 0))
  ```

---

### Task 2: Update Test Suite

**Files:**
- Modify: [test_tui_config.py](file:///Users/sn/ws/daily-paper-reading/tests/test_tui_config.py)

**Interfaces:**
- Consumes: Updated `src/ppagent/tui.py`
- Produces: Validated test suite passing.

- [ ] **Step 1: Update `Latest Models` tests to use `Models`**
  Modify the menu validation tests in `tests/test_tui_config.py` (lines 194-307) to test for the `Models` nav item instead of `Latest Models`.

  Specifically, replace:
  ```python
  def test_latest_models_nav_item_shown_for_every_provider_with_aliases():
      """Every provider with a non-empty latest_models shows 'Latest Models'."""
      cfg = AppConfig()
      expected_with_picker = {spec.key for spec in PROVIDERS if spec.latest_models}
      expected_without = {spec.key for spec in PROVIDERS if not spec.latest_models}

      for vendor_key in expected_with_picker:
          settings = get_menu_definition(f"llm_text_{vendor_key}", cfg)
          assert any(item.label == "Latest Models" for item in settings), (
              f"{vendor_key} should show Latest Models"
          )

      # Providers without a picker must not show the item.
      for vendor_key in expected_without:
          settings = get_menu_definition(f"llm_text_{vendor_key}", cfg)
          assert not any(item.label == "Latest Models" for item in settings), (
              f"{vendor_key} should NOT show Latest Models"
          )


  def test_latest_models_nav_target_is_correct():
      grok_settings = get_menu_definition("llm_text_grok", AppConfig())
      nav = next(item for item in grok_settings if item.label == "Latest Models")
      assert nav.target == "llm_text_grok_latest"

      searcher_settings = get_menu_definition("llm_searcher_openai", AppConfig())
      nav = next(item for item in searcher_settings if item.label == "Latest Models")
      assert nav.target == "llm_searcher_openai_latest"
  ```

  With:
  ```python
  def test_models_nav_item_shown_for_every_provider():
      """Every provider shows 'Models' nav item and does NOT show 'Model Name'."""
      cfg = AppConfig()
      for spec in PROVIDERS:
          settings = get_menu_definition(f"llm_text_{spec.key}", cfg)
          assert any(item.label == "Models" for item in settings), (
              f"{spec.key} should show Models"
          )
          assert not any(item.label == "Model Name" for item in settings), (
              f"{spec.key} should NOT show Model Name"
          )


  def test_models_nav_target_is_correct():
      grok_settings = get_menu_definition("llm_text_grok", AppConfig())
      nav = next(item for item in grok_settings if item.label == "Models")
      assert nav.target == "llm_text_grok_models"

      searcher_settings = get_menu_definition("llm_searcher_openai", AppConfig())
      nav = next(item for item in searcher_settings if item.label == "Models")
      assert nav.target == "llm_searcher_openai_models"
  ```

- [ ] **Step 2: Update all other picker references in tests**
  Change all occurrences of `_latest` menu IDs in tests to `_models` and rename tests to reflect the new `Models` naming.
  
  For example, in:
  - `test_latest_models_picker_lists_vendor_aliases`
  - `test_picker_always_ends_with_custom_model_entry`
  - `test_latest_models_picker_marks_current_model_active`
  - `test_picker_set_value_branch_writes_model_and_is_idempotent`
  - `test_latest_picker_menu_id_not_misrouted_to_vendor_settings`
  
  Change `_latest` to `_models` in targets, menu names, and variables.

- [ ] **Step 3: Run the test suite and verify that all tests pass**
  Run: `pytest tests/test_tui_config.py`
  Expected: PASS
