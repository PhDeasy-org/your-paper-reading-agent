"""Interactive TUI configuration menu for ppagent."""

from __future__ import annotations

import re
import select
import sys
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

import typer

from ppagent.config import (
    AppConfig,
    LLMConfig,
    load_config,
    CONFIG_PATH,
    _LLM_ROLES,
)
from ppagent.providers import PROVIDERS, detect_provider, get_provider

console = Console()


class MenuItem:
    def __init__(
        self,
        label: str,
        target: str | None = None,
        key: str | None = None,
        val_type: type = str,
        secret: bool = False,
        description: str = "",
        set_value: Any | None = None,
    ):
        self.label = label
        self.target = target  # Submenu name or action ('back', etc.)
        self.key = key  # Configuration key path (if config option)
        self.val_type = val_type
        self.secret = secret
        self.description = description
        # When set, this item is a picker option: activating it writes
        # ``set_value`` to ``key`` and returns to the previous menu. Used by
        # the "-latest" model picker for Grok/Gemini.
        self.set_value = set_value


def _vendor_default_model(vendor_key: str) -> str:
    """Return the default model name for a vendor key (empty for custom)."""
    spec = get_provider(vendor_key)
    return spec.default_model if spec else ""


def _snapshot_active_vendor(cfg: AppConfig, role: str) -> str:
    """Persist the role's current live LLMConfig under its detected vendor key.

    Returns the vendor key the active config was saved under. This lets the user
    switch to a different provider and later switch back without losing the
    previously entered api_key/model/etc.
    """
    import copy

    active: LLMConfig = getattr(cfg.llms, role)
    vendor_key = detect_provider(active.base_url)
    cfg.llms.saved_vendors.setdefault(role, {})[vendor_key] = copy.deepcopy(active)
    return vendor_key


def _shared_api_key(cfg: AppConfig, vendor_key: str) -> str | None:
    """Return a non-blank api_key for ``vendor_key`` from any role, if known.

    API keys are per *provider*, not per role — the same OpenAI key works for
    text/searcher alike. So when the user fills in a key in one place,
    every other role visiting that provider can reuse it instead of asking for
    the key again.
    """
    for role in _LLM_ROLES:
        active = getattr(cfg.llms, role)
        if detect_provider(active.base_url) == vendor_key and active.api_key:
            return active.api_key
        snapshot = cfg.llms.saved_vendors.get(role, {}).get(vendor_key)
        if snapshot is not None and snapshot.api_key:
            return snapshot.api_key
    return None


def _propagate_api_key(
    cfg: AppConfig, role: str, vendor_key: str, api_key: str
) -> None:
    """Mirror an api_key change into every role's live config + snapshot.

    When the user edits the api_key for ``(role, vendor_key)``, copy the value
    into every other role whose live config or saved snapshot uses the same
    provider. This keeps sibling roles (e.g. text and searcher both on OpenAI) in
    lockstep so a key entered once is immediately available everywhere.
    """
    for other_role in _LLM_ROLES:
        if other_role == role:
            continue
        other = getattr(cfg.llms, other_role)
        if detect_provider(other.base_url) == vendor_key:
            other.api_key = api_key
        snapshot = cfg.llms.saved_vendors.get(other_role, {}).get(vendor_key)
        if snapshot is not None:
            snapshot.api_key = api_key


def _switch_vendor(cfg: AppConfig, role: str, vendor_key: str) -> None:
    """Make ``vendor_key`` the active provider for ``role``.

    - Snapshot the currently-active config first (so its edits are preserved).
    - If the requested vendor is already the active one, do nothing (re-entering
      the same provider's settings page must not wipe anything).
    - Otherwise, load that vendor's saved config if one exists; if not, seed a
      fresh LLMConfig with the vendor's ``base_url``/``default_model`` and blank
      credentials so the user only fills in the fields that matter. If some
      other role already has an api_key for this provider, reuse it instead of
      asking the user to type the same key again.
    """
    import copy

    active = getattr(cfg.llms, role)
    active_vendor = detect_provider(active.base_url)
    if active_vendor == vendor_key:
        return  # already active — no-op to avoid clobbering in-progress edits

    # Preserve what's currently in the live slot.
    cfg.llms.saved_vendors.setdefault(role, {})[active_vendor] = copy.deepcopy(active)

    saved = cfg.llms.saved_vendors.get(role, {}).get(vendor_key)
    if saved is not None:
        new_cfg = copy.deepcopy(saved)
    else:
        # First time visiting this vendor: fresh defaults, blank creds.
        spec = get_provider(vendor_key)
        base = spec.base_url if spec else None
        new_cfg = LLMConfig(
            base_url=base or "",
            api_key="",
            model=_vendor_default_model(vendor_key) or active.model,
        )

    # API keys are shared per provider: if a sibling role already holds a key
    # for this vendor, inherit it so the user isn't asked to type it twice.
    if not new_cfg.api_key:
        inherited = _shared_api_key(cfg, vendor_key)
        if inherited:
            new_cfg.api_key = inherited

    setattr(cfg.llms, role, new_cfg)


def _llm_submenu_items(role: str, vendor_key: str) -> list[MenuItem]:
    """Build the LLM submenu for a given role and vendor.

    If vendor_key is 'custom', 'API Base URL' is displayed and editable.
    Otherwise, it is hidden from the settings page, and is set automatically.
    """
    prefix = f"llms.{role}"
    spec = get_provider(vendor_key)
    items = [
        MenuItem("<- Back to Providers", target="back"),
    ]
    if vendor_key == "custom":
        items.append(
            MenuItem(
                "API Base URL",
                key=f"{prefix}.base_url",
                val_type=str,
                description="Endpoint URL for the LLM API provider.",
            )
        )

    items.append(
        MenuItem(
            "API Key",
            key=f"{prefix}.api_key",
            val_type=str,
            secret=True,
            description="Authentication key for the LLM API.",
        )
    )
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

    items.extend(
        [
            MenuItem(
                "Temperature",
                key=f"{prefix}.temperature",
                val_type=float,
                description="LLM sampling temperature (higher is more creative).",
            ),
            MenuItem(
                "Max Tokens",
                key=f"{prefix}.max_tokens",
                val_type=int,
                description="Max tokens generated in each API response.",
            ),
            MenuItem(
                "Timeout",
                key=f"{prefix}.timeout",
                val_type=int,
                description="HTTP request timeout in seconds.",
            ),
            MenuItem(
                "Instructor Mode",
                key=f"{prefix}.instructor_mode",
                val_type=str,
                description="Structured output mode: auto, json, tool_call, etc.",
            ),
            MenuItem(
                "Enable Thinking",
                key=f"{prefix}.enable_thinking",
                val_type=bool,
                description="Enable extended reasoning (uses model's default thinking budget).",
            ),
        ]
    )
    return items


MENUS: dict[str, list[MenuItem]] = {
    "main": [
        MenuItem(
            "LLM API Settings",
            target="llms",
            description="Configure per-role LLM providers: text (writer/finder/criticizer), searcher (paper scoring).",
        ),
        MenuItem(
            "Search & Discovery Settings",
            target="search",
            description="Configure default date, fetching limits, and relevance matching.",
        ),
        MenuItem(
            "Report Settings",
            target="report",
            description="Configure report directories, formats, and output language.",
        ),
        MenuItem(
            "Scheduler Settings",
            target="scheduler",
            description="Configure cron-like background paper discovery times.",
        ),
        MenuItem(
            "Publishing Settings",
            target="publish",
            description="Configure destinations to share discovered papers (Notion, WeChat, GitHub Pages).",
        ),
    ],
    "llms": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem(
            "Text LLM (writer/finder/criticizer)",
            target="llm_text_vendor",
            description="LLM used by the writer, finder, and criticizer agents for paper analysis.",
        ),
        MenuItem(
            "Searcher LLM (paper scoring)",
            target="llm_searcher_vendor",
            description="LLM used by the searcher agent to score paper relevance to your profile.",
        ),
    ],
    "search": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem(
            "Default Date",
            key="search.default_date",
            val_type=str,
            description="Default papers date (YYYY-MM-DD or 'today').",
        ),
        MenuItem(
            "Default Limit",
            key="search.default_limit",
            val_type=int,
            description="Default max number of papers to fetch.",
        ),
        MenuItem(
            "Sort Order",
            key="search.sort",
            val_type=str,
            description="ArXiv sorting option (trending, recent).",
        ),
        MenuItem(
            "Profile Path",
            key="search.profile_path",
            val_type=str,
            description="Markdown file containing user research profile/interests.",
        ),
        MenuItem(
            "Relevance Threshold",
            key="search.relevance_threshold",
            val_type=float,
            description="Threshold (0.0 to 1.0) above which papers are selected.",
        ),
        MenuItem(
            "Max Reports per Run",
            key="search.max_reports_per_run",
            val_type=int,
            description="Limits maximum paper reports created per pipeline run.",
        ),
    ],
    "report": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem(
            "Output Directory",
            key="report.output_dir",
            val_type=str,
            description="Folder to save generated reports.",
        ),
        MenuItem(
            "Template Directory",
            key="report.template_dir",
            val_type=str,
            description="Folder containing custom Jinja2 templates.",
        ),
        MenuItem(
            "Download PDF",
            key="report.download_pdf",
            val_type=bool,
            description="Download and cache paper PDFs locally.",
        ),
        MenuItem(
            "PDF Cache Directory",
            key="report.pdf_cache_dir",
            val_type=str,
            description="Folder where downloaded PDFs are cached.",
        ),
        MenuItem(
            "Language",
            key="report.language",
            val_type=str,
            description="Language to write the paper summaries/reports in.",
        ),
    ],
    "scheduler": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem(
            "Enabled",
            key="scheduler.enabled",
            val_type=bool,
            description="Run discovery pipeline on a schedule.",
        ),
        MenuItem(
            "Cron Hour",
            key="scheduler.cron_hour",
            val_type=int,
            description="Hour of the day to execute (0-23).",
        ),
        MenuItem(
            "Cron Minute",
            key="scheduler.cron_minute",
            val_type=int,
            description="Minute of the hour to execute (0-59).",
        ),
        MenuItem(
            "Timezone",
            key="scheduler.timezone",
            val_type=str,
            description="Timezone to use for the scheduler schedule.",
        ),
    ],
    "publish": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem(
            "Global Enabled",
            key="publish.enabled",
            val_type=bool,
            description="Enable paper publishing globally.",
        ),
        MenuItem(
            "Notion Integration Settings",
            target="publish_notion",
            description="Configure publishing to a Notion database.",
        ),
        MenuItem(
            "WeChat Integration Settings",
            target="publish_wechat",
            description="Configure publishing to a WeChat Official Account.",
        ),
        MenuItem(
            "GitHub Pages Blog Settings",
            target="publish_github_pages",
            description="Configure publishing to a GitHub Pages blog repo.",
        ),
    ],
    "publish_notion": [
        MenuItem("<- Back to Publishing Settings", target="back"),
        MenuItem(
            "Enabled",
            key="publish.notion.enabled",
            val_type=bool,
            description="Publish reports to Notion.",
        ),
        MenuItem(
            "API Key",
            key="publish.notion.api_key",
            val_type=str,
            secret=True,
            description="Notion Integration Secret Token.",
        ),
        MenuItem(
            "Database ID",
            key="publish.notion.database_id",
            val_type=str,
            description="Notion Database ID to insert pages into.",
        ),
    ],
    "publish_wechat": [
        MenuItem("<- Back to Publishing Settings", target="back"),
        MenuItem(
            "Enabled",
            key="publish.wechat.enabled",
            val_type=bool,
            description="Publish reports to WeChat.",
        ),
        MenuItem(
            "App ID",
            key="publish.wechat.appid",
            val_type=str,
            description="WeChat Official Account App ID.",
        ),
        MenuItem(
            "Secret",
            key="publish.wechat.secret",
            val_type=str,
            secret=True,
            description="WeChat Official Account App Secret.",
        ),
    ],
    "publish_github_pages": [
        MenuItem("<- Back to Publishing Settings", target="back"),
        MenuItem(
            "Enabled",
            key="publish.github_pages.enabled",
            val_type=bool,
            description="Publish reports to a GitHub Pages blog.",
        ),
        MenuItem(
            "GitHub Username",
            key="publish.github_pages.username",
            val_type=str,
            description="GitHub username that owns the Pages repo.",
        ),
        MenuItem(
            "Repo Name",
            key="publish.github_pages.repo",
            val_type=str,
            description="Name of the GitHub Pages repository (e.g. my-paper-blog).",
        ),
        MenuItem(
            "Local Repo Path",
            key="publish.github_pages.repo_path",
            val_type=str,
            description="Local working-copy path of the cloned Pages repo (must have push access).",
        ),
        MenuItem(
            "Branch",
            key="publish.github_pages.branch",
            val_type=str,
            description="Branch GitHub Pages serves from (push target).",
        ),
        MenuItem(
            "Posts Subdirectory",
            key="publish.github_pages.posts_subdir",
            val_type=str,
            description="Subfolder inside the repo where report directories are written.",
        ),
    ],
}


def get_menu_definition(menu_id: str, cfg: AppConfig) -> list[MenuItem]:
    if menu_id in MENUS:
        return MENUS[menu_id]

    # Check if vendor list menu
    # e.g., "llm_text_vendor", "llm_searcher_vendor"
    vendor_list_match = re.match(r"^llm_(text|searcher)_vendor$", menu_id)
    if vendor_list_match:
        role = vendor_list_match.group(1)
        role_label = {
            "text": "Text LLM",
            "searcher": "Searcher LLM",
        }[role]

        current_base_url = get_config_value(cfg, f"llms.{role}.base_url")
        active_vendor = detect_provider(current_base_url)

        items = [
            MenuItem("<- Back to LLM Roles", target="back"),
        ]
        for spec in PROVIDERS:
            if role == "searcher" and spec.key not in (
                "openai",
                "qwen",
                "doubao",
                "grok",
            ):
                continue
            is_active = spec.key == active_vendor
            label = spec.name
            if is_active:
                label = f"{label} [bold green](Active)[/bold green]"
            items.append(
                MenuItem(
                    label=label,
                    target=f"llm_{role}_{spec.key}",
                    description=f"Configure {spec.name} settings for {role_label}.",
                )
            )
        return items

    # Check if "-latest" model picker menu.
    # e.g., "llm_text_grok_latest", "llm_searcher_gemini_latest"
    # Must be matched BEFORE the generic vendor-setting regex below, otherwise
    # its "_latest" suffix gets swallowed by the vendor_key capture group
    # (e.g. "grok_latest").
    picker_match = re.match(r"^llm_(text|searcher)_([a-z0-9_]+)_latest$", menu_id)
    if picker_match:
        role, vendor_key = picker_match.groups()
        spec = get_provider(vendor_key)
        current_model = get_config_value(cfg, f"llms.{role}.model")
        items = [
            MenuItem("<- Back to Provider Settings", target="back"),
        ]
        if spec is not None:
            for model in spec.latest_models:
                label = model
                if model == current_model:
                    label = f"{label} [bold green](Active)[/bold green]"
                items.append(
                    MenuItem(
                        label=label,
                        key=f"llms.{role}.model",
                        set_value=model,
                        description=f"Use {model} (auto-routes to the newest "
                        f"{spec.name} {model.rsplit('-latest', 1)[0]} version).",
                    )
                )
        # Escape hatch: type any model name not listed above. Has a target
        # (not set_value) so it routes through the free-text edit path.
        items.append(
            MenuItem(
                label="Custom Model (type your own)...",
                key=f"llms.{role}.model",
                target=f"llm_{role}_{vendor_key}_custom_model",
                description="Enter any model name by hand (for models not in "
                "the list above).",
            )
        )
        return items

    # Check if specific vendor setting menu
    # e.g., "llm_text_openai"
    vendor_setting_match = re.match(r"^llm_(text|searcher)_([a-z0-9_]+)$", menu_id)
    if vendor_setting_match:
        role, vendor_key = vendor_setting_match.groups()
        return _llm_submenu_items(role, vendor_key)

    raise KeyError(f"Menu '{menu_id}' not found.")


def get_config_path() -> Path:
    """The single source-of-truth settings.toml path."""
    return CONFIG_PATH


def get_config_value(cfg: AppConfig, key_path: str) -> Any:
    """Get nested attribute of AppConfig by path string (e.g. 'llm.api_key')."""
    parts = key_path.split(".")
    val = cfg
    for p in parts:
        val = getattr(val, p)
    return val


def set_config_value(cfg: AppConfig, key_path: str, new_val: Any) -> None:
    """Set nested attribute of AppConfig by path string (e.g. 'llm.api_key').

    For ``llms.<role>.api_key``, the new value is propagated to every sibling
    role whose active provider matches: API keys are per-provider, not per-role,
    so entering the key once (e.g. in "text") makes it appear immediately in
    "text"/"searcher" when they use the same provider.
    """
    parts = key_path.split(".")
    val = cfg
    for p in parts[:-1]:
        val = getattr(val, p)
    setattr(val, parts[-1], new_val)

    if (
        len(parts) == 3
        and parts[0] == "llms"
        and parts[1] in _LLM_ROLES
        and parts[2] == "api_key"
    ):
        role = parts[1]
        vendor_key = detect_provider(getattr(cfg.llms, role).base_url)
        _propagate_api_key(cfg, role, vendor_key, new_val)


def toggle_config_value(cfg: AppConfig, key_path: str) -> None:
    """Toggle a boolean value in configuration."""
    current = get_config_value(cfg, key_path)
    set_config_value(cfg, key_path, not current)


def strip_markup(text: str) -> str:
    """Strip Rich/bbcode-like markup tags."""
    return re.sub(r"\[/?.*?\]", "", text)


def format_menu_item(item: MenuItem, current_val: Any, is_selected: bool) -> str:
    """Format a menu item into a colorized string."""
    marker = "▶ " if is_selected else "  "

    if is_selected:
        label_part = f"[bold cyan]{item.label}[/bold cyan]"
    else:
        label_part = item.label

    if item.key:
        # Action / navigation items that also carry a key (e.g. the "Custom
        # Model (type your own)" picker entry) render label-only: showing the
        # underlying value would wrongly imply this item *is* that value.
        if item.set_value is not None or item.target is not None:
            return f"{marker}{label_part}"

        if item.val_type is bool:
            val_str = (
                "[bold green]ON[/bold green]"
                if current_val
                else "[bold red]OFF[/bold red]"
            )
        elif item.secret:
            val_str = (
                "[yellow]••••••••[/yellow]" if current_val else "[dim](not set)[/dim]"
            )
        else:
            val_str = (
                f"[cyan]{current_val}[/cyan]"
                if (current_val not in (None, ""))
                else "[dim](empty)[/dim]"
            )

        plain_label = strip_markup(item.label)
        padding = " " * max(0, 32 - len(plain_label))
        return f"{marker}{label_part}{padding} : {val_str}"
    else:
        return f"{marker}{label_part}"


def make_ui(menu_id: str, selected_idx: int, cfg: AppConfig) -> Panel:
    """Build the UI component hierarchy for Rich."""
    menu_def = get_menu_definition(menu_id, cfg)

    title = Text.assemble(
        ("⚙  ", "bold purple"),
        ("ppagent Config Manager", "bold white"),
        (f"  ({get_config_path()})", "dim"),
    )

    options_group = []
    for i, item in enumerate(menu_def):
        current_val = get_config_value(cfg, item.key) if item.key else None
        line = format_menu_item(item, current_val, i == selected_idx)
        options_group.append(line)

    body_content = "\n".join(options_group)

    selected_item = menu_def[selected_idx]
    desc = selected_item.description

    content = Group(
        Panel(body_content, border_style="cyan", title="Navigation Menu"),
        Panel(
            f"[bold yellow]Info:[/bold yellow] {desc}\n[dim]Controls: \\[↑/↓] Navigate  \\[Enter] Navigate/Toggle  \\[Space] Select  \\[←] Back  \\[q] Save & Quit[/dim]",
            border_style="dim blue",
            title="Help & Controls",
        ),
    )

    outer_panel = Panel(
        content, title=title, border_style="purple", title_align="left", expand=True
    )
    return outer_panel


def get_key() -> str:
    """Read a keypress in raw mode on macOS/Linux."""
    if not sys.stdin.isatty():
        char = sys.stdin.read(1)
        if char == "\n":
            return "enter"
        if char == " ":
            return "space"
        return char

    import tty
    import termios
    import os

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Read first byte
        b = os.read(fd, 1)
        if b == b"\x03":  # Ctrl+C
            raise KeyboardInterrupt
        if b == b"\x1b":
            # Check if more bytes are available (escape sequence)
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                b2 = os.read(fd, 1)
                if b2 == b"[":
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if r:
                        b3 = os.read(fd, 1)
                        if b3 == b"A":
                            return "up"
                        elif b3 == b"B":
                            return "down"
                        elif b3 == b"C":
                            return "right"
                        elif b3 == b"D":
                            return "left"
            return "esc"
        elif b == b" ":
            return "space"
        elif b in (b"\r", b"\n"):
            return "enter"
        elif b in (b"\x7f", b"\x08"):
            return "backspace"
        return b.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def edit_setting(name: str, current_value: Any, val_type: type) -> Any:
    """Prompt the user for a new setting value, handling conversions."""
    console.print(f"\n[bold]Editing {name}[/bold]")
    curr_str = "" if current_value is None else str(current_value)
    console.print(f"Current value: [cyan]{curr_str or '(empty)'}[/cyan]")

    prompt_str = "Enter new value"

    while True:
        try:
            val_str = Prompt.ask(prompt_str, default=curr_str, show_default=False)
            if val_str == curr_str:
                return current_value

            # Type conversion
            if val_type is int:
                return int(val_str)
            elif val_type is float:
                return float(val_str)
            else:
                return val_str
        except KeyboardInterrupt:
            console.print("\n[yellow]Editing cancelled.[/yellow]")
            return current_value
        except ValueError:
            console.print(
                f"[red]Invalid value for type {val_type.__name__}. Please try again.[/red]"
            )


def save_config(cfg: AppConfig, path: Path) -> None:
    """Save the current configuration back to ``path``.

    Before dumping, snapshot each role's currently-active LLMConfig into
    ``saved_vendors`` so the most recent edits are preserved for next time the
    user re-enters that provider's page. ``path`` is the single source of
    truth (``~/.config/ppagent/settings.toml``) — there is no separate backup.
    """
    import tomli_w

    for role in _LLM_ROLES:
        _snapshot_active_vendor(cfg, role)
    data = cfg.model_dump()
    data.pop("root", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def run_config_tui() -> None:
    """Run the interactive TUI configuration loop."""
    # Config lives at a single source-of-truth path (~/.config/ppagent/
    # settings.toml). Load from and save back to that path directly — no
    # separate backup, no project-file mirror.
    try:
        cfg = load_config(CONFIG_PATH)
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}")
        raise typer.Exit(1)

    config_file = CONFIG_PATH

    # Navigation stack stores tuples of (menu_id, selected_idx)
    menu_stack: list[tuple[str, int]] = [("main", 0)]

    with Live(
        make_ui("main", 0, cfg), console=console, screen=True, auto_refresh=False
    ) as live:
        while True:
            menu_id, selected_idx = menu_stack[-1]
            menu_def = get_menu_definition(menu_id, cfg)

            live.update(make_ui(menu_id, selected_idx, cfg), refresh=True)

            try:
                key = get_key()
            except KeyboardInterrupt:
                # Ctrl+C always saves & quits — there is no "discard" path.
                break

            if key == "up" or key == "k":
                selected_idx = (selected_idx - 1) % len(menu_def)
                menu_stack[-1] = (menu_id, selected_idx)
            elif key == "down" or key == "j":
                selected_idx = (selected_idx + 1) % len(menu_def)
                menu_stack[-1] = (menu_id, selected_idx)
            elif key == "left" or key == "esc":
                if len(menu_stack) > 1:
                    menu_stack.pop()
            elif key == "q":
                # The only way out — always persist changes.
                save_config(cfg, config_file)
                live.stop()
                console.print(
                    f"[bold green]Configuration saved successfully to {config_file}[/bold green]"
                )
                return
            elif key in ("enter", "space"):
                item = menu_def[selected_idx]
                if item.target:
                    if item.target == "back":
                        if len(menu_stack) > 1:
                            menu_stack.pop()
                    # The "-latest" model picker is a submenu of an *already
                    # active* vendor — navigate in without calling
                    # _switch_vendor, which would corrupt state by treating
                    # the "_latest" suffix as a vendor key. Must be checked
                    # before the vendor-switch regex below, which would
                    # otherwise match (e.g. "grok_latest").
                    elif re.match(
                        r"^llm_(text|searcher)_[a-z0-9_]+_latest$",
                        item.target,
                    ):
                        menu_stack.append((item.target, 0))
                    # "Custom Model (type your own)" picker entry: open the
                    # free-text edit prompt, then stay on the picker menu so
                    # the typed value is reflected via the Active marker.
                    # Checked before the vendor-switch regex for the same
                    # "_custom_model" suffix-swallowing reason as _latest.
                    elif re.match(
                        r"^llm_(text|searcher)_[a-z0-9_]+_custom_model$",
                        item.target,
                    ):
                        live.stop()
                        curr_val = get_config_value(cfg, item.key or "")
                        new_val = edit_setting(item.label, curr_val, str)
                        set_config_value(cfg, item.key or "", new_val)
                        live.start()
                    else:
                        # Detect vendor setting item (e.g. llm_text_openai),
                        # but NOT vendor *list* items (e.g. llm_text_vendor).
                        match = re.match(
                            r"^llm_(text|searcher)_(?!vendor$)([a-z0-9_]+)$",
                            item.target,
                        )
                        if match:
                            role, vendor_key = match.groups()
                            if key == "space":
                                # Space = select this vendor (switch but stay on list)
                                _switch_vendor(cfg, role, vendor_key)
                            else:
                                # Enter = navigate into the vendor's settings submenu
                                _switch_vendor(cfg, role, vendor_key)
                                menu_stack.append((item.target, 0))
                        else:
                            menu_stack.append((item.target, 0))
                elif item.key:
                    if item.set_value is not None:
                        # Picker option (e.g. "-latest" model alias): write the
                        # literal value and return to the previous menu so the
                        # updated "Model Name" field is visible.
                        set_config_value(cfg, item.key, item.set_value)
                        if len(menu_stack) > 1:
                            menu_stack.pop()
                    elif item.val_type is bool:
                        toggle_config_value(cfg, item.key)
                    else:
                        live.stop()
                        curr_val = get_config_value(cfg, item.key)
                        new_val = edit_setting(item.label, curr_val, item.val_type)
                        set_config_value(cfg, item.key, new_val)
                        live.start()

    # Reached only via Ctrl+C: every exit persists — there is no discard path.
    save_config(cfg, config_file)
    console.print(
        f"[bold green]Configuration saved successfully to {config_file}[/bold green]"
    )
