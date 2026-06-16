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

from ppagent.config import AppConfig, load_config, PROJECT_ROOT, _DEFAULT_CONFIG_PATHS
from ppagent.cli import config_init

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
    ):
        self.label = label
        self.target = target  # Submenu name or action ('save', 'discard', 'back')
        self.key = key        # Configuration key path (if config option)
        self.val_type = val_type
        self.secret = secret
        self.description = description


VENDORS = [
    {"key": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1"},
    {"key": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com"},
    {"key": "mistral", "name": "Mistral", "base_url": "https://api.mistral.ai/v1"},
    {"key": "gemini", "name": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    {"key": "anthropic", "name": "Anthropic", "base_url": "https://api.anthropic.com/v1"},
    {"key": "qwen", "name": "Qwen (Alibaba)", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    {"key": "kimi", "name": "Kimi (Moonshot)", "base_url": "https://api.moonshot.ai/v1"},
    {"key": "glm", "name": "GLM (Zhipu)", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    {"key": "grok", "name": "Grok (xAI)", "base_url": "https://api.x.ai/v1"},
    {"key": "stepfun", "name": "StepFun", "base_url": "https://api.stepfun.ai/v1"},
    {"key": "minimax", "name": "MiniMax", "base_url": "https://api.minimax.io/v1"},
    {"key": "mimo", "name": "MiMo (Xiaomi)", "base_url": "https://api.xiaomimimo.com/v1"},
    {"key": "doubao", "name": "Doubao (ByteDance)", "base_url": "https://ark.cn-beijing.volces.com/api/v3"},
    {"key": "tencent", "name": "Tencent Hunyuan", "base_url": "https://api.hunyuan.cloud.tencent.com"},
    {"key": "custom", "name": "Custom OpenAI Compatible", "base_url": None},
]


def detect_vendor(base_url: str | None) -> str:
    if not base_url:
        return "custom"
    base = base_url.lower()
    if "openai.com" in base:
        return "openai"
    if "deepseek.com" in base or "deepseek" in base:
        return "deepseek"
    if "mistral.ai" in base:
        return "mistral"
    if "googleapis.com" in base or "google" in base:
        return "gemini"
    if "anthropic.com" in base:
        return "anthropic"
    if "dashscope" in base or "aliyuncs.com" in base:
        return "qwen"
    if "moonshot.cn" in base or "kimi.ai" in base or "moonshot.ai" in base:
        return "kimi"
    if "bigmodel.cn" in base or "z.ai" in base:
        return "glm"
    if "x.ai" in base:
        return "grok"
    if "stepfun" in base:
        return "stepfun"
    if "minimax" in base:
        return "minimax"
    if "xiaomimimo.com" in base or "mimo" in base:
        return "mimo"
    if "volces.com" in base or "volcengine.com" in base:
        return "doubao"
    if "hunyuan" in base or "tencent" in base:
        return "tencent"
    return "custom"


def _llm_submenu_items(role: str, vendor_key: str) -> list[MenuItem]:
    """Build the LLM submenu for a given role and vendor.

    If vendor_key is 'custom', 'API Base URL' is displayed and editable.
    Otherwise, it is hidden from the settings page, and is set automatically.
    """
    prefix = f"llms.{role}"
    items = [
        MenuItem("<- Back to Providers", target="back"),
    ]
    if vendor_key == "custom":
        items.append(MenuItem("API Base URL", key=f"{prefix}.base_url", val_type=str, description="Endpoint URL for the LLM API provider."))

    items.extend([
        MenuItem("API Key", key=f"{prefix}.api_key", val_type=str, secret=True, description="Authentication key for the LLM API."),
        MenuItem("Model Name", key=f"{prefix}.model", val_type=str, description="Target model (e.g. gpt-4o, deepseek-chat)."),
        MenuItem("Temperature", key=f"{prefix}.temperature", val_type=float, description="LLM sampling temperature (higher is more creative)."),
        MenuItem("Max Tokens", key=f"{prefix}.max_tokens", val_type=int, description="Max tokens generated in each API response."),
        MenuItem("Timeout", key=f"{prefix}.timeout", val_type=int, description="HTTP request timeout in seconds."),
        MenuItem("Instructor Mode", key=f"{prefix}.instructor_mode", val_type=str, description="Structured output mode: auto, json, tool_call, etc."),
        MenuItem("Enable Thinking", key=f"{prefix}.enable_thinking", val_type=bool, description="Enable extended reasoning (uses model's default thinking budget)."),
    ])
    return items


MENUS: dict[str, list[MenuItem]] = {
    "main": [
        MenuItem("LLM API Settings", target="llms", description="Configure per-role LLM providers: text (writer/finder/criticizer), vision (figure selector), searcher (paper scoring)."),
        MenuItem("Search & Discovery Settings", target="search", description="Configure default date, fetching limits, and relevance matching."),
        MenuItem("Report Settings", target="report", description="Configure report directories, formats, and output language."),
        MenuItem("Scheduler Settings", target="scheduler", description="Configure cron-like background paper discovery times."),
        MenuItem("Publishing Settings", target="publish", description="Configure destinations to share discovered papers (Notion, WeChat, Blog)."),
    ],
    "llms": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem("Text LLM (writer/finder/criticizer)", target="llm_text_vendor", description="LLM used by the writer, finder, and criticizer agents for paper analysis."),
        MenuItem("Vision LLM (figure selector)", target="llm_vision_vendor", description="Vision-capable LLM used by the figure_selector agent to pick pipeline diagrams."),
        MenuItem("Searcher LLM (paper scoring)", target="llm_searcher_vendor", description="LLM used by the searcher agent to score paper relevance to your profile."),
    ],
    "search": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem("Default Date", key="search.default_date", val_type=str, description="Default papers date (YYYY-MM-DD or 'today')."),
        MenuItem("Default Limit", key="search.default_limit", val_type=int, description="Default max number of papers to fetch."),
        MenuItem("Sort Order", key="search.sort", val_type=str, description="ArXiv sorting option (trending, recent)."),
        MenuItem("Profile Path", key="search.profile_path", val_type=str, description="Markdown file containing user research profile/interests."),
        MenuItem("Relevance Threshold", key="search.relevance_threshold", val_type=float, description="Threshold (0.0 to 1.0) above which papers are selected."),
        MenuItem("Max Reports per Run", key="search.max_reports_per_run", val_type=int, description="Limits maximum paper reports created per pipeline run."),
    ],
    "report": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem("Output Directory", key="report.output_dir", val_type=str, description="Folder to save generated reports."),
        MenuItem("Template Directory", key="report.template_dir", val_type=str, description="Folder containing custom Jinja2 templates."),
        MenuItem("Download PDF", key="report.download_pdf", val_type=bool, description="Download and cache paper PDFs locally."),
        MenuItem("PDF Cache Directory", key="report.pdf_cache_dir", val_type=str, description="Folder where downloaded PDFs are cached."),
        MenuItem("Language", key="report.language", val_type=str, description="Language to write the paper summaries/reports in."),
    ],
    "scheduler": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem("Enabled", key="scheduler.enabled", val_type=bool, description="Run discovery pipeline on a schedule."),
        MenuItem("Cron Hour", key="scheduler.cron_hour", val_type=int, description="Hour of the day to execute (0-23)."),
        MenuItem("Cron Minute", key="scheduler.cron_minute", val_type=int, description="Minute of the hour to execute (0-59)."),
        MenuItem("Timezone", key="scheduler.timezone", val_type=str, description="Timezone to use for the scheduler schedule."),
    ],
    "publish": [
        MenuItem("<- Back to Main Menu", target="back"),
        MenuItem("Global Enabled", key="publish.enabled", val_type=bool, description="Enable paper publishing globally."),
        MenuItem("Notion Integration Settings", target="publish_notion", description="Configure publishing to a Notion database."),
        MenuItem("WeChat Integration Settings", target="publish_wechat", description="Configure publishing to a WeChat Official Account."),
        MenuItem("Blog/Webhook Integration Settings", target="publish_blog", description="Configure publishing to a custom blog/webhook."),
    ],
    "publish_notion": [
        MenuItem("<- Back to Publishing Settings", target="back"),
        MenuItem("Enabled", key="publish.notion.enabled", val_type=bool, description="Publish reports to Notion."),
        MenuItem("API Key", key="publish.notion.api_key", val_type=str, secret=True, description="Notion Integration Secret Token."),
        MenuItem("Database ID", key="publish.notion.database_id", val_type=str, description="Notion Database ID to insert pages into."),
    ],
    "publish_wechat": [
        MenuItem("<- Back to Publishing Settings", target="back"),
        MenuItem("Enabled", key="publish.wechat.enabled", val_type=bool, description="Publish reports to WeChat."),
        MenuItem("App ID", key="publish.wechat.appid", val_type=str, description="WeChat Official Account App ID."),
        MenuItem("Secret", key="publish.wechat.secret", val_type=str, secret=True, description="WeChat Official Account App Secret."),
    ],
    "publish_blog": [
        MenuItem("<- Back to Publishing Settings", target="back"),
        MenuItem("Enabled", key="publish.blog.enabled", val_type=bool, description="Publish reports to a custom blog/webhook."),
        MenuItem("Webhook URL", key="publish.blog.webhook_url", val_type=str, description="Target endpoint URL to send report data."),
        MenuItem("API Key", key="publish.blog.api_key", val_type=str, secret=True, description="Authorization API token for the blog endpoint."),
    ],
}


def get_menu_definition(menu_id: str, cfg: AppConfig) -> list[MenuItem]:
    if menu_id in MENUS:
        return MENUS[menu_id]

    # Check if vendor list menu
    # e.g., "llm_text_vendor", "llm_vision_vendor", "llm_searcher_vendor"
    vendor_list_match = re.match(r"^llm_(text|vision|searcher)_vendor$", menu_id)
    if vendor_list_match:
        role = vendor_list_match.group(1)
        role_label = {
            "text": "Text LLM",
            "vision": "Vision LLM",
            "searcher": "Searcher LLM"
        }[role]

        current_base_url = get_config_value(cfg, f"llms.{role}.base_url")
        active_vendor = detect_vendor(current_base_url)

        items = [
            MenuItem(f"<- Back to LLM Roles", target="back"),
        ]
        for v in VENDORS:
            is_active = (v["key"] == active_vendor)
            label = v["name"]
            if is_active:
                label = f"{label} [bold green](Active)[/bold green]"
            items.append(
                MenuItem(
                    label=label,
                    target=f"llm_{role}_{v['key']}",
                    description=f"Configure {v['name']} settings for {role_label}."
                )
            )
        return items

    # Check if specific vendor setting menu
    # e.g., "llm_text_openai"
    vendor_setting_match = re.match(r"^llm_(text|vision|searcher)_([a-z0-9_]+)$", menu_id)
    if vendor_setting_match:
        role, vendor_key = vendor_setting_match.groups()
        return _llm_submenu_items(role, vendor_key)

    raise KeyError(f"Menu '{menu_id}' not found.")



def get_config_path() -> Path:
    """Find current settings.toml path or return default."""
    for candidate in _DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "config" / "settings.toml"


def get_config_value(cfg: AppConfig, key_path: str) -> Any:
    """Get nested attribute of AppConfig by path string (e.g. 'llm.api_key')."""
    parts = key_path.split(".")
    val = cfg
    for p in parts:
        val = getattr(val, p)
    return val


def set_config_value(cfg: AppConfig, key_path: str, new_val: Any) -> None:
    """Set nested attribute of AppConfig by path string (e.g. 'llm.api_key')."""
    parts = key_path.split(".")
    val = cfg
    for p in parts[:-1]:
        val = getattr(val, p)
    setattr(val, parts[-1], new_val)


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
        if item.val_type is bool:
            val_str = "[bold green]ON[/bold green]" if current_val else "[bold red]OFF[/bold red]"
        elif item.secret:
            val_str = "[yellow]••••••••[/yellow]" if current_val else "[dim](not set)[/dim]"
        else:
            val_str = f"[cyan]{current_val}[/cyan]" if (current_val not in (None, "")) else "[dim](empty)[/dim]"
        
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
        (f"  ({get_config_path().relative_to(PROJECT_ROOT)})", "dim")
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
            f"[bold yellow]Info:[/bold yellow] {desc}\n[dim]Controls: [↑/↓] Navigate  [Enter] Select/Toggle  [←] Back  [q] Save & Quit  [x] Discard & Quit[/dim]",
            border_style="dim blue",
            title="Help & Controls"
        )
    )
    
    outer_panel = Panel(
        content,
        title=title,
        border_style="purple",
        title_align="left",
        expand=True
    )
    return outer_panel


def get_key() -> str:
    """Read a keypress in raw mode on macOS/Linux."""
    if not sys.stdin.isatty():
        char = sys.stdin.read(1)
        if char == '\n':
            return 'enter'
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
        if b == b'\x03':  # Ctrl+C
            raise KeyboardInterrupt
        if b == b'\x1b':
            # Check if more bytes are available (escape sequence)
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                b2 = os.read(fd, 1)
                if b2 == b'[':
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if r:
                        b3 = os.read(fd, 1)
                        if b3 == b'A':
                            return 'up'
                        elif b3 == b'B':
                            return 'down'
                        elif b3 == b'C':
                            return 'right'
                        elif b3 == b'D':
                            return 'left'
            return 'esc'
        elif b in (b'\r', b'\n'):
            return 'enter'
        elif b in (b'\x7f', b'\x08'):
            return 'backspace'
        return b.decode('utf-8', errors='ignore')
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
            console.print(f"[red]Invalid value for type {val_type.__name__}. Please try again.[/red]")


def save_config(cfg: AppConfig, path: Path) -> None:
    """Save the current configuration back to settings.toml."""
    import tomli_w
    data = cfg.model_dump()
    data.pop("root", None)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def run_config_tui() -> None:
    """Run the interactive TUI configuration loop."""
    config_file = get_config_path()
    if not config_file.exists():
        console.print("[yellow]Configuration file settings.toml not found. Initializing default config...[/yellow]")
        config_init()
        
    try:
        cfg = load_config()
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}")
        raise typer.Exit(1)
        
    # Navigation stack stores tuples of (menu_id, selected_idx)
    menu_stack: list[tuple[str, int]] = [("main", 0)]
    
    with Live(make_ui("main", 0, cfg), console=console, screen=True, auto_refresh=False) as live:
        while True:
            menu_id, selected_idx = menu_stack[-1]
            menu_def = get_menu_definition(menu_id, cfg)
            
            live.update(make_ui(menu_id, selected_idx, cfg), refresh=True)
            
            try:
                key = get_key()
            except KeyboardInterrupt:
                break
                
            if key == 'up' or key == 'k':
                selected_idx = (selected_idx - 1) % len(menu_def)
                menu_stack[-1] = (menu_id, selected_idx)
            elif key == 'down' or key == 'j':
                selected_idx = (selected_idx + 1) % len(menu_def)
                menu_stack[-1] = (menu_id, selected_idx)
            elif key == 'left' or key == 'esc':
                if len(menu_stack) > 1:
                    menu_stack.pop()
            elif key == 'q':
                save_config(cfg, config_file)
                live.stop()
                console.print(f"[bold green]Configuration saved successfully to {config_file.relative_to(PROJECT_ROOT)}[/bold green]")
                return
            elif key == 'x':
                live.stop()
                console.print("[yellow]Changes discarded.[/yellow]")
                return
            elif key == 'enter':
                item = menu_def[selected_idx]
                if item.target:
                    if item.target == 'back':
                        if len(menu_stack) > 1:
                            menu_stack.pop()
                    else:
                        # Set default base URL if selecting a specific vendor
                        match = re.match(r"^llm_(text|vision|searcher)_([a-z0-9_]+)$", item.target)
                        if match:
                            role, vendor_key = match.groups()
                            if vendor_key != "custom":
                                for v in VENDORS:
                                    if v["key"] == vendor_key:
                                        set_config_value(cfg, f"llms.{role}.base_url", v["base_url"])
                                        break
                        menu_stack.append((item.target, 0))
                elif item.key:
                    if item.val_type is bool:
                        toggle_config_value(cfg, item.key)
                    else:
                        live.stop()
                        curr_val = get_config_value(cfg, item.key)
                        new_val = edit_setting(item.label, curr_val, item.val_type)
                        set_config_value(cfg, item.key, new_val)
                        live.start()
    
    # If the loop ended via break/escape on main menu/Ctrl+C
    console.print("[yellow]TUI config session closed. Changes not saved.[/yellow]")
