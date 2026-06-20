"""Configuration models and loading for ppagent."""

from __future__ import annotations

import logging
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project root — where pyproject.toml lives
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_CONFIG_PATHS = [
    PROJECT_ROOT / "config" / "settings.toml",
    Path.home() / ".config" / "ppagent" / "settings.toml",
]

# Persistent backup location — survives project directory reinstalls.
# save_config() writes a copy here on every TUI save; load_config() falls back
# to this file when no project-level settings.toml exists.
_BACKUP_CONFIG_PATH = Path.home() / ".config" / "ppagent" / "settings.toml"

# One-generation safety net. On every save, the *previous* backup is copied
# here before being overwritten, so a single bad overwrite remains undoable.
_BACKUP_PATH_BAK = _BACKUP_CONFIG_PATH.with_suffix(".toml.bak")


class LLMConfig(BaseModel):
    """LLM API configuration."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120  # seconds
    instructor_mode: str = "auto"
    enable_thinking: bool = False


# Maps each agent name to the LLM role it should use.
# writer/finder/criticizer share the "text" role; figure_selector needs vision;
# searcher is isolated so paper scoring can use a cheaper/faster model.
AGENT_LLM_ROLE: dict[str, str] = {
    "classifier": "text",
    "writer": "text",
    "finder": "searcher",
    "criticizer": "text",
    "figure_selector": "vision",
    "searcher": "searcher",
}


def _vision_default() -> LLMConfig:
    """Default LLM config for the vision role (a vision-capable model)."""
    return LLMConfig(model="gpt-4o")


class LLMsConfig(BaseModel):
    """Per-role LLM configurations.

    - ``text``: agents that reason over paper text (writer, finder, criticizer).
    - ``vision``: the figure_selector agent, which sends images to the LLM.
    - ``searcher``: the paper-scoring/relevance agent (discovery phase).

    ``saved_vendors`` stores the last-edited LLMConfig for each
    ``(role, vendor_key)`` pair so the user can switch providers in the TUI
    without losing previously entered keys/models. The live role field
    (``text`` / ``vision`` / ``searcher``) is the *currently active* provider;
    the pipeline only ever reads from the live field, so it requires no
    changes. ``saved_vendors`` is keyed as ``{role: {vendor_key: <LLMConfig>}}``.
    """

    text: LLMConfig = Field(default_factory=LLMConfig)
    vision: LLMConfig = Field(default_factory=_vision_default)
    searcher: LLMConfig = Field(default_factory=LLMConfig)
    saved_vendors: dict[str, dict[str, LLMConfig]] = Field(default_factory=dict)

    def for_role(self, role: str) -> LLMConfig:
        """Return the LLMConfig for a given role name."""
        if role not in ("text", "vision", "searcher"):
            raise ValueError(f"Unknown LLM role: {role!r}")
        return getattr(self, role)


class SearchConfig(BaseModel):
    """Paper search/discovery configuration."""

    default_date: str = Field(default="today", description="The default date to fetch papers for (e.g., 'today', 'yesterday').")
    default_limit: int = Field(default=50, description="The default number of papers to fetch and evaluate.")
    sort: str = Field(default="trending", description="How to sort the fetched papers (e.g., 'trending', 'recent').")
    profile_path: str = Field(default="config/profile.md", description="Path to the Markdown file containing the user's interests profile.")
    relevance_threshold: float = Field(default=0.6, description="Minimum score (0.0 to 1.0) required for a paper to be considered relevant.")
    max_reports_per_run: int = Field(default=5, description="Maximum number of reports to generate in a single run.")


class ReportConfig(BaseModel):
    """Report generation configuration."""

    output_dir: str = Field(default="output", description="Directory where generated reports will be saved.")
    template_dir: str = Field(default="templates", description="Directory containing Jinja2 templates for reports.")
    formats: list[str] = Field(default_factory=lambda: ["md", "html"], description="List of formats to output (e.g., 'md', 'html').")
    download_pdf: bool = Field(default=True, description="Whether to download the paper's PDF for figure extraction.")
    pdf_cache_dir: str = Field(default=".cache/pdfs", description="Directory to cache downloaded PDFs.")
    custom_agents: list[str] = Field(default_factory=list, description="List of custom agent modules to load.")
    language: str = Field(default="English", description="The language to generate the report in.")
    writer_research: bool = Field(
        default=True,
        description=(
            "Enable multi-turn research before the writer produces its analysis. "
            "The writer will search for unfamiliar concepts, cited works, and "
            "benchmarks to produce a more thorough and accurate report."
        ),
    )
    stream: bool = Field(
        default=False,
        description=(
            "Stream the Writer/Finder research-phase prose to the terminal as it "
            "is generated. When enabled, Phase 4 runs the two agents sequentially "
            "and prints their text deltas live instead of showing a spinner."
        ),
    )


class SchedulerConfig(BaseModel):
    """Auto-fetch scheduler configuration."""

    enabled: bool = Field(default=False, description="Whether the auto-fetch scheduler is enabled.")
    cron_hour: int = Field(default=8, description="The hour of the day (0-23) to run the scheduler.")
    cron_minute: int = Field(default=0, description="The minute of the hour (0-59) to run the scheduler.")
    timezone: str = Field(default="Asia/Shanghai", description="The timezone to use for the scheduler.")


class WechatPublishConfig(BaseModel):
    enabled: bool = False
    appid: str = ""
    secret: str = ""


class BlogPublishConfig(BaseModel):
    enabled: bool = False
    webhook_url: str = ""
    api_key: str = ""


class NotionPublishConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    database_id: str = ""


class PublishConfig(BaseModel):
    """Publishing configuration."""

    enabled: bool = False
    wechat: WechatPublishConfig = Field(default_factory=WechatPublishConfig)
    blog: BlogPublishConfig = Field(default_factory=BlogPublishConfig)
    notion: NotionPublishConfig = Field(default_factory=NotionPublishConfig)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    # Per-role LLM configs (text / vision / searcher). Backward-compatible with
    # legacy flat `[llm]` sections via migration in load_config().
    llms: LLMsConfig = Field(default_factory=LLMsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)

    # Resolved paths (set after loading)
    root: Path = PROJECT_ROOT

    @property
    def llm(self) -> LLMConfig:
        """Backward-compat accessor: the primary ("text") LLM config.

        Prefer ``config.llms.text`` / ``config.llms.for_role(...)`` in new code.
        """
        return self.llms.text

    @property
    def profile_path(self) -> Path:
        p = Path(self.search.profile_path)
        return p if p.is_absolute() else self.root / p

    @property
    def output_dir(self) -> Path:
        p = Path(self.report.output_dir)
        return p if p.is_absolute() else self.root / p

    @property
    def template_dir(self) -> Path:
        p = Path(self.report.template_dir)
        return p if p.is_absolute() else self.root / p

    @property
    def pdf_cache_dir(self) -> Path:
        p = Path(self.report.pdf_cache_dir)
        return p if p.is_absolute() else self.root / p


_LLM_ROLES = ("text", "vision", "searcher")


def _migrate_legacy_llm(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate a legacy flat ``[llm]`` section to the new ``[llms.*]`` structure.

    If ``raw`` has a flat ``llm`` mapping but no ``llms`` key, the flat config is
    cloned into ``llms.text``, ``llms.vision``, and ``llms.searcher`` so existing
    setups keep working unchanged. The legacy ``llm`` key is left in place only
    until the config is re-saved (callers should write ``llms`` on save).
    """
    flat = raw.get("llm")
    if not isinstance(flat, dict) or "llms" in raw:
        return raw
    # Clone the flat config into all three roles.
    import copy

    raw["llms"] = {role: copy.deepcopy(flat) for role in _LLM_ROLES}
    logger.info(
        "Migrated legacy [llm] config to [llms.text/vision/searcher]. "
        "Re-run `ppagent config` and save to persist the new structure."
    )
    return raw


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Override config values with environment variables.

    ``PPA_LLM_*`` apply to ALL three LLM roles (text/vision/searcher) so a single
    env var can configure the whole app headlessly.
    """
    llm_env = {
        "PPA_LLM_API_KEY": "api_key",
        "PPA_LLM_BASE_URL": "base_url",
        "PPA_LLM_MODEL": "model",
    }
    publish_env = {
        "PPA_NOTION_API_KEY": ("publish", "notion", "api_key"),
        "PPA_WECHAT_APPID": ("publish", "wechat", "appid"),
        "PPA_WECHAT_SECRET": ("publish", "wechat", "secret"),
        "PPA_BLOG_API_KEY": ("publish", "blog", "api_key"),
    }

    # LLM overrides → applied to each role under llms.<role>.<field>
    llms = raw.setdefault("llms", {})
    for env_var, field in llm_env.items():
        val = os.environ.get(env_var)
        if val is None:
            continue
        for role in _LLM_ROLES:
            llms.setdefault(role, {})[field] = val

    # Publishing overrides
    for env_var, keys in publish_env.items():
        val = os.environ.get(env_var)
        if val is None:
            continue
        d = raw
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    return raw


def _sync_backup(source: Path) -> None:
    """Mirror ``source`` into the persistent backup at ``_BACKUP_CONFIG_PATH``.

    The backup survives a project-directory reinstall (the project tree is wiped
    by a fresh ``git clone``), so this is what lets the user's keys and provider
    settings persist. ``save_config()`` already kept the backup in sync on TUI
    save, but a user who edits ``config/settings.toml`` directly — or who runs
    any CLI command without opening the TUI — would otherwise never seed the
    backup, and lose everything on the next reinstall.

    This is best-effort and non-fatal: a backup write failure must never block
    loading config. ``source`` must not already be the backup file itself
    (avoids a redundant in-place copy and the TUI's backup-as-source path).
    """
    try:
        backup = _BACKUP_CONFIG_PATH
        if source.resolve() == backup.resolve():
            return
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
    except OSError:
        logger.debug("Failed to mirror config backup", exc_info=True)


def _files_equal(a: Path, b: Path) -> bool:
    """Byte-compare two files. ``False`` if either is missing/unreadable."""
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _reconcile_config(explicit_path: Path | None) -> Path | None:
    """Resolve which config file to load, making the backup authoritative.

    The persistent backup is the source of truth (it survives reinstalls); the
    project-level ``config/settings.toml`` is a transient, gitignored working
    copy that can be recreated with stale defaults by a reinstall or a stray
    template. The previous behavior mirrored project → backup on *every* read,
    which let a stale project file silently destroy the user's real keys.

    Reconciliation rules (backup always wins on conflict):

    * ``explicit_path`` given → honor it verbatim (caller knows best; e.g. the
      TUI loading from the backup, or tests pointing at a fixture).
    * No backup exists → seed the backup from the project file (preserves the
      first-run flow) and trust the project file.
    * Backup exists, project file missing or differs → restore the backup over
      the project file and load the backup.
    * Both exist and are identical → no-op, load either.

    All writes are best-effort and non-fatal. Returns the path to load from,
    or ``None`` when no config exists anywhere (caller falls back to defaults).
    """
    if explicit_path is not None:
        # An explicit path is caller-authored (e.g. the TUI loading from the
        # backup, or tests pointing at a fixture); honor it verbatim and keep
        # the long-standing contract that a missing explicit path raises.
        if not explicit_path.exists():
            raise FileNotFoundError(f"Config file not found: {explicit_path}")
        return explicit_path

    project = _DEFAULT_CONFIG_PATHS[0]
    backup = _BACKUP_CONFIG_PATH
    backup_exists = backup.exists()
    project_exists = project.exists()

    if not backup_exists and not project_exists:
        return None

    try:
        if not backup_exists:
            # First run: seed the backup so future reinstalls can restore it.
            _sync_backup(project)
            return project
        if not project_exists or not _files_equal(project, backup):
            # Backup is authoritative — a stale/missing project file must never
            # overwrite it. Restore the working copy from the backup.
            project.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, project)
        return backup if project == _DEFAULT_CONFIG_PATHS[0] else project
    except OSError:
        logger.debug("Config reconciliation failed; using fallback", exc_info=True)
        # Fall back to whichever file is readable.
        return project if project_exists else (backup if backup_exists else None)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a TOML file.

    Searches in order: explicit path → ./config/settings.toml → ~/.config/ppagent/settings.toml.
    A legacy flat ``[llm]`` section is auto-migrated to ``[llms.*]``.
    Environment variables override TOML values.

    The persistent backup is authoritative: if a stale project-level
    ``config/settings.toml`` (recreated with defaults by a reinstall) disagrees
    with the backup, the backup wins and is restored over the project file.
    """
    config_path = _reconcile_config(path)

    raw: dict[str, Any] = {}
    if config_path is not None:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)

    raw = _migrate_legacy_llm(raw)
    raw = _apply_env_overrides(raw)
    # Drop the legacy `llm` key so model_validate doesn't choke (no such field now).
    raw.pop("llm", None)
    return AppConfig.model_validate(raw)
