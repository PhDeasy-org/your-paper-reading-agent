"""Configuration models and loading for ppagent."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Project root — where pyproject.toml lives
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_CONFIG_PATHS = [
    PROJECT_ROOT / "config" / "settings.toml",
    Path.home() / ".config" / "ppagent" / "settings.toml",
]


class LLMConfig(BaseModel):
    """LLM API configuration."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120  # seconds
    instructor_mode: str = "auto"


class SearchConfig(BaseModel):
    """Paper search/discovery configuration."""

    default_date: str = "today"
    default_limit: int = 50
    sort: str = "trending"
    profile_path: str = "config/profile.md"
    relevance_threshold: float = 0.6
    max_reports_per_run: int = 5


class ReportConfig(BaseModel):
    """Report generation configuration."""

    output_dir: str = "output"
    template_dir: str = "templates"
    formats: list[str] = Field(default_factory=lambda: ["md", "html"])
    download_pdf: bool = True
    pdf_cache_dir: str = ".cache/pdfs"
    custom_agents: list[str] = Field(default_factory=list)
    language: str = "English"


class SchedulerConfig(BaseModel):
    """Auto-fetch scheduler configuration."""

    enabled: bool = False
    cron_hour: int = 8
    cron_minute: int = 0
    timezone: str = "Asia/Shanghai"


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

    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)

    # Resolved paths (set after loading)
    root: Path = PROJECT_ROOT

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


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Override config values with environment variables."""
    env_map = {
        "PPA_LLM_API_KEY": ("llm", "api_key"),
        "PPA_LLM_BASE_URL": ("llm", "base_url"),
        "PPA_LLM_MODEL": ("llm", "model"),
        "PPA_NOTION_API_KEY": ("publish", "notion", "api_key"),
        "PPA_WECHAT_APPID": ("publish", "wechat", "appid"),
        "PPA_WECHAT_SECRET": ("publish", "wechat", "secret"),
        "PPA_BLOG_API_KEY": ("publish", "blog", "api_key"),
    }
    for env_var, keys in env_map.items():
        val = os.environ.get(env_var)
        if val is None:
            continue
        d = raw
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return raw


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a TOML file.

    Searches in order: explicit path → ./config/settings.toml → ~/.config/ppagent/settings.toml.
    Environment variables override TOML values.
    """
    config_path: Path | None = None
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        config_path = path
    else:
        for candidate in _DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                config_path = candidate
                break

    raw: dict[str, Any] = {}
    if config_path is not None:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)

    raw = _apply_env_overrides(raw)
    return AppConfig.model_validate(raw)
