"""Tests for config persistence under the single-path model.

Config lives at one location — ``~/.config/ppagent/settings.toml`` — outside
the project tree so it survives reinstalls. There is no backup, no reconcile,
no project-file mirror. These tests pin the rules of that model:

* ``load_config()`` reads from ``CONFIG_PATH`` directly.
* On the first load after upgrade, a legacy ``config/settings.toml`` is
  migrated into ``CONFIG_PATH`` once — and never re-read afterward.
* When nothing exists, built-in defaults are returned.
* An explicit ``path`` is honored verbatim; a missing one raises.
* ``save_config()`` writes only its target path (no ``.bak``, no mirror).
* The default ``profile_path`` points into ``CONFIG_DIR`` and expands ``~``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import ppagent.config as C
import ppagent.tui as T
from ppagent.config import AppConfig, load_config
from ppagent.tui import save_config


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every config path constant into a tmp sandbox.

    Keeps the real ``~/.config/ppagent/settings.toml`` untouched.
    """
    project = tmp_path / "project"
    config_dir = tmp_path / "cfgdir"
    config_path = config_dir / "settings.toml"
    legacy = project / "config" / "settings.toml"

    monkeypatch.setattr(C, "PROJECT_ROOT", project)
    monkeypatch.setattr(C, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(C, "CONFIG_PATH", config_path)
    monkeypatch.setattr(C, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.setattr(T, "CONFIG_PATH", config_path)
    return project, config_dir, config_path, legacy


def _write(path: Path, model: str, api_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[llms.text]\nmodel = "{model}"\napi_key = "{api_key}"\n')


def _key(path: Path) -> str:
    return C.tomllib.load(path.open("rb"))["llms"]["text"]["api_key"]


# ---------------------------------------------------------------------------
# Basic single-path reading.
# ---------------------------------------------------------------------------


def test_load_reads_config_path(isolated_paths):
    """load_config() reads from CONFIG_PATH directly."""
    _project, _config_dir, config_path, _legacy = isolated_paths
    _write(config_path, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")

    cfg = load_config()

    assert cfg.llms.text.api_key == "sk-REAL-DEEPSEEK-KEY"
    assert cfg.llms.text.model == "deepseek-chat"


def test_load_returns_defaults_when_nothing_exists(isolated_paths):
    """When no config exists anywhere, built-in defaults are returned."""
    _project, _config_dir, _config_path, _legacy = isolated_paths

    cfg = load_config()

    # Default model for the text role is gpt-4o (LLMConfig default).
    assert cfg.llms.text.model == "gpt-4o"
    assert cfg.llms.text.api_key == ""


# ---------------------------------------------------------------------------
# One-time migration from the legacy project file.
# ---------------------------------------------------------------------------


def test_load_migrates_legacy_project_file_once(isolated_paths):
    """First load after upgrade: CONFIG_PATH absent but legacy file exists →
    legacy is copied into CONFIG_PATH once, then loaded."""
    _project, _config_dir, config_path, legacy = isolated_paths
    _write(legacy, "qwen3.7-plus", "sk-MY-REAL-QWEN-KEY")
    assert not config_path.exists()

    cfg = load_config()

    assert cfg.llms.text.api_key == "sk-MY-REAL-QWEN-KEY"
    assert config_path.exists()
    assert _key(config_path) == "sk-MY-REAL-QWEN-KEY"


def test_second_load_does_not_re_read_legacy(isolated_paths):
    """After migration, edits to the legacy file are ignored — CONFIG_PATH is
    the sole source of truth."""
    _project, _config_dir, config_path, legacy = isolated_paths
    _write(legacy, "qwen3.7-plus", "sk-FIRST")
    load_config()  # migrates

    # Tamper with the legacy file post-migration.
    _write(legacy, "gpt-4o", "sk-SHOULD-BE-IGNORED")

    cfg = load_config()
    assert cfg.llms.text.api_key == "sk-FIRST"


def test_existing_config_path_wins_over_legacy(isolated_paths):
    """When both exist, CONFIG_PATH wins (legacy is never consulted)."""
    _project, _config_dir, config_path, legacy = isolated_paths
    _write(config_path, "deepseek-chat", "sk-CONFIG-PATH-KEY")
    _write(legacy, "gpt-4o", "sk-LEGACY-KEY")

    cfg = load_config()

    assert cfg.llms.text.api_key == "sk-CONFIG-PATH-KEY"


# ---------------------------------------------------------------------------
# Explicit-path contract.
# ---------------------------------------------------------------------------


def test_explicit_path_honored_and_missing_raises(isolated_paths):
    """An explicit path is honored verbatim (no migration seeding), and a
    missing explicit path raises FileNotFoundError."""
    _project, _config_dir, config_path, _legacy = isolated_paths
    other = _project / "elsewhere.toml"
    _write(other, "gpt-4o", "sk-EXPLICIT")

    cfg = load_config(other)
    assert cfg.llms.text.api_key == "sk-EXPLICIT"
    # Explicit path must not seed CONFIG_PATH.
    assert not config_path.exists()

    with pytest.raises(FileNotFoundError):
        load_config(_project / "missing.toml")


# ---------------------------------------------------------------------------
# save_config: writes only its target path.
# ---------------------------------------------------------------------------


def test_save_writes_only_target_path(isolated_paths):
    """save_config() writes only the path it's given — no .bak, no mirror
    into the project directory."""
    _project, _config_dir, config_path, legacy = isolated_paths
    target = _config_dir / "saved.toml"
    cfg = AppConfig()
    cfg.llms.text.api_key = "sk-NEW-KEY"
    cfg.llms.text.model = "deepseek-chat"

    save_config(cfg, target)

    assert _key(target) == "sk-NEW-KEY"
    # No backup, no .bak, nothing written to the project tree.
    assert not (target.with_suffix(".toml.bak")).exists()
    assert not config_path.exists()
    assert not legacy.exists()
    assert not (_project / "config").exists()


def test_save_then_load_roundtrip(isolated_paths):
    """Saving to CONFIG_PATH and loading reads the same values back."""
    _project, _config_dir, config_path, _legacy = isolated_paths
    cfg = AppConfig()
    cfg.llms.text.api_key = "sk-ROUNDTRIP"
    cfg.llms.text.model = "glm-4"

    save_config(cfg, config_path)
    loaded = load_config()

    assert loaded.llms.text.api_key == "sk-ROUNDTRIP"
    assert loaded.llms.text.model == "glm-4"


# ---------------------------------------------------------------------------
# profile_path default points into CONFIG_DIR and expands ~.
# ---------------------------------------------------------------------------


def test_profile_path_default_is_config_dir():
    """The default profile_path points at ~/.config/ppagent/profile.md."""
    cfg = AppConfig()
    assert cfg.search.profile_path == "~/.config/ppagent/profile.md"
    resolved = cfg.profile_path
    assert resolved == Path.home() / ".config" / "ppagent" / "profile.md"


# ---------------------------------------------------------------------------
# End-to-end reinstall scenario.
# ---------------------------------------------------------------------------


def test_reinstall_survives(isolated_paths):
    """Edit → save → reinstall (project wiped) → reload: keys survive because
    they were never in the project tree."""
    project, _config_dir, config_path, _legacy = isolated_paths
    # The project tree exists (e.g. a git clone) before the reinstall wipes it.
    project.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig()
    cfg.llms.text.api_key = "sk-REAL-DEEPSEEK-KEY"
    cfg.llms.text.model = "deepseek-chat"
    save_config(cfg, config_path)

    # Reinstall wipes the project tree entirely.
    shutil.rmtree(project)
    assert not project.exists()

    # Config is untouched — it never lived in the project tree.
    cfg2 = load_config()
    assert cfg2.llms.text.api_key == "sk-REAL-DEEPSEEK-KEY"
