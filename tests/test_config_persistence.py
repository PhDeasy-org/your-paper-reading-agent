"""Tests for config persistence across project-directory reinstalls.

The persistent backup at ``~/.config/ppagent/settings.toml`` is what lets a
user's provider keys/settings survive a reinstall (the project tree is wiped by
a fresh ``git clone``). These tests pin the three ways the backup must stay in
sync:

* ``load_config()`` mirrors the project file into the backup on every read, so
  edits made directly to ``config/settings.toml`` (outside the TUI) survive.
* Loading from the backup itself does not try to copy it over itself.
* End-to-end: edit → reinstall → ``config init`` restores the real keys.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import ppagent.config as C
from ppagent.config import load_config


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every config path constant into a tmp sandbox.

    Keeps the real ``~/.config/ppagent/settings.toml`` untouched.
    """
    project = tmp_path / "project"
    backup = tmp_path / "backup" / "settings.toml"

    monkeypatch.setattr(C, "PROJECT_ROOT", project)
    project_config = project / "config" / "settings.toml"
    monkeypatch.setattr(C, "_DEFAULT_CONFIG_PATHS", [project_config, backup])
    monkeypatch.setattr(C, "_BACKUP_CONFIG_PATH", backup)
    return project_config, backup


def _write(path: Path, model: str, api_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[llms.text]\n"
        f'model = "{model}"\n'
        f'api_key = "{api_key}"\n'
    )


def test_load_from_project_mirrors_backup(isolated_paths):
    """load_config() mirrors the project file into the persistent backup."""
    project_config, backup = isolated_paths
    _write(project_config, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")

    assert not backup.exists()
    load_config()

    # Backup now mirrors the project file the user edited directly.
    assert backup.exists()
    raw = C.tomllib.load(backup.open("rb"))
    assert raw["llms"]["text"]["api_key"] == "sk-REAL-DEEPSEEK-KEY"


def test_load_from_backup_does_not_self_copy(isolated_paths):
    """When the backup itself is the source, no copy-onto-self happens."""
    _project_config, backup = isolated_paths
    _write(backup, "glm-4", "sk-GLM-KEY")
    original = backup.read_bytes()

    load_config()  # falls through to the backup (project file absent)

    # File is byte-identical — no in-place corruption.
    assert backup.read_bytes() == original


def test_explicit_path_is_not_mirrored(isolated_paths):
    """An explicit path passed to load_config() is treated as ad-hoc."""
    project_config, backup = isolated_paths
    # No project config on disk; an explicit file must not seed the backup.
    other = project_config.parent / "elsewhere.toml"
    _write(other, "gpt-4o", "sk-EXPLICIT")

    load_config(other)

    assert not backup.exists()


def test_reinstall_survives(isolated_paths):
    """End-to-end: direct edit → reinstall → config init restores real keys."""
    project_config, backup = isolated_paths

    # 1) User edits config/settings.toml directly with their real key.
    _write(project_config, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")
    # 2) Any CLI command triggers load_config(), which mirrors the backup.
    load_config()
    assert backup.exists()

    # 3) Reinstall wipes the project tree (fresh git clone).
    shutil.rmtree(project_config.parent)
    assert not project_config.exists()

    # 4) config_init-style restore copies the backup back into the project.
    project_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, project_config)
    cfg = load_config()
    assert cfg.llms.text.api_key == "sk-REAL-DEEPSEEK-KEY"
