"""Tests for config persistence across project-directory reinstalls.

The persistent backup at ``~/.config/ppagent/settings.toml`` is the source of
truth: it survives a reinstall (the project tree is wiped by a fresh ``git
clone``), while the transient ``config/settings.toml`` can be recreated with
stale OpenAI defaults by a reinstall or a stray template.

These tests pin the three rules that keep the user's keys safe:

* On conflict, the **backup always wins** — a stale project file can never
  overwrite real keys in the backup (the regression that previously caused data
  loss on reinstall).
* The backup is seeded from the project file only on the first run (when no
  backup exists yet), so a direct edit to ``config/settings.toml`` is captured.
* End-to-end: edit → reinstall → ``config init`` restores the real keys, even
  when a stale ``settings.toml`` has been dropped on top.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import ppagent.config as C
import ppagent.tui as T
from ppagent.config import load_config
from ppagent.tui import save_config
from ppagent.config import AppConfig


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
    # The one-generation .bak sits next to the backup.
    monkeypatch.setattr(C, "_BACKUP_PATH_BAK", backup.with_suffix(".toml.bak"))
    monkeypatch.setattr(T, "_BACKUP_CONFIG_PATH", backup)
    monkeypatch.setattr(T, "_BACKUP_PATH_BAK", backup.with_suffix(".toml.bak"))
    return project_config, backup


def _write(path: Path, model: str, api_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[llms.text]\n"
        f'model = "{model}"\n'
        f'api_key = "{api_key}"\n'
    )


def _key(path: Path) -> str:
    return C.tomllib.load(path.open("rb"))["llms"]["text"]["api_key"]


# ---------------------------------------------------------------------------
# Core regression: a stale project file must NOT destroy the backup.
# ---------------------------------------------------------------------------


def test_stale_project_file_does_not_overwrite_backup(isolated_paths):
    """The data-loss regression. Backup has real keys; a reinstall dropped a
    stale project file with defaults. load_config() must return the real key
    AND leave the backup intact."""
    project_config, backup = isolated_paths
    _write(backup, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")
    _write(project_config, "gpt-4o", "sk-your-key-here")  # stale defaults

    cfg = load_config()

    # Loaded value is the real key (backup won).
    assert cfg.llms.text.api_key == "sk-REAL-DEEPSEEK-KEY"
    # The backup was NOT clobbered with stale defaults.
    assert _key(backup) == "sk-REAL-DEEPSEEK-KEY"


def test_backup_restored_over_stale_project_file(isolated_paths):
    """When the backup wins, the stale project file is itself overwritten with
    the backup content so subsequent loads are consistent."""
    project_config, backup = isolated_paths
    _write(backup, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")
    _write(project_config, "gpt-4o", "sk-your-key-here")

    load_config()

    assert project_config.exists()
    assert _key(project_config) == "sk-REAL-DEEPSEEK-KEY"


# ---------------------------------------------------------------------------
# Seeding & no-op behavior.
# ---------------------------------------------------------------------------


def test_load_seeds_backup_when_none_exists(isolated_paths):
    """First run (no backup) seeds the backup from the project file, so a direct
    edit survives a future reinstall."""
    project_config, backup = isolated_paths
    _write(project_config, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")

    assert not backup.exists()
    load_config()

    assert backup.exists()
    assert _key(backup) == "sk-REAL-DEEPSEEK-KEY"


def test_load_identical_files_is_noop(isolated_paths):
    """When project and backup are byte-identical, load_config() writes nothing."""
    project_config, backup = isolated_paths
    _write(project_config, "glm-4", "sk-GLM-KEY")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project_config, backup)

    project_mtime_before = project_config.stat().st_mtime_ns
    backup_mtime_before = backup.stat().st_mtime_ns

    cfg = load_config()

    assert cfg.llms.text.api_key == "sk-GLM-KEY"
    # No rewrites → mtimes unchanged.
    assert project_config.stat().st_mtime_ns == project_mtime_before
    assert backup.stat().st_mtime_ns == backup_mtime_before


def test_load_from_backup_does_not_self_copy(isolated_paths):
    """When only the backup exists, loading does not corrupt it."""
    _project_config, backup = isolated_paths
    _write(backup, "glm-4", "sk-GLM-KEY")
    original = backup.read_bytes()

    load_config()  # project file absent → backup is the source

    assert backup.read_bytes() == original


def test_explicit_path_keeps_old_contract(isolated_paths):
    """An explicit path passed to load_config() is honored verbatim (no backup
    seeding), and a missing explicit path raises FileNotFoundError."""
    project_config, backup = isolated_paths
    other = project_config.parent / "elsewhere.toml"
    _write(other, "gpt-4o", "sk-EXPLICIT")

    load_config(other)  # ad-hoc; must not seed the backup
    assert not backup.exists()

    with pytest.raises(FileNotFoundError):
        load_config(project_config.parent / "missing.toml")


# ---------------------------------------------------------------------------
# End-to-end reinstall scenarios.
# ---------------------------------------------------------------------------


def test_reinstall_survives(isolated_paths):
    """Direct edit → load (seeds backup) → reinstall (project wiped) → restore."""
    project_config, backup = isolated_paths

    # 1) User edits config/settings.toml directly with their real key.
    _write(project_config, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")
    # 2) Any CLI command triggers load_config(), which seeds the backup.
    load_config()
    assert _key(backup) == "sk-REAL-DEEPSEEK-KEY"

    # 3) Reinstall wipes the project tree (fresh git clone).
    shutil.rmtree(project_config.parent)
    assert not project_config.exists()

    # 4) config_init-style restore copies the backup back into the project.
    project_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, project_config)
    cfg = load_config()
    assert cfg.llms.text.api_key == "sk-REAL-DEEPSEEK-KEY"


def test_reinstall_with_stale_dropped_file_still_restores(isolated_paths):
    """The full gap-#1 + gap-#2 scenario: reinstall drops a stale settings.toml
    AND config_init is skipped (because a file exists). load_config() must still
    surface the real keys from the backup."""
    project_config, backup = isolated_paths
    _write(project_config, "deepseek-chat", "sk-REAL-DEEPSEEK-KEY")
    load_config()  # seeds backup

    # Reinstall: wipes real file, then a stale template/default reappears.
    _write(project_config, "gpt-4o", "sk-your-key-here")

    # config_init early-returns on the existing file (gap #2), so the rescue
    # must come from load_config()'s reconcile.
    cfg = load_config()
    assert cfg.llms.text.api_key == "sk-REAL-DEEPSEEK-KEY"
    assert _key(backup) == "sk-REAL-DEEPSEEK-KEY"


# ---------------------------------------------------------------------------
# .bak safety net on save.
# ---------------------------------------------------------------------------


def test_save_rotates_one_generation_bak(isolated_paths):
    """save_config() copies the previous backup to settings.toml.bak before
    overwriting, giving a one-generation undo."""
    project_config, backup = isolated_paths
    bak = backup.with_suffix(".toml.bak")
    # Previous backup holds an older key.
    _write(backup, "glm-4", "sk-PREVIOUS-BACKUP-KEY")

    # Build a cfg whose save will carry the new key, then save it.
    cfg = AppConfig()
    cfg.llms.text.api_key = "sk-NEW-KEY"
    cfg.llms.text.model = "deepseek-chat"
    save_config(cfg, project_config)

    # The previous backup content survives in the .bak.
    assert bak.exists()
    assert _key(bak) == "sk-PREVIOUS-BACKUP-KEY"
    # The new backup holds the just-saved value.
    assert _key(backup) == "sk-NEW-KEY"


def test_save_writes_no_bak_when_backup_absent(isolated_paths):
    """On a first-ever save (no prior backup), no .bak is created."""
    project_config, backup = isolated_paths
    bak = backup.with_suffix(".toml.bak")

    assert not backup.exists()
    save_config(AppConfig(), project_config)

    assert not bak.exists()
    assert backup.exists()
