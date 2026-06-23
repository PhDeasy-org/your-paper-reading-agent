"""Tests for the publisher framework: registration, config wiring, and the
GitHub Pages file-copy + git-commit behavior.

The GitHub Pages test builds a real throwaway git repo under ``tmp_path`` so
we assert the actual end state (files landed in the repo + a commit was made)
rather than mocking subprocess.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from ppagent.config import AppConfig, GithubPagesPublishConfig
from ppagent.models import Paper, PaperReport, ReportSection
from ppagent.publishers import get_publisher, list_publishers
from ppagent.publishers.github_pages import GithubPagesPublisher
from ppagent.storage import Storage


# ---------------------------------------------------------------------------
# Registry & config wiring.
# ---------------------------------------------------------------------------


def test_github_pages_publisher_is_registered():
    """The new publisher registers under 'github_pages' and the old 'blog'
    name is gone."""
    assert "github_pages" in list_publishers()
    assert "blog" not in list_publishers()


def test_github_pages_default_config():
    """AppConfig ships the new config block with sensible defaults."""
    cfg = AppConfig()
    ghp = cfg.publish.github_pages
    assert isinstance(ghp, GithubPagesPublishConfig)
    assert ghp.enabled is False
    assert ghp.username == ""
    assert ghp.repo == ""
    assert ghp.repo_path == ""
    assert ghp.branch == "main"
    assert ghp.posts_subdir == "papers"


def test_github_pages_env_override(monkeypatch):
    """PPA_GH_PAGES_* env vars populate the github_pages config block."""
    monkeypatch.setenv("PPA_GH_PAGES_USERNAME", "octocat")
    monkeypatch.setenv("PPA_GH_PAGES_REPO", "my-paper-blog")
    monkeypatch.setenv("PPA_GH_PAGES_REPO_PATH", "/tmp/blog")
    monkeypatch.setenv("PPA_GH_PAGES_BRANCH", "gh-pages")
    monkeypatch.setenv("PPA_GH_PAGES_POSTS_SUBDIR", "posts")

    # Reload the override function in isolation against a fresh dict.
    from ppagent.config import _apply_env_overrides

    raw = _apply_env_overrides({})
    ghp = raw["publish"]["github_pages"]
    assert ghp["username"] == "octocat"
    assert ghp["repo"] == "my-paper-blog"
    assert ghp["repo_path"] == "/tmp/blog"
    assert ghp["branch"] == "gh-pages"
    assert ghp["posts_subdir"] == "posts"


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_validate_config_rejects_empty_fields():
    p = GithubPagesPublisher()
    assert p.validate_config() is False

    p = GithubPagesPublisher(username="octocat", repo="blog", repo_path="/tmp/blog")
    assert p.validate_config() is True


def test_publish_without_report_dir_returns_false(sample_report):
    p = GithubPagesPublisher(username="octocat", repo="blog", repo_path="/tmp/blog")
    assert p.publish(sample_report, md_content="md", html_content="html") is False


# ---------------------------------------------------------------------------
# Behavioral end-to-end test using a real throwaway git repo.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def git_env(monkeypatch):
    """Ensure git identity env vars are configured for all subprocesses in these tests."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command inside ``repo`` with a deterministic identity."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )


def _make_report_dir(tmp_path: Path) -> Path:
    """Stand up a fake report dir mirroring what the assembler writes."""
    report_dir = tmp_path / "src_report" / "25-06-My-Paper"
    report_dir.mkdir(parents=True)
    (report_dir / "report.html").write_text("<h1>hi</h1>")
    (report_dir / "report.md").write_text("# hi")
    (report_dir / "metadata.json").write_text("{}")
    (report_dir / "figures").mkdir()
    (report_dir / "figures" / "figure_1.png").write_bytes(b"\x89PNG fake")
    return report_dir


def test_publish_copies_files_and_commits(tmp_path, monkeypatch):
    """publish() copies the report dir into the repo, commits, and pushes
    (to a local 'origin' remote inside tmp_path so no network is needed)."""
    # Skip the test entirely if git isn't available.
    if subprocess.run(["which", "git"], capture_output=True).returncode != 0:
        pytest.skip("git not installed")

    # --- Set up the blog repo as a bare "remote" + a working clone. ---
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    repo = tmp_path / "blog"
    _git(tmp_path, "init", str(repo))
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "remote", "add", "origin", str(remote))
    (repo / ".gitkeep").write_text("")
    _git(repo, "add", ".gitkeep")
    _git(repo, "commit", "-m", "init")

    # --- Configure the publisher against the working clone. ---
    report_dir = _make_report_dir(tmp_path)
    paper = Paper(id="2506.12345", title="My Paper", published_at=datetime(2025, 6, 10))
    report = PaperReport(
        paper=paper,
        metadata=ReportSection(name="metadata", content=""),
        benchmarks=ReportSection(name="benchmarks", content=""),
        tldr=ReportSection(name="tldr", content=""),
        previous_works=ReportSection(name="previous_works", content=""),
        method=ReportSection(name="method", content=""),
        evaluation=ReportSection(name="evaluation", content=""),
        critique=ReportSection(name="critique", content=""),
    )

    publisher = GithubPagesPublisher(
        username="octocat",
        repo="blog",
        repo_path=str(repo),
        branch="main",
        posts_subdir="papers",
    )

    ok = publisher.publish(
        report, md_content="# hi", html_content="<h1>hi</h1>", report_dir=report_dir
    )
    assert ok is True

    # --- Assert files landed in the repo working copy. ---
    safe_name = Storage._safe_filename(paper.title, paper.published_at)
    dest = repo / "papers" / safe_name
    assert (dest / "report.html").read_text() == "<h1>hi</h1>"
    assert (dest / "figures" / "figure_1.png").read_bytes() == b"\x89PNG fake"

    # --- Assert a commit was made and pushed to the remote. ---
    log = _git(repo, "log", "--oneline").stdout
    assert "Publish: My Paper" in log
    # The push landed on the remote.
    remote_main = _git(repo, "ls-remote", "origin", "refs/heads/main").stdout
    assert remote_main.strip() != ""


def test_publish_is_idempotent_when_unchanged(tmp_path):
    """Re-publishing an identical report must not raise (no-op commit)."""
    if subprocess.run(["which", "git"], capture_output=True).returncode != 0:
        pytest.skip("git not installed")

    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    repo = tmp_path / "blog"
    _git(tmp_path, "init", str(repo))
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "remote", "add", "origin", str(remote))
    (repo / ".gitkeep").write_text("")
    _git(repo, "add", ".gitkeep")
    _git(repo, "commit", "-m", "init")

    report_dir = _make_report_dir(tmp_path)
    paper = Paper(id="2506.12345", title="My Paper", published_at=datetime(2025, 6, 10))
    report = PaperReport(
        paper=paper,
        metadata=ReportSection(name="metadata", content=""),
        benchmarks=ReportSection(name="benchmarks", content=""),
        tldr=ReportSection(name="tldr", content=""),
        previous_works=ReportSection(name="previous_works", content=""),
        method=ReportSection(name="method", content=""),
        evaluation=ReportSection(name="evaluation", content=""),
        critique=ReportSection(name="critique", content=""),
    )
    publisher = GithubPagesPublisher(
        username="octocat",
        repo="blog",
        repo_path=str(repo),
        branch="main",
        posts_subdir="papers",
    )

    assert (
        publisher.publish(
            report, md_content="# hi", html_content="<h1>hi</h1>", report_dir=report_dir
        )
        is True
    )
    # Second publish with identical content: commit is a no-op, must not raise.
    assert (
        publisher.publish(
            report, md_content="# hi", html_content="<h1>hi</h1>", report_dir=report_dir
        )
        is True
    )


def test_get_publisher_instantiates_from_config_dump():
    """get_publisher(name, **cfg.model_dump(exclude={'enabled'})) round-trips
    the config fields — this is how the pipeline constructs publishers
    (``enabled`` is a config-only flag, not a ctor param)."""
    cfg = GithubPagesPublishConfig(
        enabled=True,
        username="octocat",
        repo="blog",
        repo_path="/tmp/blog",
        branch="main",
        posts_subdir="papers",
    )
    p = get_publisher("github_pages", **cfg.model_dump(exclude={"enabled"}))
    assert isinstance(p, GithubPagesPublisher)
    assert p.username == "octocat"
    assert p.repo == "blog"
