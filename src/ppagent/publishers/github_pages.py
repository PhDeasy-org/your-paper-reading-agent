"""GitHub Pages blog publisher.

Copies each generated report directory into a local working copy of a
GitHub Pages-enabled repository, then commits and pushes it. GitHub Pages
serves the static files. The user owns the repo (provides username + repo
name) and must have a local clone with push access configured (``repo_path``).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from ppagent.models import PaperReport
from ppagent.publishers import register_publisher
from ppagent.publishers.base import PublisherBase
from ppagent.storage import Storage

logger = logging.getLogger(__name__)


@register_publisher
class GithubPagesPublisher(PublisherBase):
    """Publishes reports to a GitHub Pages blog via a local git clone."""

    name = "github_pages"

    def __init__(
        self,
        username: str = "",
        repo: str = "",
        repo_path: str = "",
        branch: str = "main",
        posts_subdir: str = "papers",
    ) -> None:
        self.username = username
        self.repo = repo
        self.repo_path = Path(repo_path).expanduser() if repo_path else Path()
        self.branch = branch
        self.posts_subdir = posts_subdir or "papers"

    def validate_config(self) -> bool:
        missing = [
            label
            for label, val in (
                ("username", self.username),
                ("repo", self.repo),
                ("repo_path", str(self.repo_path)),
            )
            if not val
        ]
        if missing:
            logger.error("GitHub Pages publisher requires: %s", ", ".join(missing))
            return False
        if not (self.repo_path / ".git").is_dir():
            logger.warning(
                "GitHub Pages repo_path %s is not a git repository "
                "(commit/push will fail until you `git clone` it there)",
                self.repo_path,
            )
        return True

    def publish(
        self,
        report: PaperReport,
        *,
        md_content: str,
        html_content: str,
        report_dir: Path | None = None,
    ) -> bool:
        if not self.validate_config():
            return False
        if report_dir is None:
            logger.error(
                "GitHub Pages publisher requires the on-disk report_dir; got None."
            )
            return False

        # Stable, collision-free destination name (matches the local output dir).
        safe_name = Storage._safe_filename(
            report.paper.title, report.paper.published_at
        )
        dest_root = self.repo_path / self.posts_subdir
        dest = dest_root / safe_name

        try:
            dest_root.mkdir(parents=True, exist_ok=True)
            # Copy report files (html/md/metadata.json + figures/) into the repo.
            shutil.copytree(report_dir, dest, dirs_exist_ok=True)
            self._git_commit_and_push(report.paper.title, safe_name)
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.error("GitHub Pages publish failed: %s", exc)
            return False

        rel = dest.relative_to(self.repo_path)
        logger.info("GitHub Pages publish successful: %s → %s", report.paper.title, rel)
        return True

    def _git_commit_and_push(self, title: str, safe_name: str) -> None:
        """Stage the new post, commit, and push to the configured branch.

        Raises ``subprocess.CalledProcessError`` on any git failure so the
        caller can log and return False without aborting the pipeline.
        """
        add_path = str(Path(self.posts_subdir) / safe_name)
        env = {
            "GIT_TERMINAL_PROMPT": "0",  # never hang on credential prompts
        }
        subprocess.run(
            ["git", "-C", str(self.repo_path), "add", add_path],
            check=True,
            capture_output=True,
            env=env,
        )
        commit = subprocess.run(
            [
                "git",
                "-C",
                str(self.repo_path),
                "commit",
                "-m",
                f"Publish: {title}",
            ],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        if commit.returncode != 0:
            # ``git commit`` exits non-zero when there is nothing to commit
            # (e.g. re-publishing an identical report). Treat that as success;
            # surface any other failure.
            if "nothing to commit" not in (commit.stdout + commit.stderr).lower():
                raise subprocess.CalledProcessError(
                    commit.returncode, commit.args, commit.stdout, commit.stderr
                )
            return
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo_path),
                "push",
                "origin",
                self.branch,
            ],
            check=True,
            capture_output=True,
            env=env,
        )
