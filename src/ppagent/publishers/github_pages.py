"""GitHub Pages blog publisher.

Copies each generated report directory into a local working copy of a
GitHub Pages-enabled repository, then commits and pushes it. GitHub Pages
serves the static files. The user owns the repo (provides username + repo
name) and must have a local clone with push access configured (``repo_path``).
"""

from __future__ import annotations

import logging
import os
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

            # Generate and stage index.html
            self._generate_and_stage_index_html()

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
            **os.environ,
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

    def _generate_and_stage_index_html(self) -> None:
        """Scan all published papers, generate an index.html, and stage it."""
        dest_root = self.repo_path / self.posts_subdir
        if not dest_root.is_dir():
            return

        papers = []
        import json
        import re

        for item in dest_root.iterdir():
            if item.is_dir():
                metadata_path = item / "metadata.json"
                if metadata_path.is_file():
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        paper_data = data.get("paper", {})
                        title = paper_data.get("title", "")
                        published_at = paper_data.get("published_at", "")

                        if published_at and len(published_at) >= 10:
                            published_date = published_at[:10]
                        else:
                            published_date = published_at

                        tldr = data.get("tldr", {}).get("content", "")
                        if not tldr:
                            tldr = paper_data.get("summary", "")

                        authors = paper_data.get("authors", [])
                        author_str = ", ".join(authors[:3])
                        if len(authors) > 3:
                            author_str += ", ..."

                        keywords_str = ""
                        meta_content = data.get("metadata", {}).get("content", "")
                        if meta_content:
                            match = re.search(
                                r"\*\*(?:Keywords|Key Words)\*\*\s*\|\s*([^|\n]+)",
                                meta_content,
                                re.IGNORECASE,
                            )
                            if match:
                                keywords_str = match.group(1).strip()
                            else:
                                keywords_str = "paper"
                        else:
                            keywords_str = "paper"

                        paper_type = data.get("paper_type", "research")
                        rel_link = f"./{self.posts_subdir}/{item.name}/report.html"
                        arxiv_url = paper_data.get("arxiv_url", "")

                        papers.append(
                            {
                                "title": title,
                                "published_date": published_date,
                                "tldr": tldr,
                                "authors": author_str,
                                "keywords": keywords_str,
                                "paper_type": paper_type,
                                "link": rel_link,
                                "arxiv_url": arxiv_url,
                            }
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse metadata for %s: %s", item.name, exc
                        )

        papers.sort(key=lambda x: x["published_date"], reverse=True)

        try:
            from jinja2 import Template

            template_content = self._get_index_template()
            template = Template(template_content)
            rendered_html = template.render(
                papers=papers,
                repo_name=self.repo,
                username=self.username,
            )

            index_path = self.repo_path / "index.html"
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(rendered_html)

            # Stage the index.html
            env = {
                **os.environ,
                "GIT_TERMINAL_PROMPT": "0",
            }
            subprocess.run(
                ["git", "-C", str(self.repo_path), "add", "index.html"],
                check=True,
                capture_output=True,
                env=env,
            )
            logger.info("Successfully generated and staged index.html")
        except Exception as exc:
            logger.error("Failed to generate or stage index.html: %s", exc)

    def _get_index_template(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Curated summaries and deep-dive technical reports of trending machine learning and AI research papers.">
    <title>AI Research Journal | Daily Paper Readings</title>
    
    <!-- Modern Typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 30, 49, 0.5);
            --card-border: rgba(99, 102, 241, 0.15);
            --card-border-hover: rgba(99, 102, 241, 0.4);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #6366f1;
            --accent-secondary: #06b6d4;
            --glow-primary: rgba(99, 102, 241, 0.15);
            --glow-secondary: rgba(6, 118, 212, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }

        /* Decorative background glows */
        .ambient-glow-1 {
            position: absolute;
            width: 500px;
            height: 500px;
            top: -200px;
            left: -100px;
            background: radial-gradient(circle, var(--glow-primary) 0%, transparent 70%);
            z-index: -1;
            filter: blur(40px);
            pointer-events: none;
        }

        .ambient-glow-2 {
            position: absolute;
            width: 600px;
            height: 600px;
            bottom: 10%;
            right: -200px;
            background: radial-gradient(circle, var(--glow-secondary) 0%, transparent 70%);
            z-index: -1;
            filter: blur(50px);
            pointer-events: none;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem 1.5rem 5rem 1.5rem;
        }

        /* Header / Navbar */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 4rem;
        }

        .logo {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 1.5rem;
            background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .logo svg {
            width: 28px;
            height: 28px;
            stroke: #818cf8;
        }

        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.95rem;
            transition: color 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .nav-links a:hover {
            color: var(--text-primary);
        }

        /* Hero Section */
        .hero {
            text-align: center;
            max-width: 800px;
            margin: 0 auto 5rem auto;
            position: relative;
        }

        .hero h1 {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 3.5rem;
            line-height: 1.15;
            margin-bottom: 1.5rem;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #ffffff 30%, #c7d2fe 70%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            font-size: 1.2rem;
            color: var(--text-secondary);
            font-weight: 300;
            line-height: 1.6;
            margin-bottom: 2.5rem;
        }

        /* Search bar */
        .search-container {
            position: relative;
            max-width: 500px;
            margin: 0 auto;
        }

        .search-input {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 9999px;
            padding: 1rem 1.5rem 1rem 3rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 1rem;
            outline: none;
            backdrop-filter: blur(10px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        }

        .search-input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 25px 0 rgba(99, 102, 241, 0.25), 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        }

        .search-icon {
            position: absolute;
            left: 1.2rem;
            top: 50%;
            transform: translateY(-50%);
            width: 20px;
            height: 20px;
            fill: none;
            stroke: var(--text-muted);
            pointer-events: none;
        }

        /* Papers Grid */
        .section-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.75rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .section-title span {
            background-color: rgba(99, 102, 241, 0.1);
            color: var(--accent-primary);
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .papers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 2rem;
        }

        /* Card Style */
        .paper-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
        }

        .paper-card:hover {
            transform: translateY(-6px);
            border-color: var(--card-border-hover);
            box-shadow: 0 20px 40px -15px rgba(99, 102, 241, 0.15), 0 10px 30px -10px rgba(0, 0, 0, 0.6);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }

        .paper-tag {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }

        .paper-date {
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .paper-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: 1rem;
            color: var(--text-primary);
            transition: color 0.2s ease;
        }

        .paper-card:hover .paper-title {
            color: #c7d2fe;
        }

        .paper-tldr {
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 1.5rem;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .paper-meta {
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 1.25rem;
            margin-top: auto;
        }

        .meta-item {
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .meta-item svg {
            width: 16px;
            height: 16px;
            margin-top: 2px;
            flex-shrink: 0;
            stroke: var(--text-muted);
        }

        .authors-list {
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-actions {
            margin-top: 1.5rem;
            display: flex;
            gap: 1rem;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.9rem;
            text-decoration: none;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .btn-primary {
            background: var(--accent-primary);
            color: #ffffff;
            flex-grow: 1;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
        }

        .btn-primary:hover {
            background: #4f46e5;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-secondary);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* Empty State */
        .empty-state {
            grid-column: 1 / -1;
            text-align: center;
            padding: 4rem 2rem;
            background: var(--card-bg);
            border: 1px dashed var(--card-border);
            border-radius: 16px;
            display: none;
        }

        .empty-state h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }

        .empty-state p {
            color: var(--text-muted);
        }

        /* Footer */
        footer {
            margin-top: 6rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 2rem;
        }

        footer a {
            color: var(--text-secondary);
            text-decoration: none;
        }

        footer a:hover {
            color: var(--accent-primary);
        }

        /* Responsive adaptations */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.5rem;
            }
            .hero p {
                font-size: 1rem;
            }
            .papers-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>

    <!-- Ambient glowing nodes -->
    <div class="ambient-glow-1"></div>
    <div class="ambient-glow-2"></div>

    <div class="container">
        
        <!-- Header -->
        <header>
            <div class="logo" id="headerLogo">
                <svg fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18(missing: 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
                Paper Readings
            </div>
            <div class="nav-links">
                <a href="https://github.com/PhDeasy-org/your-paper-reading-agent" target="_blank" id="githubRepoLink">
                    <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                        <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.579.688.481C19.137 20.162 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
                    </svg>
                    ppagent
                </a>
            </div>
        </header>

        <!-- Hero Section -->
        <section class="hero">
            <h1>AI Research Journal</h1>
            <p>Deep-dive agentic reports, critical audits, and illustrated summaries of trending Machine Learning papers. Generated automatically by ppagent.</p>
            
            <div class="search-container">
                <svg class="search-icon" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input type="text" id="searchInput" class="search-input" placeholder="Search papers by title, author, or keyword...">
            </div>
        </section>

        <!-- Main Papers Feed -->
        <main>
            <h2 class="section-title">
                Latest Reports
                <span id="paperCount">{{ papers|length }}</span>
            </h2>

            <div class="papers-grid" id="papersGrid">
                
                {% for paper in papers %}
                <!-- Card: {{ paper.title }} -->
                <article class="paper-card" data-title="{{ paper.title|lower }}" data-authors="{{ paper.authors|lower }}" data-keywords="{{ paper.keywords|lower }}">
                    <div>
                        <div class="card-header">
                            <span class="paper-tag">{{ paper.paper_type }}</span>
                            <span class="paper-date">
                                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                {{ paper.published_date }}
                            </span>
                        </div>
                        <h3 class="paper-title">{{ paper.title }}</h3>
                        <p class="paper-tldr">{{ paper.tldr }}</p>
                    </div>
                    
                    <div class="paper-meta">
                        <div class="meta-item">
                            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                            </svg>
                            <span class="authors-list">{{ paper.authors }}</span>
                        </div>
                        <div class="meta-item">
                            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
                            </svg>
                            <span class="keywords-list">{{ paper.keywords }}</span>
                        </div>
                        
                        <div class="card-actions">
                            <a href="{{ paper.link }}" class="btn btn-primary">
                                Read Report
                            </a>
                            {% if paper.arxiv_url %}
                            <a href="{{ paper.arxiv_url }}" target="_blank" class="btn btn-secondary">
                                arXiv
                            </a>
                            {% endif %}
                        </div>
                    </div>
                </article>
                {% else %}
                <!-- Empty State -->
                <div class="empty-state" id="emptyState" style="display: block;">
                    <h3>No reports found</h3>
                    <p>Try searching for different keywords or check back later.</p>
                </div>
                {% endfor %}

                <!-- Empty State for JS searching -->
                <div class="empty-state" id="emptyState">
                    <h3>No reports found</h3>
                    <p>Try searching for different keywords or check back later.</p>
                </div>

            </div>
        </main>

        <!-- Footer -->
        <footer>
            <p>Generated with ❤️ by <a href="https://github.com/PhDeasy-org/your-paper-reading-agent" target="_blank">ppagent</a>. Hosted on GitHub Pages.</p>
        </footer>

    </div>

    <!-- Client-side filter functionality -->
    <script>
        const searchInput = document.getElementById('searchInput');
        const papersGrid = document.getElementById('papersGrid');
        const paperCards = papersGrid.querySelectorAll('.paper-card');
        const emptyState = document.getElementById('emptyState');
        const paperCountEl = document.getElementById('paperCount');

        searchInput.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase().trim();
            let visibleCount = 0;

            paperCards.forEach(card => {
                const title = card.getAttribute('data-title') || '';
                const authors = card.getAttribute('data-authors') || '';
                const keywords = card.getAttribute('data-keywords') || '';

                if (title.includes(query) || authors.includes(query) || keywords.includes(query)) {
                    card.style.display = 'flex';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            });

            paperCountEl.textContent = visibleCount;

            if (visibleCount === 0) {
                emptyState.style.display = 'block';
            } else {
                emptyState.style.display = 'none';
            }
        });
    </script>
</body>
</html>"""
