"""CLI interface for ppagent."""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ppagent import __version__
from ppagent.config import (
    AppConfig,
    load_config,
    PROJECT_ROOT,
    CONFIG_PATH,
    CONFIG_DIR,
)
from ppagent.storage import Storage

app = typer.Typer(
    name="ppagent",
    help="Personalized paper discovery and report generation agents.",
    add_completion=True,
    no_args_is_help=True,
)

config_app = typer.Typer(help="Manage ppagent configuration.")
app.add_typer(config_app, name="config")

console = Console()
logger = logging.getLogger(__name__)


def _load() -> AppConfig:
    """Load config with a friendly error if missing."""
    try:
        return load_config()
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print(
            "Run [bold]ppagent config init[/bold] to create a default config."
        )
        raise typer.Exit(1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ppagent {__version__}")
        raise typer.Exit()


def _normalize_paper_id(paper_id: str) -> str:
    """Normalize a user-supplied paper id to a bare arXiv id.

    Accepts any of:
      - ``2506.12345`` / ``2506.12345v2``           → ``2506.12345``
      - ``arxiv:2506.12345`` / ``ARXIV:2506.12345`` → ``2506.12345``
      - ``https://arxiv.org/abs/2506.12345``        → ``2506.12345``
      - ``https://arxiv.org/pdf/2506.12345v2``      → ``2506.12345``
      - ``https://huggingface.co/papers/2506.12345``→ ``2506.12345``
    """
    import re

    pid = paper_id.strip()
    # Strip URL path prefixes first.
    pid = pid.rstrip("/").split("/")[-1]
    # Drop an explicit "arxiv:" scheme, version suffix, or a leading "abs/"/"pdf/".
    pid = re.sub(r"^(?i:arxiv:)", "", pid)
    pid = re.sub(r"^(?i:abs|pdf)/", "", pid)
    pid = re.sub(r"v\d+$", "", pid)
    return pid


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)


# ─── search ──────────────────────────────────────────────────────────────────


@app.command()
def search(
    date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Paper date (YYYY-MM-DD or 'today')."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Max papers to fetch."
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Path to profile .md file."
    ),
) -> None:
    """Discover and rank papers based on your research profile.

    Fetches the latest papers from HuggingFace and uses the Searcher LLM
    to score them against the interests defined in your profile.md.

    Options:
    - --date: Specific date to fetch (e.g. '2023-10-01'). Defaults to config.
    - --limit: Max number of papers to fetch for scoring.
    - --profile: Override the profile.md path for scoring.
    """
    from ppagent import hf
    from ppagent.llm import LLMClient
    from ppagent.agents.searcher import SearcherAgent

    cfg = _load()
    if profile:
        cfg.search.profile_path = profile

    console.print("[bold]Fetching papers from HuggingFace...[/bold]")
    with console.status("[dim]Contacting HuggingFace API...[/dim]", spinner="dots"):
        try:
            papers = hf.list_papers(
                date=date or cfg.search.default_date,
                limit=limit or cfg.search.default_limit,
                sort=cfg.search.sort,
            )
        except hf.HfCliError as exc:
            console.print(f"[red]Error fetching papers:[/red] {exc}")
            raise typer.Exit(1)

    console.print(f"Found [green]{len(papers)}[/green] papers. Scoring relevance...")

    if not cfg.profile_path.exists():
        console.print(f"[red]Profile not found:[/red] {cfg.profile_path}")
        console.print(
            "Edit [bold]config/profile.md[/bold] with your research interests."
        )
        raise typer.Exit(1)

    profile_text = cfg.profile_path.read_text()
    llm = LLMClient(cfg.llms.searcher)
    searcher = SearcherAgent(llm, cfg)
    with console.status(
        "[dim]Scoring paper relevance via Searcher LLM...[/dim]", spinner="dots"
    ):
        result = searcher.run(papers=papers, profile=profile_text)

    if not result.success:
        console.print(f"[red]Searcher failed:[/red] {result.error}")
        raise typer.Exit(1)

    matched = result.data["papers"]
    scores = result.data["scores"]

    if not matched:
        console.print(
            "[yellow]No papers matched your profile above the threshold.[/yellow]"
        )
        return

    table = Table(title=f"Matched Papers ({len(matched)})", show_lines=True)
    table.add_column("Score", style="bold cyan", width=6)
    table.add_column("Title", style="bold")
    table.add_column("ID", style="dim")

    for paper in matched:
        table.add_row(
            f"{scores.get(paper.id, 0):.2f}",
            paper.title,
            paper.id,
        )

    console.print(table)


# ─── report ──────────────────────────────────────────────────────────────────


@app.command()
def report(
    paper_ids: list[str] = typer.Argument(
        ..., help="Paper ID(s) (e.g. 2506.12345) or arXiv URL(s)."
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Regenerate without prompting if report already exists.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the report in the default browser after generation.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream the Writer/Finder research phase to the terminal as it is generated.",
    ),
) -> None:
    """Generate detailed multi-agent reports for specified papers.

    Runs the full analysis pipeline (classification, writing, finding related works,
    critique, and figure extraction) for one or more papers.

    Arguments:
    - paper_ids: One or more arXiv IDs or URLs to process.

    Options:
    - --output, -o: Override the output directory for generated reports.
    - --force, -f: Regenerate the report even if it already exists, without asking.
    - --open/--no-open: Open the resulting HTML report in the browser (default: True).
    - --stream: Stream the Writer/Finder research-phase prose to the terminal live.
    """
    from ppagent import hf
    from ppagent.pipeline import PaperPipeline

    cfg = _load()
    if output_dir:
        cfg.report.output_dir = output_dir
    cfg.report.stream = stream

    pipeline = PaperPipeline(cfg)
    has_errors = False

    for paper_id in paper_ids:
        # Normalize arxiv: / URL / version suffixes to a bare arXiv id.
        paper_id = _normalize_paper_id(paper_id)

        console.print(f"\n[bold]Generating report for paper:[/bold] {paper_id}")

        if not force:
            try:
                paper = hf.paper_info(paper_id)
            except Exception:
                paper = None
            if paper and pipeline.storage.report_exists(
                paper.title, paper.published_at
            ):
                if not typer.confirm(
                    f'Report for "{paper.title}" already exists. Regenerate?'
                ):
                    console.print("[yellow]Skipped.[/yellow]")
                    continue

        try:
            paper_report = pipeline.report(paper_id)
        except Exception as exc:
            console.print(f"[red]Report generation failed for {paper_id}:[/red] {exc}")
            has_errors = True
            continue

        report_dir = cfg.output_dir / Storage._safe_filename(
            paper_report.paper.title, paper_report.paper.published_at
        )
        console.print(f"[green]Report generated![/green] Output: {report_dir}")

        if open_browser:
            html_path = report_dir / "report.html"
            if html_path.exists():
                import webbrowser

                console.print(
                    f'Opening report for "{paper_report.paper.title}" in default browser...'
                )
                webbrowser.open(html_path.resolve().as_uri())

    if has_errors:
        raise typer.Exit(1)


# ─── run ─────────────────────────────────────────────────────────────────────


@app.command()
def run(
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Paper date."),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Max papers to fetch."
    ),
    schedule: bool = typer.Option(
        False, "--schedule", "-s", help="Enable auto-fetch scheduler."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Regenerate without prompting if reports already exist.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the report in the default browser after generation.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream the Writer/Finder research phase to the terminal as it is generated.",
    ),
) -> None:
    """Execute the full paper discovery and report generation pipeline.

    This command runs `search` to find relevant papers based on your profile,
    and then automatically runs `report` on the top matching papers.

    Options:
    - --date, -d: Specific date to fetch papers for.
    - --limit, -n: Max papers to evaluate from the source.
    - --schedule, -s: Run continuously using the configured cron schedule.
    - --force, -f: Overwrite existing reports without prompting.
    - --open/--no-open: Open generated reports in the browser.
    - --stream: Stream the Writer/Finder research-phase prose to the terminal live.
    """
    if schedule:
        from ppagent.scheduler import PaperScheduler
        from ppagent.pipeline import PaperPipeline

        cfg = _load()
        cfg.report.stream = stream
        pipeline = PaperPipeline(cfg)
        scheduler = PaperScheduler(cfg, pipeline)
        console.print("[bold]Starting scheduler...[/bold]")
        console.print(
            f"  Cron: {cfg.scheduler.cron_hour:02d}:{cfg.scheduler.cron_minute:02d} ({cfg.scheduler.timezone})"
        )
        console.print("  Press Ctrl+C to stop.\n")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Scheduler stopped.[/yellow]")
        return

    from ppagent.pipeline import PaperPipeline

    cfg = _load()
    cfg.report.stream = stream
    console.print("[bold]Running full pipeline...[/bold]")

    pipeline = PaperPipeline(cfg)
    try:
        reports = pipeline.run(date=date, limit=limit, prompt_replace=not force)
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"\n[green]Done![/green] Generated {len(reports)} report(s).")
    for r in reports:
        report_dir = cfg.output_dir / Storage._safe_filename(
            r.paper.title, r.paper.published_at
        )
        console.print(f"  - {r.paper.title} → {report_dir}")
        if open_browser:
            html_path = report_dir / "report.html"
            if html_path.exists():
                import webbrowser

                console.print(f'Opening report for "{r.paper.title}" in browser...')
                webbrowser.open(html_path.resolve().as_uri())


# ─── config commands ─────────────────────────────────────────────────────────


@config_app.callback(invoke_without_command=True)
def config_main(ctx: typer.Context) -> None:
    """Manage ppagent configuration."""
    if ctx.invoked_subcommand is None:
        from ppagent.tui import run_config_tui

        run_config_tui()


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    cfg = _load()
    console.print(f"[bold]Config loaded from:[/bold] {CONFIG_PATH}")
    console.print(
        f"  Text LLM (writer/finder/criticizer): {cfg.llms.text.model} @ {cfg.llms.text.base_url}"
    )
    console.print(
        f"  Vision LLM (figure_selector):       {cfg.llms.vision.model} @ {cfg.llms.vision.base_url}"
    )
    console.print(
        f"  Searcher LLM (paper scoring):        {cfg.llms.searcher.model} @ {cfg.llms.searcher.base_url}"
    )
    console.print(f"  Profile: {cfg.profile_path}")
    console.print(f"  Output: {cfg.output_dir}")
    console.print(f"  Language: {cfg.report.language}")
    console.print(f"  Scheduler: {'enabled' if cfg.scheduler.enabled else 'disabled'}")
    console.print(f"  Publishing: {'enabled' if cfg.publish.enabled else 'disabled'}")


@config_app.command("init")
def config_init() -> None:
    """Create a default settings.toml in ``~/.config/ppagent/`` if absent.

    The config lives entirely outside the project tree (at ``CONFIG_PATH``)
    so it survives reinstalls. This command writes only to that single path:
    if a config already exists there, it does nothing. When the config dir is
    empty it also seeds a starter ``profile.md`` from the bundled example if
    available. It never writes anything into the project directory's
    ``config/`` folder.
    """
    import copy
    import shutil
    import tomli_w

    # One source of truth. Never touch the project tree for config.
    if CONFIG_PATH.exists():
        console.print(f"[yellow]Config already exists:[/yellow] {CONFIG_PATH}")
        return

    # Per-role LLM defaults: text (writer/finder/criticizer), vision
    # (figure_selector), searcher (paper scoring). By default all three point
    # at the same OpenAI endpoint so a new user only edits one api_key.
    _llm_default = {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-your-key-here",
        "model": "gpt-4o",
        "temperature": 1.0,
        "max_tokens": 16384,
        "timeout": 120,
        "instructor_mode": "auto",
        "enable_thinking": False,
    }

    default = {
        "llms": {
            # text & searcher default to the same model; vision must be a
            # vision-capable model (gpt-4o is). Users can split roles later.
            "text": copy.deepcopy(_llm_default),
            "vision": copy.deepcopy(_llm_default),
            "searcher": copy.deepcopy(_llm_default),
        },
        "search": {
            "default_date": "today",
            "default_limit": 50,
            "sort": "trending",
            "profile_path": "~/.config/ppagent/profile.md",
            "relevance_threshold": 0.6,
            "max_reports_per_run": 5,
        },
        "report": {
            "output_dir": "output",
            "template_dir": "templates",
            "formats": ["md", "html"],
            "download_pdf": True,
            "pdf_cache_dir": ".cache/pdfs",
            "language": "English",
        },
        "scheduler": {
            "enabled": False,
            "cron_hour": 8,
            "cron_minute": 0,
            "timezone": "Asia/Shanghai",
        },
        "publish": {
            "enabled": False,
        },
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(default, f)

    # Seed a starter profile.md from the bundled example, if available, so the
    # user has a concrete file to edit. Never overwrite an existing one.
    profile_path = CONFIG_DIR / "profile.md"
    if not profile_path.exists():
        bundled = PROJECT_ROOT / "config" / "profile.md"
        try:
            if bundled.exists():
                shutil.copy2(bundled, profile_path)
        except OSError:
            pass  # non-fatal — the user can create it via the TUI

    console.print(f"[green]Created config:[/green] {CONFIG_PATH}")
    console.print("Edit it (or run `ppagent config`) to add your API key.")


# ─── entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    app()
