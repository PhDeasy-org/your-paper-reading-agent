"""CLI interface for ppagent."""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ppagent import __version__
from ppagent.config import AppConfig, load_config, PROJECT_ROOT

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
        console.print("Run [bold]ppagent config init[/bold] to create a default config.")
        raise typer.Exit(1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ppagent {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)


# ─── search ──────────────────────────────────────────────────────────────────


@app.command()
def search(
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Paper date (YYYY-MM-DD or 'today')."),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max papers to fetch."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Path to profile .md file."),
) -> None:
    """Discover and rank papers based on your research profile."""
    from ppagent import hf
    from ppagent.llm import LLMClient
    from ppagent.agents.searcher import SearcherAgent

    cfg = _load()
    if profile:
        cfg.search.profile_path = profile

    console.print("[bold]Fetching papers from HuggingFace...[/bold]")
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
        console.print("Edit [bold]config/profile.md[/bold] with your research interests.")
        raise typer.Exit(1)

    profile_text = cfg.profile_path.read_text()
    llm = LLMClient(cfg.llm)
    searcher = SearcherAgent(llm, cfg)
    result = searcher.run(papers=papers, profile=profile_text)

    if not result.success:
        console.print(f"[red]Searcher failed:[/red] {result.error}")
        raise typer.Exit(1)

    matched = result.data["papers"]
    scores = result.data["scores"]

    if not matched:
        console.print("[yellow]No papers matched your profile above the threshold.[/yellow]")
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
    paper_id: str = typer.Argument(..., help="Paper ID (e.g. 2506.12345) or arXiv URL."),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory."),
) -> None:
    """Generate a detailed report for a specific paper."""
    from ppagent.pipeline import PaperPipeline

    cfg = _load()
    if output_dir:
        cfg.report.output_dir = output_dir

    # Extract ID from URL if needed
    if "/" in paper_id:
        paper_id = paper_id.rstrip("/").split("/")[-1]

    console.print(f"[bold]Generating report for paper:[/bold] {paper_id}")

    pipeline = PaperPipeline(cfg)
    try:
        paper_report = pipeline.report(paper_id)
    except Exception as exc:
        console.print(f"[red]Report generation failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]Report generated![/green] Output: {cfg.output_dir / paper_report.paper.id}")


# ─── run ─────────────────────────────────────────────────────────────────────


@app.command()
def run(
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Paper date."),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max papers to fetch."),
    schedule: bool = typer.Option(False, "--schedule", "-s", help="Enable auto-fetch scheduler."),
) -> None:
    """Run the full pipeline: search + report generation."""
    if schedule:
        from ppagent.scheduler import PaperScheduler
        from ppagent.pipeline import PaperPipeline

        cfg = _load()
        pipeline = PaperPipeline(cfg)
        scheduler = PaperScheduler(cfg, pipeline)
        console.print("[bold]Starting scheduler...[/bold]")
        console.print(f"  Cron: {cfg.scheduler.cron_hour:02d}:{cfg.scheduler.cron_minute:02d} ({cfg.scheduler.timezone})")
        console.print("  Press Ctrl+C to stop.\n")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Scheduler stopped.[/yellow]")
        return

    from ppagent.pipeline import PaperPipeline

    cfg = _load()
    console.print("[bold]Running full pipeline...[/bold]")

    pipeline = PaperPipeline(cfg)
    try:
        reports = pipeline.run(date=date, limit=limit)
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"\n[green]Done![/green] Generated {len(reports)} report(s).")
    for r in reports:
        console.print(f"  - {r.paper.title} → {cfg.output_dir / r.paper.id}")


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
    console.print(f"[bold]Config loaded from:[/bold] {PROJECT_ROOT / 'config' / 'settings.toml'}")
    console.print(f"  LLM: {cfg.llm.model} @ {cfg.llm.base_url}")
    console.print(f"  Profile: {cfg.profile_path}")
    console.print(f"  Output: {cfg.output_dir}")
    console.print(f"  Language: {cfg.report.language}")
    console.print(f"  Scheduler: {'enabled' if cfg.scheduler.enabled else 'disabled'}")
    console.print(f"  Publishing: {'enabled' if cfg.publish.enabled else 'disabled'}")


@config_app.command("init")
def config_init() -> None:
    """Create a default settings.toml if it doesn't exist."""
    import tomli_w

    target = PROJECT_ROOT / "config" / "settings.toml"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        console.print(f"[yellow]Config already exists:[/yellow] {target}")
        return

    default = {
        "llm": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-your-key-here",
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 4096,
            "timeout": 120,
        },
        "search": {
            "default_date": "today",
            "default_limit": 50,
            "sort": "trending",
            "profile_path": "config/profile.md",
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

    with open(target, "wb") as f:
        tomli_w.dump(default, f)

    console.print(f"[green]Created config:[/green] {target}")
    console.print("Edit it to add your API key and customize settings.")


# ─── entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    app()
