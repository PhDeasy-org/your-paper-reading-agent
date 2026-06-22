"""Orchestrates the full paper discovery → report generation pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console

from ppagent import hf, pdf
from ppagent.agents.assembler import Assembler
from ppagent.agents.classifier import ClassifierAgent
from ppagent.agents.criticizer import CriticizerAgent
from ppagent.agents.finder import FinderAgent
from ppagent.agents.figure_selector import FigureSelectorAgent
from ppagent.agents.searcher import SearcherAgent
from ppagent.agents.writer import WriterAgent
from ppagent.config import AppConfig
from ppagent import figures as figures_mod
from ppagent.hf import HfCliError
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, Paper, PaperContent, PaperReport
from ppagent.storage import Storage

logger = logging.getLogger(__name__)


class PaperPipeline:
    """Full pipeline: search → (writer ‖ finder) → criticizer → assembler."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.console = Console()
        # One LLMClient per role; agents are wired to the role they belong to.
        self._clients = {
            "text": LLMClient(config.llms.text),
            "vision": LLMClient(config.llms.vision),
            "searcher": LLMClient(config.llms.searcher),
        }
        self.classifier = ClassifierAgent(self._clients["text"], config)
        self.searcher = SearcherAgent(self._clients["searcher"], config)
        self.writer = WriterAgent(self._clients["text"], config)
        self.finder = FinderAgent(self._clients["searcher"], config)
        self.criticizer = CriticizerAgent(self._clients["text"], config)
        self.figure_selector = FigureSelectorAgent(self._clients["vision"], config)
        self.storage = Storage(config.output_dir)
        # Map each report-stage agent name to the model it uses, so the
        # assembler can compute a per-model cost breakdown.
        from ppagent.config import AGENT_LLM_ROLE

        model_map = {
            name: config.llms.for_role(role).model
            for name, role in AGENT_LLM_ROLE.items()
        }
        self.assembler = Assembler(
            template_dir=config.template_dir,
            storage=self.storage,
            model_map=model_map,
        )

    def search(
        self,
        *,
        date: str | None = None,
        limit: int | None = None,
    ) -> list[Paper]:
        """Fetch papers from HuggingFace and score them against the user's profile.

        Args:
            date: The date to fetch papers for (e.g., 'today', '2023-10-01'). Defaults to config.
            limit: The maximum number of papers to fetch from HuggingFace. Defaults to config.

        Returns:
            A list of Paper objects sorted by relevance score, up to the maximum configured limit.
        """
        self.console.print("[bold]Searching papers matching user profile...[/bold]")
        with self.console.status(
            "[dim]Fetching papers from HuggingFace...[/dim]", spinner="dots"
        ):
            papers = hf.list_papers(
                date=date or self.config.search.default_date,
                limit=limit or self.config.search.default_limit,
                sort=self.config.search.sort,
            )
        logger.info("Fetched %d papers from HuggingFace", len(papers))

        profile_path = self.config.profile_path
        if not profile_path.exists():
            logger.warning(
                "Profile not found at %s — returning all papers", profile_path
            )
            return papers[: self.config.search.max_reports_per_run]

        profile = profile_path.read_text()
        with self.console.status(
            "[dim]Scoring paper relevance...[/dim]", spinner="dots"
        ):
            result = self.searcher.run(papers=papers, profile=profile)

        if not result.success:
            logger.error("Searcher failed: %s", result.error)
            return []

        matched = result.data["papers"]
        logger.info("Searcher matched %d / %d papers", len(matched), len(papers))
        self.console.print(f"Found [green]{len(matched)}[/green] matched papers.")
        return matched[: self.config.search.max_reports_per_run]

    def _run_agent_streaming(
        self,
        label: str,
        run_fn: Callable[..., Any],
        **kwargs: Any,
    ) -> AgentResult:
        """Run a tool-capable agent with its research phase streamed to stdout.

        Prints a labeled header, streams text deltas as they arrive (each delta
        printed inline, soft-wrapped), and finishes with a trailing newline and
        a success/failure marker. The agent's structured-output phase is silent
        — only the prose it generates via tool-calling is streamed.
        """
        self.console.print(f"[bold cyan]{label}[/bold cyan] ", end="")
        collected: list[str] = []

        def _on_text(delta: str) -> None:
            collected.append(delta)
            self.console.print(delta, end="", soft_wrap=True, highlight=False)

        try:
            result = run_fn(on_text=_on_text, **kwargs)
        except Exception as exc:
            self.console.print()  # newline after streamed/partial output
            self.console.print(f"  [red]✗[/red] {label} agent failed: {exc}")
            return AgentResult(agent_name="", success=False, error=str(exc))

        if collected:
            self.console.print()  # newline after streamed prose
        if result.success:
            self.console.print(
                f"  [green]✓[/green] {label} agent completed successfully."
            )
        else:
            self.console.print(f"  [red]✗[/red] {label} agent failed: {result.error}")
        return result

    def report(self, paper_id: str) -> PaperReport:
        """Generate a comprehensive report for a single paper.

        This method orchestrates a multi-agent workflow:
        1. Fetch metadata and full text.
        2. Classify the paper type.
        3. Run Writer and Finder agents (in parallel by default, or
           sequentially with live streaming when ``config.report.stream``).
        4. Run Criticizer agent on the Writer's output to find weaknesses.
        5. Extract and select figures using the Vision LLM.
        6. Assemble the final report.

        Args:
            paper_id: The arXiv ID or HuggingFace paper ID to process.

        Returns:
            A populated PaperReport object containing the generated sections.
        """
        self.console.print("\n[bold cyan]🚀 Starting Report Generation[/bold cyan]")
        self.console.print(f"[bold]Paper ID:[/bold] {paper_id}")
        self.console.print("[bold]LLM configuration in use:[/bold]")
        self.console.print(
            f"  • [bold]Text model (Writer/Finder/Criticizer):[/bold] [cyan]{self.config.llms.text.model}[/cyan]"
        )
        self.console.print(
            f"    [dim]Base URL: {self.config.llms.text.base_url} | Temperature: {self.config.llms.text.temperature} | Max tokens: {self.config.llms.text.max_tokens} | Thinking: {self.config.llms.text.enable_thinking}[/dim]"
        )
        self.console.print(
            f"  • [bold]Vision model (Figure Selector):[/bold] [cyan]{self.config.llms.vision.model}[/cyan]"
        )
        self.console.print(
            f"    [dim]Base URL: {self.config.llms.vision.base_url} | Temperature: {self.config.llms.vision.temperature} | Max tokens: {self.config.llms.vision.max_tokens} | Thinking: {self.config.llms.vision.enable_thinking}[/dim]"
        )

        # Fetch paper info
        self.console.print(
            "\n[bold yellow]🔄 Phase 1/8: Fetching paper metadata...[/bold yellow]"
        )
        with self.console.status(
            "[dim]Contacting HuggingFace/arXiv APIs...[/dim]", spinner="dots"
        ):
            try:
                paper = hf.paper_info(paper_id)
                self.console.print(
                    f'  [green]✓[/green] Found paper on HuggingFace: [bold]"{paper.title}"[/bold]'
                )
            except HfCliError:
                self.console.print(
                    "  [dim]HuggingFace metadata fetch failed, trying arXiv API fallback...[/dim]"
                )
                arxiv_paper = hf.fetch_arxiv_info(paper_id)
                if arxiv_paper:
                    paper = arxiv_paper
                    self.console.print(
                        f'  [green]✓[/green] Found paper on arXiv: [bold]"{paper.title}"[/bold]'
                    )
                else:
                    self.console.print(
                        "  [yellow]⚠[/yellow] Metadata fetch failed on both HuggingFace and arXiv, using minimal placeholder."
                    )
                    paper = Paper(id=paper_id, title=paper_id)

        # Get paper content: try hf papers read first, fall back to PDF
        self.console.print(
            "[bold yellow]🔄 Phase 2/8: Retrieving paper full text content...[/bold yellow]"
        )
        content_md = ""
        pdf_path = None
        with self.console.status(
            "[dim]Retrieving full text from HuggingFace or extracting from PDF...[/dim]",
            spinner="dots",
        ):
            try:
                content_md = hf.paper_read(paper_id)
                self.console.print(
                    f"  [green]✓[/green] Successfully retrieved paper content via HuggingFace API ({len(content_md)} characters)"
                )
            except HfCliError:
                self.console.print(
                    "  [dim]HuggingFace paper read failed. Attempting to download PDF and extract text...[/dim]"
                )
                if self.config.report.download_pdf:
                    try:
                        self.console.print(
                            f"  Downloading PDF to: [dim]{self.config.pdf_cache_dir}[/dim]"
                        )
                        pdf_path = pdf.download_pdf(paper, self.config.pdf_cache_dir)
                        self.console.print("  Extracting text from PDF...")
                        content_md = pdf.extract_text(pdf_path)
                        self.console.print(
                            f"  [green]✓[/green] Successfully extracted PDF text ({len(content_md)} characters)"
                        )
                    except Exception as pdf_exc:
                        self.console.print(
                            f"  [red]✗[/red] PDF download/extraction failed: {pdf_exc}"
                        )

        if not content_md:
            self.console.print(
                "  [yellow]⚠[/yellow] No full text content available. Falling back to using paper abstract/summary."
            )
            content_md = paper.summary or "Paper content unavailable."

        paper_content = PaperContent(paper=paper, markdown=content_md)

        # Classify paper type
        self.console.print(
            "[bold yellow]🔄 Phase 3/8: Classifying paper type...[/bold yellow]"
        )
        paper_type = "method"  # default fallback
        with self.console.status(
            "[dim]Running Classifier LLM...[/dim]", spinner="dots"
        ):
            classifier_result = self.classifier.run(content=paper_content)
            if classifier_result.success:
                paper_type = classifier_result.data.get("paper_type", "method")
                confidence = classifier_result.data.get("confidence", 0.0)
                reasoning = classifier_result.data.get("reasoning", "")
                self.console.print(
                    f"  [green]✓[/green] Paper classified as: [bold cyan]{paper_type}[/bold cyan] (confidence: {confidence:.0%})"
                )
                if reasoning:
                    self.console.print(f"    [dim]{reasoning}[/dim]")
            else:
                self.console.print(
                    f"  [yellow]⚠[/yellow] Classification failed ({classifier_result.error}); defaulting to 'method'"
                )

        # Run writer and finder
        self.console.print(
            "[bold yellow]🔄 Phase 4/8: Running Writer and Finder agents...[/bold yellow]"
        )
        writer_result: AgentResult | None = None
        finder_result: AgentResult | None = None

        if self.config.report.stream:
            # Streaming: run the two agents sequentially so their text deltas
            # don't interleave. Each agent's research/narrative phase streams
            # under a labeled header; the final structured phase stays quiet.
            writer_result = self._run_agent_streaming(
                "📝 Writer",
                self.writer.run,
                content=paper_content,
                paper_type=paper_type,
            )
            finder_result = self._run_agent_streaming(
                "🔍 Finder",
                self.finder.run,
                content=paper_content,
            )
        else:
            with self.console.status(
                "[dim]Running Writer and Finder agents concurrently...[/dim]",
                spinner="dots",
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    self.console.print(
                        f"  Running Writer agent using text model: [cyan]{self.config.llms.text.model}[/cyan]..."
                    )
                    writer_future = executor.submit(
                        self.writer.run, content=paper_content, paper_type=paper_type
                    )
                    self.console.print(
                        f"  Running Finder agent using text model: [cyan]{self.config.llms.text.model}[/cyan]..."
                    )
                    finder_future = executor.submit(
                        self.finder.run, content=paper_content
                    )

                    for future in as_completed([writer_future, finder_future]):
                        result = future.result()
                        if result.agent_name == "writer":
                            writer_result = result
                            if result.success:
                                self.console.print(
                                    "  [green]✓[/green] Writer agent completed successfully."
                                )
                            else:
                                self.console.print(
                                    f"  [red]✗[/red] Writer agent failed: {result.error}"
                                )
                        elif result.agent_name == "finder":
                            finder_result = result
                            if result.success:
                                self.console.print(
                                    "  [green]✓[/green] Finder agent completed successfully."
                                )
                            else:
                                self.console.print(
                                    f"  [red]✗[/red] Finder agent failed: {result.error}"
                                )

        # Ensure results exist
        if writer_result is None:
            writer_result = AgentResult(
                agent_name="writer", success=False, error="Writer did not complete"
            )
        if finder_result is None:
            finder_result = AgentResult(
                agent_name="finder", success=False, error="Finder did not complete"
            )

        # Criticizer depends on writer output
        self.console.print(
            "[bold yellow]🔄 Phase 5/8: Running Criticizer agent to refine report...[/bold yellow]"
        )
        with self.console.status(
            "[dim]Running Criticizer LLM...[/dim]", spinner="dots"
        ):
            self.console.print(
                f"  Running Criticizer agent using text model: [cyan]{self.config.llms.text.model}[/cyan]..."
            )
            criticizer_result = self.criticizer.run(
                content=paper_content,
                writer_sections=writer_result.data if writer_result.success else None,
                paper_type=paper_type,
            )
            if criticizer_result.success:
                self.console.print(
                    "  [green]✓[/green] Criticizer agent completed successfully."
                )
            else:
                self.console.print(
                    f"  [red]✗[/red] Criticizer agent failed: {criticizer_result.error}"
                )

        # Ensure we have the PDF downloaded for figure extraction.
        # (hf papers read may have succeeded without a local PDF.)
        self.console.print(
            "[bold yellow]🔄 Phase 6/8: Ensuring PDF is downloaded for figure extraction...[/bold yellow]"
        )
        with self.console.status(
            "[dim]Checking and downloading PDF if necessary...[/dim]", spinner="dots"
        ):
            if pdf_path is None and self.config.report.download_pdf:
                try:
                    self.console.print(
                        f"  Downloading PDF to: [dim]{self.config.pdf_cache_dir}[/dim]"
                    )
                    pdf_path = pdf.download_pdf(paper, self.config.pdf_cache_dir)
                    self.console.print(
                        f"  [green]✓[/green] PDF downloaded successfully to [dim]{pdf_path}[/dim]"
                    )
                except Exception as exc:
                    self.console.print(
                        f"  [yellow]⚠[/yellow] Could not download PDF for figures: {exc}"
                    )
                    pdf_path = None
            elif pdf_path is not None:
                self.console.print(
                    f"  [green]✓[/green] PDF already available at [dim]{pdf_path}[/dim]"
                )
            else:
                self.console.print(
                    "  [dim]PDF downloading is disabled; skipping PDF check.[/dim]"
                )

        # Extract captioned figures and let the vision LLM decide which (if any)
        # to insert. The LLM may pick zero, one, or several figures and assign
        # each to the report section it best illustrates.
        # Figures are written into the paper's report directory so they sit
        # next to report.html and can be referenced by a relative path.
        self.console.print(
            "[bold yellow]🔄 Phase 7/8: Extracting and selecting paper figures...[/bold yellow]"
        )
        paper_dir = self.storage.paper_dir(paper.title, paper.published_at)
        selected_figures: list[figures_mod.SelectedFigure] = []
        figure_selector_result: AgentResult | None = None
        if pdf_path is not None:
            with self.console.status(
                "[dim]Extracting figures and running Vision LLM selector...[/dim]",
                spinner="dots",
            ):
                try:
                    self.console.print("  Extracting figures from PDF...")
                    figures = figures_mod.extract_figures(pdf_path, paper_dir)
                    self.console.print(
                        f"  Found {len(figures)} figures/tables in the PDF."
                    )
                except Exception as exc:
                    self.console.print(
                        f"  [red]✗[/red] Figure extraction failed: {exc}"
                    )
                    figures = []
                if figures:
                    self.console.print(
                        f"  Running figure selector agent using vision model: [cyan]{self.config.llms.vision.model}[/cyan]..."
                    )
                    figure_selector_result = self.figure_selector.run(
                        figures=figures, base_dir=paper_dir
                    )
                    if figure_selector_result.success:
                        selected_figures = figure_selector_result.data.get(
                            "selected_figures", []
                        )
                        if selected_figures:
                            self.console.print(
                                f"  [green]✓[/green] Selected {len(selected_figures)} figure(s):"
                            )
                            for sf in selected_figures:
                                fig_title = (
                                    sf.figure.caption[:60] + "..."
                                    if len(sf.figure.caption) > 60
                                    else sf.figure.caption
                                )
                                self.console.print(
                                    f"    • Figure {sf.figure.figure_number} → [cyan]{sf.section}[/cyan] - [italic]{fig_title}[/italic]"
                                )
                        else:
                            none_reason = figure_selector_result.data.get("none_reason")
                            reason_msg = (
                                f" (reason: {none_reason})" if none_reason else ""
                            )
                            self.console.print(
                                f"  [dim]Figure selector chose not to insert any figures{reason_msg}.[/dim]"
                            )
                    else:
                        self.console.print(
                            f"  [red]✗[/red] Figure selector agent failed: {figure_selector_result.error}"
                        )
        else:
            self.console.print(
                "  [dim]No PDF path available; skipping figure extraction.[/dim]"
            )

        # Assemble
        self.console.print(
            "[bold yellow]🔄 Phase 8/8: Assembling final report...[/bold yellow]"
        )
        with self.console.status(
            "[dim]Formatting, generating LaTeX equations, and writing report files...[/dim]",
            spinner="dots",
        ):
            report, md_content, html_content = self.assembler.assemble(
                paper=paper,
                writer_result=writer_result,
                finder_result=finder_result,
                criticizer_result=criticizer_result,
                figure_selector_result=figure_selector_result,
                classifier_result=classifier_result,
                selected_figures=selected_figures or None,
                paper_type=paper_type,
            )
            self.console.print("  [green]✓[/green] Report assembled successfully!")

        # Publishing is opt-in (publish.enabled) and best-effort: each enabled
        # publisher runs independently and failures are non-fatal.
        self._publish_report(
            report,
            md_content=md_content,
            html_content=html_content,
            report_dir=paper_dir,
        )

        logger.info("Report generated for %s: %s", paper_id, paper.title)
        return report

    def _publish_report(
        self,
        report: PaperReport,
        *,
        md_content: str,
        html_content: str,
        report_dir: Path,
    ) -> None:
        """Run every enabled publisher against the freshly generated report.

        Each publisher is configured in ``config.publish.<name>``; this method
        instantiates the enabled ones from their config sub-models and invokes
        them. A publisher failure is logged but never aborts the pipeline or
        sibling publishers.
        """
        if not self.config.publish.enabled:
            return

        from ppagent.publishers import get_publisher

        # name → the config sub-model whose fields seed the publisher's ctor.
        targets = [
            ("notion", self.config.publish.notion),
            ("wechat", self.config.publish.wechat),
            ("github_pages", self.config.publish.github_pages),
        ]
        enabled = [(name, cfg) for name, cfg in targets if cfg.enabled]
        if not enabled:
            logger.debug("Publishing enabled but no individual publisher enabled.")
            return

        self.console.print("\n[bold cyan]📣 Publishing report...[/bold cyan]")
        for name, cfg in enabled:
            label = name.replace("_", " ").title()
            try:
                # ``enabled`` is a config-only flag, not a publisher ctor param;
                # drop it before forwarding the rest of the config fields.
                publisher = get_publisher(name, **cfg.model_dump(exclude={"enabled"}))
                ok = publisher.publish(
                    report,
                    md_content=md_content,
                    html_content=html_content,
                    report_dir=report_dir,
                )
            except Exception as exc:  # never let a publisher abort the run
                logger.exception("%s publisher raised", name)
                ok = False
                err = str(exc)
            else:
                err = ""
            if ok:
                self.console.print(f"  [green]✓[/green] {label}")
            else:
                msg = f" ({err})" if err else ""
                self.console.print(f"  [red]✗[/red] {label}{msg}")

    def run(
        self,
        *,
        date: str | None = None,
        limit: int | None = None,
        prompt_replace: bool = True,
    ) -> list[PaperReport]:
        """Execute the full pipeline: discover relevant papers and generate reports for them.

        This combines `search()` and `report()`. It will skip papers that already have
        reports generated unless `prompt_replace` is True and the user confirms.

        Args:
            date: The date to fetch papers for.
            limit: The maximum number of papers to evaluate.
            prompt_replace: If True, prompt the user via CLI before regenerating an existing report.

        Returns:
            A list of PaperReport objects that were successfully generated.
        """
        papers = self.search(date=date, limit=limit)
        if not papers:
            logger.info("No papers to process.")
            return []

        reports: list[PaperReport] = []
        for paper in papers:
            if self.storage.report_exists(paper.title, paper.published_at):
                if prompt_replace:
                    try:
                        answer = (
                            input(
                                f'Report for "{paper.title}" already exists. Regenerate? [y/N] '
                            )
                            .strip()
                            .lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        answer = ""
                    if answer != "y":
                        logger.info("Skipping %s", paper.title)
                        continue
                else:
                    logger.info("Skipping %s — report already exists", paper.title)
                    continue

            try:
                report = self.report(paper.id)
                reports.append(report)
            except Exception as exc:
                logger.error("Failed to generate report for %s: %s", paper.id, exc)

        return reports
