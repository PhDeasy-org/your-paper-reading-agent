from __future__ import annotations

import textwrap
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


@dataclass
class PhaseInfo:
    """Tracks the lifecycle of a single pipeline phase."""

    name: str
    start_time: float
    logs: list[str] = field(default_factory=list)
    status: str = "running"  # "running" | "success" | "failed" | "warning"
    summary: str = ""
    elapsed: float = 0.0


# Status icons for each phase state.
_STATUS_ICONS: dict[str, str] = {
    "running": "[bold cyan]⟳[/bold cyan]",
    "success": "[green]✓[/green]",
    "failed": "[red]✗[/red]",
    "warning": "[yellow]⚠[/yellow]",
}


class PipelineConsoleWrapper:
    """Wraps rich Console to render a phase-by-phase Live layout.

    The display is divided into three sections:

    1. **Completed phases** — compact one-line summaries with elapsed time.
    2. **Current phase** — a scrolling log of the active phase's output.
    3. **LLM stream panel** — a slim auto-scrolling area showing the live
       token stream from the currently active LLM agent.
    """

    def __init__(self, original_console: Console) -> None:
        self._console = original_console
        # Phase tracking.
        self._completed_phases: list[PhaseInfo] = []
        self._current_phase: PhaseInfo | None = None
        # Lines printed before any phase begins (header info).
        self._header_lines: list[str] = []
        # Active status message (shown between panels).
        self._active_status: str | None = None
        # Per-agent streamed text, split into answer content and reasoning
        # trace.  Reasoning models (GLM/Grok/…) emit a long thinking trace
        # on a separate field before the answer; keeping them separate lets us
        # render reasoning dimmed so the user sees live activity without it
        # looking like the final output.
        self._active_streams: dict[str, dict[str, str]] = {}
        # RLock (not Lock): several methods call _render_layout() while
        # already holding _lock (start_live, status, update_stream,
        # suspend_live), and _render_layout re-acquires it.  A plain Lock
        # self-deadlocks there, freezing the whole pipeline so nothing ever
        # reaches the terminal.
        self._lock = threading.RLock()
        self._live: Live | None = None

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def begin_phase(self, name: str) -> None:
        """Start a new pipeline phase.

        If a previous phase is still running, it is automatically ended with
        ``status="success"`` and an empty summary so the display transitions
        cleanly.
        """
        with self._lock:
            if self._current_phase is not None:
                self._finish_current_phase(success=True)
            self._current_phase = PhaseInfo(name=name, start_time=time.monotonic())
        if self._live:
            self._live.update(self._render_layout())

    def end_phase(
        self,
        *,
        success: bool = True,
        summary: str = "",
        warning: bool = False,
    ) -> None:
        """End the current phase with a result status and optional summary."""
        with self._lock:
            if self._current_phase is None:
                return
            if warning:
                status = "warning"
            elif success:
                status = "success"
            else:
                status = "failed"
            self._finish_current_phase(success=success, summary=summary, status=status)
        if self._live:
            self._live.update(self._render_layout())

    def _finish_current_phase(
        self,
        *,
        success: bool = True,
        summary: str = "",
        status: str | None = None,
    ) -> None:
        """Move the current phase into ``_completed_phases``.

        Must be called while holding ``_lock``.
        """
        phase = self._current_phase
        if phase is None:
            return
        phase.elapsed = time.monotonic() - phase.start_time
        phase.summary = summary
        phase.status = status or ("success" if success else "failed")
        self._completed_phases.append(phase)
        self._current_phase = None

    # ------------------------------------------------------------------
    # Live display management
    # ------------------------------------------------------------------

    def start_live(self) -> None:
        """Start the live UI panel."""
        import sys

        if "pytest" in sys.modules or not sys.stdout.isatty():
            return
        with self._lock:
            if self._live is not None:
                return
            self._live = Live(
                self._render_layout(),
                console=self._console,
                refresh_per_second=10,
                transient=False,
                auto_refresh=True,
            )
            self._live.start()

    def stop_live(self) -> None:
        """Stop the live UI panel."""
        with self._lock:
            if self._live is None:
                return
            self._live.stop()
            self._live = None

    # ------------------------------------------------------------------
    # Layout rendering
    # ------------------------------------------------------------------

    def _render_layout(self) -> Group:
        with self._lock:
            term_width = self._console.size.width or 80
            term_height = self._console.size.height or 24

            # ── Build progress panel content ─────────────────────────
            progress_lines: list[Text] = []

            # Header lines (paper title, model info) — before any phase.
            for line in self._header_lines:
                progress_lines.append(Text.from_ansi(line))

            # Completed phase summaries — one compact line each.
            for phase in self._completed_phases:
                icon = _STATUS_ICONS.get(phase.status, "·")
                elapsed = f"({phase.elapsed:.1f}s)"
                if phase.summary:
                    progress_lines.append(
                        Text.from_markup(f" {icon} {phase.name}  [dim]{elapsed}[/dim]  {phase.summary}")
                    )
                else:
                    progress_lines.append(
                        Text.from_markup(f" {icon} {phase.name}  [dim]{elapsed}[/dim]")
                    )

            # Current phase header + scrolling logs.
            if self._current_phase is not None:
                phase = self._current_phase
                elapsed = time.monotonic() - phase.start_time
                icon = _STATUS_ICONS["running"]
                progress_lines.append(Text(""))
                progress_lines.append(
                    Text.from_markup(f" {icon} [bold]{phase.name}[/bold]  [dim]({elapsed:.1f}s)[/dim]")
                )
                progress_lines.append(
                    Text.from_markup(f" [dim]{'─' * min(50, term_width - 6)}[/dim]")
                )
                # Show the last N log lines for the current phase that fit.
                # Reserve space: completed phases + header + phase header (3
                # lines) + borders (2) + stream panel.
                used_lines = (
                    len(self._header_lines)
                    + len(self._completed_phases)
                    + 3  # phase header lines
                )
                # Target stream panel height: ~30% of terminal, min 4.
                stream_panel_height = max(4, int(term_height * 0.30))
                status_height = 1 if self._active_status else 0
                # Borders: 2 for progress, 2 for stream.
                budget = term_height - used_lines - stream_panel_height - status_height - 4
                visible_logs = max(3, budget)
                for log_line in phase.logs[-visible_logs:]:
                    line_text = Text.from_ansi(log_line)
                    progress_lines.append(Text("   ") + line_text)

            if progress_lines:
                progress_text = Text("\n").join(progress_lines)
            else:
                progress_text = Text.from_markup("[dim]Starting pipeline…[/dim]")

            # Dynamic height for progress panel.
            # Rendered line count (with wrapping) is approximated by the raw
            # line count — Rich wraps internally, so we just set a height that
            # fills the remaining vertical budget.
            stream_panel_height = max(4, int(term_height * 0.30))
            status_height = 1 if self._active_status else 0
            progress_height = max(
                6,
                term_height - stream_panel_height - status_height - 2,
            )

            progress_panel = Panel(
                progress_text,
                title="[bold yellow]Pipeline Progress[/bold yellow]",
                border_style="yellow",
                height=progress_height,
            )

            # ── Build status line ────────────────────────────────────
            status_text = ""
            if self._active_status:
                status_text = f" [bold cyan]Status:[/bold cyan] {self._active_status}"

            # ── Build LLM stream panel ───────────────────────────────
            # Show only the most recently active agent's stream, word-wrapped
            # to fill the panel width, auto-scrolling to the latest content.
            inner_width = max(20, term_width - 4)  # panel border padding
            stream_content_lines: list[Text] = []

            # Pick the most recently updated agent (last in dict order).
            active_agents = list(self._active_streams.keys())
            if active_agents:
                agent_name = active_agents[-1]
                parts = self._active_streams[agent_name]
                reasoning = parts.get("reasoning", "")
                content = parts.get("content", "")

                # Agent label header.
                stream_content_lines.append(
                    Text.from_markup(f"[bold green]▶ {agent_name}[/bold green]")
                )

                # Reasoning trace (dimmed, last portion, word-wrapped).
                if reasoning:
                    # Show last ~600 chars of reasoning, word-wrapped.
                    tail = reasoning[-600:]
                    if len(reasoning) > 600:
                        tail = "…" + tail
                    wrapped = textwrap.fill(
                        tail, width=inner_width - 2, break_on_hyphens=False
                    )
                    for wline in wrapped.splitlines():
                        stream_content_lines.append(
                            Text.from_markup(f"[dim italic]💭 {wline}[/dim italic]")
                        )

                # Content stream (normal style, word-wrapped, auto-scrolling).
                if content:
                    # Show the tail of content that fits in the panel.
                    max_content_chars = inner_width * (stream_panel_height + 5)
                    tail = content[-max_content_chars:]
                    if len(content) > max_content_chars:
                        tail = "…" + tail
                    wrapped = textwrap.fill(
                        tail, width=inner_width - 2, break_on_hyphens=False
                    )
                    wrapped_lines = wrapped.splitlines()
                    for wline in wrapped_lines:
                        stream_content_lines.append(Text(wline))
                elif not reasoning:
                    stream_content_lines.append(
                        Text.from_markup("[dim](waiting for output…)[/dim]")
                    )
            else:
                stream_content_lines = [
                    Text.from_markup("[dim]No active LLM streams…[/dim]")
                ]

            # Trim to fit panel height (show latest lines).
            max_visible = max(2, stream_panel_height - 2)  # -2 for border
            if len(stream_content_lines) > max_visible:
                stream_content_lines = stream_content_lines[-max_visible:]

            streams_panel = Panel(
                Group(*stream_content_lines),
                title="[bold green]💬 LLM Stream[/bold green]",
                border_style="green",
                height=stream_panel_height,
            )

            if status_text:
                return Group(
                    progress_panel,
                    Text.from_markup(status_text),
                    streams_panel,
                )
            return Group(progress_panel, streams_panel)

    # ------------------------------------------------------------------
    # Output capture
    # ------------------------------------------------------------------

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Capture printed output and route to the current phase's log."""
        # Use a dummy console for formatting so we don't accidentally capture
        # the Live layout if the background refresh thread runs at the same time.
        dummy = Console(width=self._console.size.width)
        with dummy.capture() as capture:
            dummy.print(*args, **kwargs)
        formatted_str = capture.get().rstrip("\n")

        with self._lock:
            target = (
                self._current_phase.logs
                if self._current_phase is not None
                else self._header_lines
            )
            for line in formatted_str.split("\n"):
                if line.strip() or not target or target[-1].strip():
                    target.append(line)

        if self._live:
            self._live.update(self._render_layout())
        else:
            self._console.print(*args, **kwargs)

    @contextmanager
    def suspend_live(self) -> Generator[None, None, None]:
        """Temporarily stop Live to allow clean terminal prompts/input."""
        live_instance = None
        with self._lock:
            if self._live is not None:
                live_instance = self._live
                self._live.stop()
                self._live = None
        try:
            yield
        finally:
            if live_instance is not None:
                with self._lock:
                    self._live = Live(
                        self._render_layout(),
                        console=self._console,
                        refresh_per_second=10,
                        transient=False,
                        auto_refresh=True,
                    )
                    self._live.start()

    def input(self, prompt: str = "") -> str:
        """Wrapper around console.input to suspend Live layout cleanly."""
        with self.suspend_live():
            return self._console.input(prompt)

    @contextmanager
    def status(
        self, status_msg: str, *args: Any, **kwargs: Any
    ) -> Generator[None, None, None]:
        """Wrapper context manager to track active status."""
        with self._lock:
            self._active_status = status_msg
        if self._live:
            self._live.update(self._render_layout())
        try:
            yield
        finally:
            with self._lock:
                self._active_status = None
            if self._live:
                self._live.update(self._render_layout())

    # ------------------------------------------------------------------
    # LLM stream tracking
    # ------------------------------------------------------------------

    def update_stream(self, agent_name: str, delta: Any) -> None:
        """Update the stream content for an agent.

        ``delta`` may be:
        - ``None`` — clear the agent's stream (end of generation);
        - a :class:`~ppagent.llm.StreamDelta` carrying ``kind`` of ``"content"``
          or ``"reasoning"`` — routed into the matching channel;
        - a bare ``str`` — treated as content for backward compatibility.
        """
        from ppagent.llm import StreamDelta

        with self._lock:
            if delta is None:
                # Clear but keep the agent in the dictionary so the UI shows
                # it as active (e.g. "(waiting for output...)") instead of hiding it.
                self._active_streams[agent_name] = {}
                # Move to the end of the dict to make it the most recently active
                self._active_streams[agent_name] = self._active_streams.pop(agent_name)
            else:
                if isinstance(delta, StreamDelta):
                    kind = delta.kind
                    text = delta.text
                else:
                    kind, text = "content", str(delta)
                parts = self._active_streams.setdefault(agent_name, {})
                parts[kind] = parts.get(kind, "") + text
        if self._live:
            self._live.update(self._render_layout())

    def clear_stream(self, agent_name: str) -> None:
        """Clear the stream content for an agent."""
        self.update_stream(agent_name, None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._console, name)
