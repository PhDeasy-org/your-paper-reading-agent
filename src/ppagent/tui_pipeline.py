from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class PipelineConsoleWrapper:
    """Wraps rich Console to capture printed logs and render a split Live layout."""

    def __init__(self, original_console: Console) -> None:
        self._console = original_console
        self._logs: list[str] = []
        self._active_status: str | None = None
        # Per-agent streamed text, split into answer content and reasoning
        # trace. Reasoning models (GLM/Grok/…) emit a long thinking trace on a
        # separate field before the answer; keeping them separate lets us render
        # reasoning dimmed so the user sees live activity without it looking like
        # the final output.
        self._active_streams: dict[str, dict[str, str]] = {}
        # RLock (not Lock): several methods call _render_layout() while already
        # holding _lock (start_live, status, update_stream, suspend_live), and
        # _render_layout re-acquires it. A plain Lock self-deadlocks there,
        # freezing the whole pipeline so nothing ever reaches the terminal.
        self._lock = threading.RLock()
        self._live: Live | None = None

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

    def _render_layout(self) -> Group:
        with self._lock:
            # The Live UI must fit inside the terminal viewport. Panels have
            # fixed heights by default (progress=17, streams=8) plus a status
            # line, which together exceed ~27 rows; on shorter terminals Live
            # redraws them on top of each other, producing garbled overlapping
            # output. So size the progress panel to whatever vertical budget
            # remains after the streams panel + status line.
            term_height = self._console.size.height or 24
            streams_panel_height = 8
            status_height = 1 if self._active_status else 0
            # 2 border rows for the progress panel itself.
            progress_height = max(
                6,
                term_height - streams_panel_height - status_height - 2,
            )

            # Log history panel - parse ANSI escapes so colorization is preserved
            log_text = Text.from_ansi("\n".join(self._logs[-15:]))
            progress_panel = Panel(
                log_text,
                title="[bold yellow]Pipeline Progress[/bold yellow]",
                border_style="yellow",
                height=progress_height,
            )

            # Active status line
            status_text = ""
            if self._active_status:
                status_text = f" [bold cyan]Status:[/bold cyan] {self._active_status}"

            # Streams panel
            streams_group: list[Text] = []
            for agent, parts in sorted(self._active_streams.items()):
                content = parts.get("content", "")
                reasoning = parts.get("reasoning", "")
                # Show only the last characters of each channel to keep it clean.
                # Reasoning gets a smaller window (it's the thinking trace) and
                # is rendered dimmed so it reads as background activity.
                pieces: list[tuple[str, str]] = [(f"▶ {agent}: ", "bold green")]
                if reasoning:
                    pieces.append(("💭 ", "dim"))
                    pieces.append((reasoning[-200:], "dim italic"))
                if content:
                    label = "✍ " if reasoning else ""
                    pieces.append((label, "dim"))
                    pieces.append((content[-300:], "default"))
                if len(pieces) == 1:
                    # No content yet — show a faint placeholder so the panel
                    # isn't blank between agent start and first token.
                    pieces.append(("(waiting for output…)", "dim"))
                streams_group.append(Text.assemble(*pieces))

            if not streams_group:
                streams_group = [
                    Text.from_markup("[dim]No active LLM streams...[/dim]")
                ]

            streams_panel = Panel(
                Group(*streams_group),
                title="[bold green]Live LLM Streams[/bold green]",
                border_style="green",
                height=streams_panel_height,
            )

            if status_text:
                return Group(
                    progress_panel, Text.from_markup(status_text), streams_panel
                )
            return Group(progress_panel, streams_panel)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Capture printed output and append to log history."""
        with self._console.capture() as capture:
            self._console.print(*args, **kwargs)
        formatted_str = capture.get().rstrip("\n")

        with self._lock:
            for line in formatted_str.split("\n"):
                if line.strip() or not self._logs or self._logs[-1].strip():
                    self._logs.append(line)

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
                self._active_streams.pop(agent_name, None)
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
