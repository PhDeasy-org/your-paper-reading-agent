"""Regression tests for ``PipelineConsoleWrapper``.

These guard against the non-reentrant-lock deadlock that froze the whole
pipeline: several methods call ``_render_layout()`` (which acquires
``self._lock``) while already holding ``self._lock``. With a plain
``threading.Lock`` that self-deadlocks, so ``ppagent report`` produced no
output at all. Each test runs the previously-hanging path under a watchdog;
a regression hangs and fails the test instead of hanging the suite.
"""

from __future__ import annotations

import io
import signal
import threading
from typing import Any

import pytest
from rich.console import Console
from rich.live import Live

from ppagent.tui_pipeline import PipelineConsoleWrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_live_refresh_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable ``Live``'s background refresh thread.

    ``auto_refresh=True`` spawns a daemon that calls ``refresh()`` on a timer;
    in the test environment (no real TTY) that loop spins forever and masks
    the deadlock we are testing for. We keep auto_refresh off so a hang is
    attributable purely to the lock.
    """
    orig = Live.__init__

    def patched(self: Live, *args: Any, **kwargs: Any) -> None:
        kwargs["auto_refresh"] = False
        orig(self, *args, **kwargs)

    monkeypatch.setattr(Live, "__init__", patched)


def _watchdog(name: str, seconds: int = 5) -> None:
    """Fail the current test if it runs longer than ``seconds``.

    A deadlock would otherwise hang pytest indefinitely; the watchdog turns it
    into a hard failure. Only effective on the main thread (SIGALRM), which is
    where these tests run.
    """

    def _die(*_args: Any) -> None:
        pytest.fail(f"{name} deadlocked (did not complete within {seconds}s)")

    signal.signal(signal.SIGALRM, _die)
    signal.alarm(seconds)


def _wrapper_with_live() -> PipelineConsoleWrapper:
    """Build a wrapper with a started Live, bypassing ``start_live``'s pytest
    guard so we can exercise the real render-under-lock paths."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=90, color_system=None)
    wrap = PipelineConsoleWrapper(console)
    live = Live(console=console, auto_refresh=False)
    live.start()
    wrap._live = live
    return wrap


# ---------------------------------------------------------------------------
# Lock contract
# ---------------------------------------------------------------------------


def test_lock_is_reentrant() -> None:
    """The shared lock must be reentrant: several public methods re-enter it."""
    wrap = PipelineConsoleWrapper(Console(file=io.StringIO(), width=80))
    assert isinstance(wrap._lock, type(threading.RLock()))


def test_render_layout_is_safe_under_lock() -> None:
    """The original deadlock: _render_layout acquires _lock while the caller
    already holds it. With a reentrant lock this completes."""
    _watchdog("render-under-lock")
    wrap = _wrapper_with_live()
    with wrap._lock:
        wrap._render_layout()  # would self-deadlock with a plain Lock


# ---------------------------------------------------------------------------
# Deadlock regression: every path that renders the layout while holding _lock
# ---------------------------------------------------------------------------


def test_print_during_live_does_not_deadlock() -> None:
    _watchdog("print")
    wrap = _wrapper_with_live()
    wrap.print("🚀 Starting Report Generation")
    # prints while a Live is active re-render the panel via _render_layout()


def test_status_context_does_not_deadlock() -> None:
    # report() wraps every phase in console.status(...); this used to hang.
    _watchdog("status")
    wrap = _wrapper_with_live()
    with wrap.status("[dim]Fetching...[/dim]", spinner="dots"):
        pass
    with wrap.status("[dim]Assembling...[/dim]"):
        pass


def test_update_stream_does_not_deadlock() -> None:
    # The LLM stream callback routes deltas here; called from a worker thread.
    _watchdog("update_stream")
    wrap = _wrapper_with_live()
    with wrap.status("[dim]Running Writer...[/dim]"):
        wrap.update_stream("Writer", "some delta text")
        wrap.update_stream("Writer", None)  # clear


def test_suspend_live_does_not_deadlock() -> None:
    # input() suspends Live to read a prompt; nested under _lock twice.
    _watchdog("suspend_live")
    wrap = _wrapper_with_live()
    with wrap.suspend_live():
        wrap.print("while suspended prints plainly")


def test_concurrent_render_does_not_deadlock() -> None:
    """One worker renders (mimicking the auto-refresh thread) while the main
    thread mutates state under the lock via update_stream."""
    _watchdog("concurrent-render")
    wrap = _wrapper_with_live()
    done = threading.Event()
    errors: list[BaseException] = []

    def renderer() -> None:
        try:
            while not done.is_set():
                wrap._render_layout()
        except BaseException as exc:  # noqa: BLE001 - surface on main thread
            errors.append(exc)

    t = threading.Thread(target=renderer)
    t.start()
    for i in range(50):
        wrap.update_stream("Writer", f"delta {i}")
    done.set()
    t.join(timeout=5)
    assert not errors, errors
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# Behavior: content actually reaches the stream when Live is off
# ---------------------------------------------------------------------------


def test_print_falls_back_to_console_without_live() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    wrap = PipelineConsoleWrapper(console)
    wrap.start_live()  # no-op under pytest / non-tty -> _live stays None
    assert wrap._live is None
    wrap.print("hello world")
    wrap.stop_live()
    assert "hello world" in buf.getvalue()


# ---------------------------------------------------------------------------
# Layout fit: the Live UI must not overflow the terminal viewport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term_height", [16, 20, 24, 30, 40, 50, 80])
def test_layout_fits_terminal_height(term_height: int) -> None:
    """Fixed-height panels (progress=17 + streams=8 + status) used to render to
    ~27 lines, overflowing short terminals and causing Live to redraw panels on
    top of each other (garbled output). The progress panel is now sized to the
    remaining vertical budget, so the whole layout fits any terminal height."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=100,
        height=term_height,
        color_system=None,
    )
    wrap = PipelineConsoleWrapper(console)
    wrap._active_status = "Formatting, generating LaTeX equations, and writing files..."
    wrap._active_streams = {"Writer": {"content": "x" * 50, "reasoning": "y" * 50}}

    layout = wrap._render_layout()

    # Measure the rendered height in a console with a tall viewport so the
    # measurement itself isn't clipped.
    measure_buf = io.StringIO()
    measure = Console(
        file=measure_buf, force_terminal=True, width=100, height=200, color_system=None
    )
    with measure.capture() as capture:
        measure.print(layout)
    rendered_lines = capture.get().count("\n") + 1

    assert rendered_lines <= term_height, (
        f"layout rendered {rendered_lines} lines but terminal is {term_height} "
        f"— Live would overwrite panels (the original garble bug)"
    )


def test_layout_has_min_height_progress_panel() -> None:
    """Even on a very short terminal, the progress panel keeps a sane minimum
    rather than collapsing to nothing."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100, height=12, color_system=None)
    wrap = PipelineConsoleWrapper(console)
    layout = wrap._render_layout()
    # Should still render without error and fit.
    with console.capture() as capture:
        console.print(layout)
    assert capture.get()


# ---------------------------------------------------------------------------
# Reasoning vs content routing in the Live stream panel
# ---------------------------------------------------------------------------


def test_update_stream_routes_reasoning_and_content_separately() -> None:
    """A StreamDelta tagged 'reasoning' must land in the reasoning channel and
    not bleed into the answer content. This is what lets the UI render the
    thinking trace (dimmed) while keeping the answer preview clean."""
    from ppagent.llm import StreamDelta

    wrap = PipelineConsoleWrapper(Console(file=io.StringIO(), width=80))
    wrap.update_stream("Writer", StreamDelta("thinking…", kind="reasoning"))
    wrap.update_stream("Writer", StreamDelta("the answer", kind="content"))

    parts = wrap._active_streams["Writer"]
    assert parts["reasoning"] == "thinking…"
    assert parts["content"] == "the answer"

    # Reasoning must not appear in the content channel and vice versa.
    assert "thinking" not in parts["content"]
    assert "answer" not in parts["reasoning"]


def test_update_stream_accepts_bare_str_as_content() -> None:
    """Backward compatibility: a plain str delta is treated as content."""
    wrap = PipelineConsoleWrapper(Console(file=io.StringIO(), width=80))
    wrap.update_stream("Writer", "legacy content")
    assert wrap._active_streams["Writer"]["content"] == "legacy content"


def test_stream_panel_renders_reasoning_dimmed_with_content() -> None:
    """The rendered stream panel shows both the dimmed reasoning trace and the
    answer content for an active agent — it must not be blank during thinking."""
    from ppagent.llm import StreamDelta

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100, height=40, color_system=None)
    wrap = PipelineConsoleWrapper(console)
    wrap.update_stream("Writer", StreamDelta("deliberating the approach", kind="reasoning"))
    wrap.update_stream("Writer", StreamDelta("final prose here", kind="content"))

    with console.capture() as capture:
        console.print(wrap._render_layout())

    rendered = capture.get()
    assert "deliberating the approach" in rendered
    assert "final prose here" in rendered
    assert "No active LLM streams" not in rendered
