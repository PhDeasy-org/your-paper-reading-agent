"""Agent registry for ppagent."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ppagent.agents.base import AgentBase

logger = logging.getLogger(__name__)

_AGENT_REGISTRY: dict[str, type[AgentBase]] = {}


def register_agent(cls: type[AgentBase]) -> type[AgentBase]:
    """Decorator to register an agent class by its name."""
    _AGENT_REGISTRY[cls.name] = cls
    return cls


def get_agent(name: str, **kwargs) -> AgentBase:
    """Instantiate a registered agent by name."""
    if name not in _AGENT_REGISTRY:
        raise KeyError(
            f"Agent '{name}' not registered. Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return _AGENT_REGISTRY[name](**kwargs)


def list_agents() -> list[str]:
    """Return names of all registered agents."""
    return list(_AGENT_REGISTRY.keys())


def load_custom_agents(directory: Path | None = None) -> None:
    """Dynamically import custom agent modules from a directory.

    Default location: ~/.config/ppagent/agents/
    """
    if directory is None:
        directory = Path.home() / ".config" / "ppagent" / "agents"
    if not directory.is_dir():
        return
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"ppagent_custom_agents_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                logger.info("Loaded custom agent module: %s", py_file.name)
        except Exception:
            logger.exception("Failed to load custom agent: %s", py_file)


# Import built-in agents so their @register_agent decorators run
from ppagent.agents.searcher import SearcherAgent  # noqa: F401, E402
from ppagent.agents.writer import WriterAgent  # noqa: F401, E402
from ppagent.agents.finder import FinderAgent  # noqa: F401, E402
from ppagent.agents.criticizer import CriticizerAgent  # noqa: F401, E402
from ppagent.agents.figure_selector import FigureSelectorAgent  # noqa: F401, E402
