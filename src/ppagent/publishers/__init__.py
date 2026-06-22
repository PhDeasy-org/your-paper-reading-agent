"""Publisher registry for ppagent."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ppagent.publishers.base import PublisherBase

logger = logging.getLogger(__name__)

_PUBLISHER_REGISTRY: dict[str, type[PublisherBase]] = {}


def register_publisher(cls: type[PublisherBase]) -> type[PublisherBase]:
    """Decorator to register a publisher class."""
    _PUBLISHER_REGISTRY[cls.name] = cls
    return cls


def get_publisher(name: str, **kwargs) -> PublisherBase:
    """Instantiate a registered publisher by name."""
    if name not in _PUBLISHER_REGISTRY:
        raise KeyError(
            f"Publisher '{name}' not registered. Available: {list(_PUBLISHER_REGISTRY.keys())}"
        )
    return _PUBLISHER_REGISTRY[name](**kwargs)


def list_publishers() -> list[str]:
    return list(_PUBLISHER_REGISTRY.keys())


def load_custom_publishers(directory: Path | None = None) -> None:
    """Dynamically import custom publisher modules from a directory."""
    if directory is None:
        directory = Path.home() / ".config" / "ppagent" / "publishers"
    if not directory.is_dir():
        return
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"ppagent_custom_publishers_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                logger.info("Loaded custom publisher module: %s", py_file.name)
        except Exception:
            logger.exception("Failed to load custom publisher: %s", py_file)


# Import built-in publishers so their @register_publisher decorators run
from ppagent.publishers.wechat import WeChatPublisher  # noqa: F401, E402
from ppagent.publishers.notion import NotionPublisher  # noqa: F401, E402
from ppagent.publishers.github_pages import GithubPagesPublisher  # noqa: F401, E402
