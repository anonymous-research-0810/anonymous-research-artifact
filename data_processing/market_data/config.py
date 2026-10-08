"""Config for the paper option-hedging pipeline."""

from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING
from dotenv import load_dotenv

if TYPE_CHECKING:
    import wrds


def load_wrds_credentials(dotenv_path: str | Path | None = None) -> tuple[str | None, str | None]:
    """Read optional WRDS credentials from the environment or .env without overriding existing environment values."""
    load_dotenv(dotenv_path=dotenv_path, override=False)
    username = os.environ.get("WRDS_USERNAME") or None
    password = os.environ.get("WRDS_PASSWORD") or None
    return (username, password)


def wrds_connect(dotenv_path: str | Path | None = None) -> "wrds.Connection":
    """Connect through the optional WRDS client. The caller is responsible for closing the connection."""
    try:
        import wrds
    except ImportError:
        raise ImportError(
            "WRDS support is optional; install this project with the 'wrds' extra"
        ) from None
    username, password = load_wrds_credentials(dotenv_path)
    kwargs: dict[str, str] = {}
    if username:
        kwargs["wrds_username"] = username
    if password:
        kwargs["wrds_password"] = password
    return wrds.Connection(**kwargs)
