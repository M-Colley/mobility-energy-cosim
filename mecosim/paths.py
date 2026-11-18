"""Path helpers."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["absolute"]

_EXTENDED = "\\\\?\\"
_EXTENDED_UNC = _EXTENDED + "UNC\\"


def absolute(path: str | os.PathLike[str]) -> Path:
    r"""Absolute path without the Windows extended-length prefix ``\\?\``.

    ``Path.resolve()`` can return ``\\?\C:\...`` on Windows; SUMO's command-line tools then read an output path as
    ``host:port`` (seen with ``netgenerate -o``), so mecosim passes plain absolute paths to external tools.
    """
    text = os.path.abspath(os.fspath(path))
    if text.startswith(_EXTENDED_UNC):
        text = "\\\\" + text[len(_EXTENDED_UNC) :]
    elif text.startswith(_EXTENDED):
        text = text[len(_EXTENDED) :]
    return Path(text)
