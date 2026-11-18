"""Path helper: no Windows extended-length prefix in paths handed to SUMO tools."""

from __future__ import annotations

import os
import sys

import pytest

from mecosim.paths import absolute

EXT = "\\\\?\\"


def test_relative_paths_become_absolute() -> None:
    assert absolute("x").is_absolute()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path syntax")
def test_extended_length_prefix_is_removed() -> None:
    assert str(absolute(EXT + "C:\\data\\net.xml")) == "C:\\data\\net.xml"
    assert str(absolute(EXT + "UNC\\server\\share\\net.xml")) == "\\\\server\\share\\net.xml"
    assert not str(absolute(os.getcwd())).startswith(EXT)
