"""Terminal colour.

Colour is decided per call rather than at import so that redirecting output
mid-process, or a test toggling the environment, behaves correctly.

Honours NO_COLOR (https://no-color.org) and FORCE_COLOR, and stays silent
when output is not a terminal — escape codes in a piped log or a CI
transcript are noise.
"""

import os
import sys

RESET = "\033[0m"
CODES = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
    "bold": "\033[1m",
}


def colour_enabled(stream=None) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, *styles: str, stream=None) -> str:
    if not styles or not colour_enabled(stream):
        return text
    prefix = "".join(CODES[s] for s in styles if s in CODES)
    return f"{prefix}{text}{RESET}" if prefix else text


def ok(text: str) -> str:
    return paint(text, "green")


def fail(text: str) -> str:
    return paint(text, "red", "bold")


def warn(text: str) -> str:
    return paint(text, "yellow")


def info(text: str) -> str:
    return paint(text, "cyan")


def dim(text: str) -> str:
    return paint(text, "grey")


def bold(text: str) -> str:
    return paint(text, "bold")


def heading(text: str) -> str:
    return paint(text, "bold", "blue")


def status(passed: bool, label: str | None = None) -> str:
    """A pass/fail tag. Includes a word as well as a colour so the meaning
    survives a colourless terminal."""
    if label is None:
        label = "ok" if passed else "FAIL"
    return ok(label) if passed else fail(label)


def value(text: str) -> str:
    return paint(text, "bold")
