import io

import pytest

from graft import console
from graft.assets.validate import Finding, ValidationReport
from graft.env import Check


class Tty(io.StringIO):
    def isatty(self):
        return True


class Pipe(io.StringIO):
    def isatty(self):
        return False


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)


def test_colour_on_a_terminal():
    assert console.colour_enabled(Tty())


def test_no_colour_when_piped():
    """Escape codes in a redirected log or CI transcript are noise."""
    assert not console.colour_enabled(Pipe())


def test_no_color_env_wins_over_a_terminal(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert not console.colour_enabled(Tty())


def test_no_color_honoured_when_empty(monkeypatch):
    """The NO_COLOR convention is presence, not value."""
    monkeypatch.setenv("NO_COLOR", "")
    assert not console.colour_enabled(Tty())


def test_force_color_overrides_a_pipe(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert console.colour_enabled(Pipe())


def test_dumb_terminal_gets_no_colour(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    assert not console.colour_enabled(Tty())


def test_paint_wraps_and_resets():
    painted = console.paint("hi", "red", stream=Tty())
    assert painted.startswith("\033[")
    assert painted.endswith(console.RESET)
    assert "hi" in painted


def test_paint_is_a_noop_without_colour():
    assert console.paint("hi", "red", stream=Pipe()) == "hi"


def test_unknown_style_is_ignored():
    assert console.paint("hi", "chartreuse", stream=Tty()) == "hi"


def test_status_carries_a_word_not_only_a_colour():
    """Meaning has to survive a colourless terminal."""
    assert "ok" in console.status(True, "ok")
    assert "FAIL" in console.status(False)


def test_check_render_is_plain_when_piped(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    rendered = Check("gpu", True, "RTX 4080").render()
    assert "\033[" not in rendered
    assert "gpu" in rendered and "RTX 4080" in rendered


def test_validation_report_is_plain_when_piped(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    report = ValidationReport(usd_path="a.usd", target_prim_path="/World")
    report.findings.append(Finding("error", "no-geometry", "no mesh"))
    rendered = report.render()
    assert "\033[" not in rendered
    assert "FAIL" in rendered and "no-geometry" in rendered


def test_validation_report_colours_errors(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    report = ValidationReport(usd_path="a.usd", target_prim_path="/World")
    report.findings.append(Finding("error", "no-geometry", "no mesh"))
    assert "\033[" in report.render()
