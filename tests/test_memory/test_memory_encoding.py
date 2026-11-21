"""Tests verifying MemoryEntry.__str__() output is encodable on all platforms.

Regression test for https://github.com/mesa/mesa-llm/issues/321 -- the
box-drawing characters previously used (U+2514, U+2500, U+251C) crash
with UnicodeEncodeError on Windows cp125x consoles.
"""

import io
from typing import ClassVar

import pytest
from rich.console import Console

from mesa_llm.memory.memory import MemoryEntry

# Code pages covering all major Windows locale families.
# Box-drawing chars (U+2514/2500/251C) crash on cp125x; bullet (U+2022)
# crashes on cp437/850/932/936/949. ASCII tree markers are safe on all.
WINDOWS_CODEPAGES: list[str] = [
    "cp1252",  # Western Europe
    "cp1250",  # Central Europe
    "cp1251",  # Cyrillic
    "cp1253",  # Greek
    "cp1254",  # Turkish
    "cp1255",  # Hebrew
    "cp1256",  # Arabic
    "cp874",  # Thai
    "cp437",  # DOS US
    "cp850",  # DOS Western Europe
    "cp932",  # Japanese
    "cp936",  # Chinese Simplified
    "cp949",  # Korean
    "utf-8",  # Universal
]


@pytest.fixture
def sample_entry(mock_agent):
    """Build a MemoryEntry with nested content covering all code paths in __str__."""
    return MemoryEntry(
        content={
            "observation": "price is $20",
            "action": [{"offer": 25, "targets": ["seller1", "seller2"]}],
            "message": [{"message": "I offer $25", "sender": 1}],
            "plan": "negotiate aggressively",
        },
        step=1,
        agent=mock_agent,
    )


class TestMemoryEntryEncoding:
    """Ensure MemoryEntry.__str__() output never crashes on any encoding."""

    CODEPAGES: ClassVar[list[str]] = WINDOWS_CODEPAGES

    @pytest.mark.parametrize("encoding", CODEPAGES)
    def test_str_encodable_all_windows_codepages(self, sample_entry, encoding):
        """MemoryEntry.__str__() output must encode without error on every
        major Windows code page and UTF-8."""
        output = str(sample_entry)
        output.encode(encoding)  # raises UnicodeEncodeError if any char is unsupported

    def test_str_no_box_drawing_characters(self, sample_entry):
        """No Unicode box-drawing characters (U+2500-U+257F) in output."""
        output = str(sample_entry)
        for cp in range(0x2500, 0x2580):
            assert chr(cp) not in output, (
                f"Found box-drawing character U+{cp:04X} in MemoryEntry output"
            )

    def test_str_preserves_tree_structure(self, sample_entry):
        """Output should still contain ASCII tree markers for hierarchy."""
        output = str(sample_entry)
        assert "+--" in output, "Expected ASCII tree marker '+--' in output"
        assert "|--" in output, "Expected ASCII tree marker '|--' in output"

    @pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
    def test_display_preserves_content_on_encoded_stream(
        self, sample_entry, monkeypatch, encoding
    ):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(
            raw, encoding=encoding, errors="strict", write_through=True
        )
        monkeypatch.setattr("sys.stdout", stream)
        sample_entry.agent.memory.display = True
        sample_entry.content = {"observation": "\u00e9 \u2615 \u65e5\u672c"}

        sample_entry.display()

        output = raw.getvalue().decode(encoding)
        assert "\u00e9" in output
        if encoding == "utf-8":
            assert "\u2615 \u65e5\u672c" in output
        else:
            assert r"\u2615 \u65e5\u672c" in output
        assert sample_entry.content["observation"] == "\u00e9 \u2615 \u65e5\u672c"
        assert stream.errors == "strict"

    @pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
    def test_display_rich_emoji_alias(self, sample_entry, monkeypatch, encoding):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(
            raw, encoding=encoding, errors="strict", write_through=True
        )
        monkeypatch.setattr("sys.stdout", stream)
        sample_entry.agent.memory.display = True
        sample_entry.content = {"observation": ":coffee:"}

        sample_entry.display()

        output = raw.getvalue().decode(encoding)
        assert (r"\u2615" if encoding == "cp1252" else "\u2615") in output
        assert stream.errors == "strict"

    def test_display_with_stream_without_encoding(self, sample_entry, monkeypatch):
        stream = io.StringIO()
        monkeypatch.setattr("sys.stdout", stream)
        sample_entry.agent.memory.display = True
        sample_entry.content = {"observation": "\u65e5\u672c"}

        sample_entry.display()

        assert "\u65e5\u672c" in stream.getvalue()

    def test_disabled_display_writes_nothing(self, sample_entry, monkeypatch):
        stream = io.StringIO()
        monkeypatch.setattr("sys.stdout", stream)
        sample_entry.agent.memory.display = False

        sample_entry.display()

        assert stream.getvalue() == ""

    def test_display_with_minimal_stdout(self, sample_entry, monkeypatch):
        class Capture:
            def __init__(self):
                self.value = ""

            def write(self, text):
                self.value += text
                return len(text)

            def flush(self):
                pass

        stream = Capture()
        monkeypatch.setattr("sys.stdout", stream)
        sample_entry.agent.memory.display = True

        sample_entry.display()

        assert "price is $20" in stream.value

    def test_escaped_content_fits_panel(self, sample_entry, monkeypatch):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", write_through=True)
        monkeypatch.setattr("sys.stdout", stream)
        sample_entry.agent.memory.display = True
        sample_entry.content = {"observation": "\u65e5\u672c" * 20}

        sample_entry.display()

        lines = raw.getvalue().decode("cp1252").splitlines()
        assert len({len(line) for line in lines}) == 1

    @pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
    def test_display_with_unicode_hyperlink(self, sample_entry, monkeypatch, encoding):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding=encoding, write_through=True)
        console = Console(file=stream, force_terminal=True, legacy_windows=False)
        monkeypatch.setattr("mesa_llm.memory.memory.Console", lambda: console)
        sample_entry.agent.memory.display = True
        sample_entry.content = {
            "observation": "[link=https://example.com/\u65e5\u672c]hello[/link]"
        }

        sample_entry.display()

        output = raw.getvalue().decode(encoding)
        assert "hello" in output
        if encoding == "utf-8":
            assert "https://example.com/\u65e5\u672c" in output
        else:
            assert "https://example.com/" not in output
