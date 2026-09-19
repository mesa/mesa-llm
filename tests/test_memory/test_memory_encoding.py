"""Tests verifying MemoryEntry.__str__() output is encodable on all platforms.

Regression test for https://github.com/mesa/mesa-llm/issues/321 -- the
box-drawing characters previously used (U+2514, U+2500, U+251C) crash
with UnicodeEncodeError on Windows cp125x consoles.
"""

from typing import ClassVar

import pytest

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
