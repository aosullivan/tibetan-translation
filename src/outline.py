"""Outline (sa bcad) parser for structure-aware translation.

Accepts markdown files containing a mix of `#`-headings and `**bold**`
entries (optionally indented, inside fenced code blocks, with trailing
`← STATUS` annotations). Produces a flat list of `OutlineEntry` items
that the translation prompt can reference by index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutlineEntry:
    index: int
    rendered: str  # literal markdown to emit, e.g. "## First Point" or "**1. General meaning**"
    title: str     # title text only (for matching / logs)

    def level_hint(self) -> int:
        m = re.match(r"^(#+)\s", self.rendered)
        return len(m.group(1)) if m else 7


_ANNOTATION_RE = re.compile(r"\s*←.*$")
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_HEADING_RE = re.compile(r"^\s*(#+)\s+(.+?)\s*$")
_BOLD_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")


def parse_outline(text: str) -> list[OutlineEntry]:
    entries: list[OutlineEntry] = []
    in_code_fence = False

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            continue

        clean = _ANNOTATION_RE.sub("", line).rstrip()
        # strip trailing "(six points)" style annotations so bold entries still match
        clean = _TRAILING_PAREN_RE.sub("", clean).rstrip()

        m = _HEADING_RE.match(clean)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            entries.append(OutlineEntry(
                index=len(entries),
                rendered="#" * level + " " + title,
                title=title,
            ))
            continue

        m = _BOLD_RE.match(clean)
        if m:
            title = m.group(1).strip()
            entries.append(OutlineEntry(
                index=len(entries),
                rendered=f"**{title}**",
                title=title,
            ))

    return entries


def load_outline(path: Path) -> list[OutlineEntry]:
    return parse_outline(path.read_text(encoding="utf-8"))


def render_window(
    entries: list[OutlineEntry],
    cursor: int,
    before: int = 3,
    after: int = 20,
) -> str:
    """Render a sliding window of outline entries around the cursor.

    Produces a numbered table pointing at the current cursor position,
    suitable for inclusion in a prompt.
    """
    if not entries:
        return ""

    header = ""
    if cursor < 0:
        header = "(no headings emitted yet — start looking from index 0)\n"
    elif cursor >= len(entries):
        header = "(all outline entries already emitted)\n"

    lo = max(0, cursor - before)
    hi = min(len(entries), max(cursor, 0) + after + 1)
    lines: list[str] = []
    for e in entries[lo:hi]:
        marker = " →" if e.index == cursor else "  "
        lines.append(f"{marker} {e.index:>4} | {e.rendered}")
    return header + "\n".join(lines)
