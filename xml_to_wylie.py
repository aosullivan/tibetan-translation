#!/usr/bin/env python3
"""
Convert ACIP-style Tibetan XML (Unicode in <page> with <line/> breaks) to EWTS/Wylie.

Uses pyewts.toWylie(), matching the same stack as src/wylie_converter.py (reverse direction).

Example:
  python xml_to_wylie.py -i "/path/to/bstan-'gyur_toh2488(rgyud.zi148).xml" -o out.wylie.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


def _local_tag(elem: ET.Element) -> str:
    if not elem.tag or elem.tag.startswith("{"):
        return elem.tag.split("}", 1)[-1]
    return elem.tag


def element_to_plain_text(elem: ET.Element, line_tag: str = "line") -> str:
    """Serialize element mixed content; each <line/> becomes a newline."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        tag = _local_tag(child)
        if tag == line_tag:
            parts.append("\n")
        else:
            parts.append(element_to_plain_text(child, line_tag))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _text_or_empty(info: ET.Element, tag: str) -> str:
    el = info.find(tag)
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def extract_info_header(root: ET.Element, converter) -> str:
    """Optional comment block from <info> (Tibetan fields → Wylie)."""
    info = root.find("info")
    if info is None:
        return ""
    lines: list[str] = ["# --- metadata (from <info>) ---"]
    xref = info.find("xref")
    if xref is not None and xref.get("tohoku"):
        lines.append(f"# Tohoku: {xref.get('tohoku')}")
    loc = info.find("location")
    if loc is not None:
        parts = [
            f"{k}={loc.get(k)}"
            for k in ("organization", "collection", "volume", "index")
            if loc.get(k)
        ]
        if parts:
            lines.append("# " + " ".join(parts))
    for label, tag in (
        ("title", "title"),
        ("author", "author"),
        ("section", "section"),
        ("volume", "volume"),
    ):
        raw = _text_or_empty(info, tag)
        if not raw:
            continue
        try:
            wylie = converter.toWylie(raw)
        except Exception as e:
            logger.warning("toWylie failed for %s: %s", tag, e)
            wylie = raw
        lines.append(f"# {label}: {wylie}")
    st = info.find("status")
    if st is not None and (st.text or "").strip():
        lines.append(f"# status: {(st.text or '').strip()}")
    lines.append("# --- body ---")
    return "\n".join(lines) + "\n"


def convert_xml_to_wylie(
    xml_path: Path,
    *,
    include_metadata: bool = True,
    page_headers: bool = True,
) -> str:
    from pyewts import pyewts

    tree = ET.parse(xml_path)
    root = tree.getroot()
    conv = pyewts()

    out_parts: list[str] = []
    if include_metadata:
        header = extract_info_header(root, conv)
        if header:
            out_parts.append(header)

    pageset = root.find("pageset")
    if pageset is None:
        raise ValueError("No <pageset> in XML root")

    pages = pageset.findall("page")
    for page in pages:
        idx = page.get("index", "")
        pid = page.get("id", "")
        body = element_to_plain_text(page)
        try:
            wylie = conv.toWylie(body)
        except Exception as e:
            logger.error("toWylie failed for page index=%s id=%s: %s", idx, pid, e)
            raise

        if page_headers:
            out_parts.append(f"\n### page index={idx} id={pid}\n")
        else:
            out_parts.append("\n")
        out_parts.append(wylie.rstrip())
        out_parts.append("\n")

    return "".join(out_parts).lstrip("\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert ACIP-style Tibetan Unicode XML to Wylie (EWTS) via pyewts."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to input .xml",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .txt path (default: input stem + .wylie.txt)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Omit converted <info> header comments",
    )
    parser.add_argument(
        "--no-page-headers",
        action="store_true",
        help="Do not print ### page index=... lines between pages",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug messages",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    xml_path = args.input.expanduser().resolve()
    if not xml_path.is_file():
        logger.error("Input not found: %s", xml_path)
        return 1

    out_path = args.output
    if out_path is None:
        out_path = xml_path.with_suffix(".wylie.txt")
    else:
        out_path = out_path.expanduser().resolve()

    try:
        text = convert_xml_to_wylie(
            xml_path,
            include_metadata=not args.no_metadata,
            page_headers=not args.no_page_headers,
        )
    except Exception as e:
        logger.error("%s", e)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", out_path, len(text.encode("utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
