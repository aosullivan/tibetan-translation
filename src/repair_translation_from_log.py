#!/usr/bin/env python3
"""
Rebuild a chunk translation output using Raw response lines from translation.log,
calling the API only for chunk indices not recoverable from logs.

Walks chunks 0..N-1 in order so previous-chunk context matches translate.py behavior.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from anthropic import Anthropic
from dotenv import load_dotenv

from src.config import TranslationConfig as cfg
from src.file_handler import FileHandler
from src.translation_client import TranslationClient

LOG_LINE_START = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - ")
CHUNK_START = re.compile(
    r"Starting translation for chunk (\d+)/(\d+)"
)


def _flush_raw_block(
    current_chunk: int | None,
    raw_lines: list[str],
    out: dict[int, str],
) -> None:
    if current_chunk is None or not raw_lines:
        return
    raw = "\n".join(raw_lines).strip()
    if not raw:
        return
    parsed = TranslationClient._parse_translation_response(raw)
    translation = parsed[0]
    out[current_chunk] = translation


def parse_translation_log(text: str) -> tuple[dict[int, str], int | None]:
    """
    Parse logging text for 'Starting translation for chunk X/Y' and following
    'Raw response:' multi-line blocks. Returns (chunk_index -> translation, total_chunks).
    Last Raw response for a given chunk index wins.
    """
    by_index: dict[int, str] = {}
    total_chunks: int | None = None

    current_chunk: int | None = None
    in_raw = False
    raw_lines: list[str] = []

    for line in text.splitlines():
        if LOG_LINE_START.match(line):
            if in_raw:
                _flush_raw_block(current_chunk, raw_lines, by_index)
                raw_lines = []
                in_raw = False

            m_chunk = CHUNK_START.search(line)
            if m_chunk:
                one_based = int(m_chunk.group(1))
                total = int(m_chunk.group(2))
                total_chunks = total
                current_chunk = one_based - 1
                continue

            if "Raw response: " in line:
                idx = line.index("Raw response: ") + len("Raw response: ")
                first = line[idx:]
                in_raw = True
                raw_lines = [first] if first else []
            continue

        if in_raw:
            raw_lines.append(line)

    if in_raw:
        _flush_raw_block(current_chunk, raw_lines, by_index)

    return by_index, total_chunks


def load_merged_logs(paths: list[Path]) -> str:
    parts: list[str] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Log not found: {p}")
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def rebuild(
    input_file: Path,
    output_file: Path,
    from_log: dict[int, str],
    client: TranslationClient | None,
) -> tuple[list[int], int]:
    """
    Returns (missing_indices_that_were_filled_via_api, total_parts).
    If there are no missing indices, client may be None (log-only rebuild).
    """
    fh = FileHandler()
    content = fh.read_input_file(str(input_file))
    parts = fh.chunk_text(content)
    n = len(parts)

    missing = [i for i in range(n) if i not in from_log]
    if missing and client is None:
        raise ValueError("API client required when some chunks are missing from the log.")

    use_summary_api = bool(missing)

    previous_translation = ""
    unsummarized: list[str] = []
    current_summary = ""
    built: list[str] = []

    for i, part in enumerate(parts):
        if i in from_log:
            translation_text = from_log[i]
        else:
            assert client is not None
            effective_summary = current_summary
            if unsummarized:
                effective_summary += "\n" + " ".join(unsummarized)
            result = client.translate_chunk(part, previous_translation, effective_summary)
            translation_text = result[0] if isinstance(result, tuple) else result

        built.append(translation_text)
        previous_translation = translation_text
        unsummarized.append(translation_text)
        if len(unsummarized) >= cfg.SUMMARY_INTERVAL:
            combined_text = current_summary + "\n" + " ".join(unsummarized)
            if use_summary_api:
                assert client is not None
                current_summary = client.generate_summary(combined_text)
            else:
                current_summary = combined_text.strip()
            unsummarized = []

    header = (
        "# Tibetan Text Translation\n\n"
        if cfg.OUTPUT_FORMAT == "markdown"
        else ""
    )
    body = "\n\n".join(built) + "\n\n"

    cfg.OUTPUT_FILE = str(output_file)
    cfg.PROGRESS_FILE = str(output_file.with_suffix(".progress.json"))
    fh.progress = {
        "_meta": {"input": os.path.abspath(str(input_file))},
        **{str(i): True for i in range(n)},
    }
    fh.current_output = header + body
    fh.save_progress()
    return missing, n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair translation output from translation.log; API only for gaps."
    )
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument(
        "--log",
        "-l",
        action="append",
        type=Path,
        default=[],
        help="Log file (repeatable). Merged in order; later entries override chunk indices.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing chunk indices and exit without calling the API or writing files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON path for missing-index report (dry-run or apply).",
    )
    args = parser.parse_args()

    log_paths = args.log
    if not log_paths:
        default_log = project_root / "translation.log"
        if default_log.is_file():
            log_paths = [default_log]
        else:
            print(
                "No --log files given and translation.log not found at repo root.\n"
                "Pass one or more logs, e.g. copies of translation.log or terminal captures.",
                file=sys.stderr,
            )
            return 1

    merged = load_merged_logs(log_paths)
    from_log, log_total = parse_translation_log(merged)

    fh = FileHandler()
    content = fh.read_input_file(str(args.input))
    parts = fh.chunk_text(content)
    n = len(parts)

    if log_total is not None and log_total != n:
        logging.warning(
            "Log reports %s chunks but input splits into %s; using input chunk count.",
            log_total,
            n,
        )

    missing = [i for i in range(n) if i not in from_log]
    print(f"Total chunks (from input): {n}")
    print(f"Chunks recovered from log: {len(from_log)}")
    print(f"Chunks missing from log (need API if applying): {len(missing)}")
    if missing:
        print(f"First missing chunk index (0-based): {min(missing)}")
        preview = missing[:40]
        more = "" if len(missing) <= 40 else f" ... (+{len(missing) - 40} more)"
        print(f"Missing indices (0-based): {preview}{more}")

    report_obj = {
        "total_chunks": n,
        "from_log_indices": sorted(from_log.keys()),
        "from_log_count": len(from_log),
        "missing_indices": missing,
        "log_paths": [str(p) for p in log_paths],
    }
    if args.report is not None:
        args.report.write_text(json.dumps(report_obj, indent=2), encoding="utf-8")
        print(f"Wrote report {args.report}")

    if args.dry_run:
        return 0

    client: TranslationClient | None = None
    if missing:
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ANTHROPIC_API_KEY not set; cannot fill gaps.", file=sys.stderr)
            return 1
        client = TranslationClient(Anthropic(api_key=api_key))

    missing_filled, _ = rebuild(args.input, args.output, from_log, client)
    print(
        f"Done. Chunks filled via API: {len(missing_filled)}; "
        f"from log: {len(from_log)}; output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
