#!/usr/bin/env python3
"""
Translate a UTF-8 Wylie (EWTS) file to English in chunks.

Uses the same pipeline as translate.py (TranslationManager + TranslationClient):
chunk boundaries snap at '/' or spaces near CHUNK_SIZE, rolling summary every few chunks,
resumable via progress.json.

Requires ANTHROPIC_API_KEY in the environment (.env supported).

Example:
  python wylie_translate_chunks.py -i bstan-gyur_toh2488_rgyud_zi148.wylie.txt
  # writes tibetan_translations/bstan-gyur_toh2488_rgyud_zi148.translation.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from src.config import TranslationConfig as cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate Wylie/EWTS text from a file to English in resumable chunks."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input file (UTF-8 Wylie / EWTS)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output markdown. Relative paths go under tibetan_translations/. "
            "Default: tibetan_translations/<input-stem>.translation.md"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2250,
        metavar="N",
        help="Target characters per chunk before snapping to a boundary (default: 2250)",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Progress JSON path for resume (default: progress.json in cwd)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("translation.log"),
        help="Log file path (default: translation.log)",
    )
    parser.add_argument(
        "--outline",
        type=Path,
        default=None,
        help="Optional markdown outline file (sa bcad) guiding section headings",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / cfg.TRANSLATIONS_OUTPUT_DIR
    out = args.output
    if out is None:
        out = out_dir / f"{inp.stem}.translation.md"
    else:
        out = out.expanduser()
        if out.is_absolute():
            out = out.resolve()
        else:
            out = (out_dir / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set (use .env or export).", file=sys.stderr)
        return 1

    cfg.CHUNK_SIZE = args.chunk_size
    cfg.OUTPUT_FILE = str(out)
    if args.progress is not None:
        cfg.PROGRESS_FILE = str(args.progress.expanduser().resolve())

    log_path = args.log.expanduser().resolve()
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    try:
        from src.translation_client import TranslationClient
        from src.file_handler import FileHandler
        from src.translation_manager import TranslationManager
        from src.outline import load_outline

        outline_entries = None
        if args.outline is not None:
            outline_path = args.outline.expanduser().resolve()
            if not outline_path.is_file():
                print(f"Outline not found: {outline_path}", file=sys.stderr)
                return 1
            outline_entries = load_outline(outline_path)
            logger.info("Loaded outline %s (%d entries)", outline_path, len(outline_entries))

        client = Anthropic(api_key=api_key)
        translation_client = TranslationClient(client)
        file_handler = FileHandler()
        translator = TranslationManager(translation_client, file_handler, outline_entries)
        logger.info("Translating %s -> %s (chunk_size=%s)", inp, out, cfg.CHUNK_SIZE)
        translator.translate_file(str(inp))
        return 0
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted; progress saved if any chunk completed.")
        return 1
    except Exception as e:
        logging.getLogger(__name__).error("Fatal error: %s\n%s", e, traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
