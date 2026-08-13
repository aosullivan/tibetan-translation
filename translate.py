import os
import argparse
import logging
import traceback
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from src.config import TranslationConfig as cfg
from src.translation_client import TranslationClient
from src.file_handler import FileHandler
from src.translation_manager import TranslationManager

_OUTPUT_ARG_DEFAULT = object()


def _resolve_translation_output(output, input_path: str) -> Path:
    """Write markdown under TRANSLATIONS_OUTPUT_DIR unless -o is an absolute path."""
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / cfg.TRANSLATIONS_OUTPUT_DIR
    if output is _OUTPUT_ARG_DEFAULT:
        stem = Path(input_path).stem
        return out_dir / f"{stem}.en.md"
    p = Path(output).expanduser()
    if p.is_absolute():
        return p
    return out_dir / p


def _resolve_progress_path(progress_arg, out_path: Path) -> Path:
    """Derive a per-input progress path so different texts don't share state."""
    if progress_arg is not None:
        p = Path(progress_arg).expanduser()
        return p if p.is_absolute() else Path.cwd() / p
    return out_path.with_suffix(".progress.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main() -> int:
    parser = argparse.ArgumentParser(description='Translate Tibetan text')
    parser.add_argument('--input', '-i', help='Input file path', default=cfg.INPUT_FILE)
    parser.add_argument(
        '--output',
        '-o',
        help=(
            'Output markdown path. Relative paths are placed under tibetan_translations/. '
            'Default: tibetan_translations/<input-stem>.en.md'
        ),
        default=_OUTPUT_ARG_DEFAULT,
    )
    parser.add_argument(
        '--progress',
        help=(
            'Progress JSON path for resume. '
            'Default: <output path>.progress.json (scoped per input file).'
        ),
        default=None,
    )
    parser.add_argument(
        '--model',
        help=f'Anthropic model id to translate with (default: {cfg.MODEL})',
        default=None,
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        help=f'Max output tokens per chunk (default: {cfg.MAX_TOKENS})',
        default=None,
    )
    args = parser.parse_args()
    if args.model:
        cfg.MODEL = args.model
    if args.max_tokens:
        cfg.MAX_TOKENS = args.max_tokens
    out_path = _resolve_translation_output(args.output, args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_FILE = str(out_path)
    cfg.PROGRESS_FILE = str(_resolve_progress_path(args.progress, out_path))

    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return 1

    try:
        
        client = Anthropic(api_key=api_key)
        translation_client = TranslationClient(client)
        file_handler = FileHandler()
        
        translator = TranslationManager(translation_client, file_handler)
        translator.translate_file(args.input)
        return 0
        
    except KeyboardInterrupt:
        logger.info("Translation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error during translation: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(code=main())
