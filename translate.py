import os
import argparse
import logging
import traceback
from anthropic import Anthropic
from dotenv import load_dotenv
from config import TranslationConfig as cfg
from translation_client import TranslationClient
from file_handler import FileHandler
from translation_manager import TranslationManager

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
    parser.add_argument('--output', '-o', help='Output file path', default=cfg.OUTPUT_FILE)
    args = parser.parse_args()

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
