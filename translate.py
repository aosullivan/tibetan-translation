import anthropic
import os
import json
import signal
import argparse
from typing import List, Optional, Dict
from pathlib import Path
from anthropic import Anthropic, APIError, RateLimitError
from dotenv import load_dotenv
import logging
import traceback
import time
import random
from config import TranslationConfig as cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, client: Anthropic):
        self.client = client
        self.progress: Dict[int, bool] = {}
        self._load_progress()
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _load_progress(self) -> None:
        """Load progress from file if it exists"""
        if os.path.exists(cfg.PROGRESS_FILE):
            with open(cfg.PROGRESS_FILE, 'r') as f:
                self.progress = json.load(f)

    def _save_progress(self) -> None:
        """Save current progress to file"""
        with open(cfg.PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f)

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle script interruption gracefully"""
        logger.info("Interruption received, saving progress...")
        self._save_progress()
        exit(0)

    @staticmethod
    def chunk_text(content: str) -> List[str]:
        """Split content into chunks of appropriate size"""
        if not content:
            raise ValueError("Empty content provided")

        parts = []
        start = 0
        while start < len(content):
            end = start + cfg.CHUNK_SIZE
            
            if end < len(content):
                # Look for sentence endings (.।॥) within a window
                window_start = max(start + cfg.CHUNK_SIZE - 100, start)
                window_end = min(start + cfg.CHUNK_SIZE + 100, len(content))
                for marker in ['།', '॥', '.', '\n']:
                    last_marker = content.rfind(marker, window_start, window_end)
                    if last_marker != -1:
                        end = last_marker + 1
                        break
            else:
                end = len(content)
            
            chunk = content[start:end].strip()
            if not chunk:  # Skip empty chunks
                continue
            parts.append(chunk)
            start = end

        return parts

    def translate_file(self, input_file: str = cfg.INPUT_FILE) -> None:
        """Main translation process"""
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                content = file.read()
        except FileNotFoundError as e:
            logger.error(f"Input file {input_file} not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}\n{traceback.format_exc()}")
            raise

        parts = self.chunk_text(content)
        previous_translation = ""
        translation_summary = ""

        for i, part in enumerate(parts):
            if str(i) in self.progress and self.progress[str(i)]:
                logger.info(f"Skipping already translated chunk {i+1}/{len(parts)}")
                continue

            logger.info(f"Processing chunk {i+1}/{len(parts)}")
            try:
                full_response = self._translate_chunk(
                    part, previous_translation, translation_summary
                )
                previous_translation = full_response
                
                if i % cfg.SUMMARY_INTERVAL == 0:
                    translation_summary = self._get_summary(previous_translation)

                self._write_chunk(full_response)
                self.progress[str(i)] = True
                self._save_progress()
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}\n{traceback.format_exc()}")
                self._save_progress()
                raise

    @staticmethod
    def _write_chunk(content: str) -> None:
        """Write translated chunk to file with error handling"""
        for attempt in range(cfg.MAX_RETRIES):
            try:
                with open(cfg.OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(content + "\n")
                return
            except Exception as e:
                logger.error(f"Error writing to file (attempt {attempt + 1}/{cfg.MAX_RETRIES}): {str(e)}\n{traceback.format_exc()}")
                if attempt < cfg.MAX_RETRIES - 1:
                    time.sleep(1)
                else:
                    raise IOError(f"Failed to write chunk after {cfg.MAX_RETRIES} attempts: {str(e)}")

    def _translate_chunk(self, part: str, prev_translation: str, summary: str) -> str:
        """Translate a single chunk with retries and error handling"""
        logger = logging.getLogger(__name__)
        max_retries = 5
        initial_delay = 1

        for attempt in range(max_retries):
            try:
                context_prompt = ""
                if summary:
                    context_prompt = f"Summary of previous content: {summary}\n\n"
                if prev_translation:
                    context_prompt += f"Previous chunk's translation: {prev_translation}\n\n"

                message = self.client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": f"""
                         You are an AI agent for translating classical Tibetan texts into English. 
                         You will receive parts of a text sequentially. Your task is to translate the current part 
                         while maintaining consistency with the previous translations.

                         {context_prompt}
                         
                         Guidelines:
                         - Maintain consistency in terminology with previous translations
                         - Ensure smooth transitions between chunks
                         - Use Sanskrit terms if that is appropriate according to the norms of Tibetan translations into English
                         - Include original Tibetan terms in brackets if the term is particularly technical or obscure, or the translation needs clarification
                         - Use enumerations where applicable (e.g., "Second, blah blah" becomes "2. Blah blah")
                         - If the text says something like 'There are two parts', list them as '1.' and '2.' 
                         - Remember these enumerations and use them as headings if they are explained in more detail later. e.g. if the enumeration is '1.1.1 The reasons for asking'. this exact text should be used as the subsequent heading, and not, '1.1.1 First, regarding the reasons for asking:' 
                         - If an enumerated part has subparts or subsections, enumerate them as 1.1, 1.2, etc. and likewise if there are further sub-enumerations then use 1.1.1, 1.1.2 and so forth 
                         - Do not put a dot after enumerations, e.g. 1.1.2 not 1.1.2. and 1.2 not 1.2. and so forth, UNLESS it is the top level enumeration, e.g. '1. The first part'
                         - Create numbered lists for any enumerated items, but otherwise do not use lists
                         - Do not say 'Here is the translation' or add any of your own comments or words. Give me ONLY the translated words.
                         
                         Translate the following text, ensuring it flows naturally from the previous content:

                         {part}
                         """}
                    ]
                )
                
                # Concatenate all text content
                return ''.join(block.text for block in message.content)

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise
                
                # Get retry delay from response headers if available
                retry_after = int(e.response.headers.get('retry-after', initial_delay))
                
                # Add jitter to prevent thundering herd
                sleep_time = retry_after + random.uniform(0, 0.5)
                logger.warning(f"Rate limited. Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                
                # Exponential backoff
                initial_delay *= 2
                
            except APIError as e:
                logger.error(f"API error: {str(e)}")
                if e.status_code and e.status_code >= 500:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(initial_delay * (2 ** attempt))
                else:
                    # Don't retry other API errors
                    raise

    def _get_summary(self, text: str) -> str:
        """Generate summary with error handling"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": f"""
                     Provide a brief summary of the key points and context from this translated text. 
                     Focus on main themes, key terms, and any ongoing topics that would be important 
                     for maintaining consistency in subsequent translations:

                     {text}
                     """}
                ]
            )
            return ''.join(block.text for block in message.content)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

def main() -> None:
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
        translator = TranslationManager(client)
        translator.translate_file(args.input)
    except KeyboardInterrupt:
        logger.info("Translation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error during translation: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)