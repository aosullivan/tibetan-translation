import json
import os
import logging
import traceback
import time  # Add this import
from typing import Dict, List
from config import TranslationConfig as cfg

logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self):
        self.progress: Dict[str, bool] = {}
        self.current_output = ""
        self._load_progress()

    def _load_progress(self) -> None:
        """Load progress from progress file if it exists"""
        try:
            with open(cfg.PROGRESS_FILE, 'r') as f:
                self.progress = json.load(f)
        except FileNotFoundError:
            logger.info("No progress file found, starting fresh")
        except json.JSONDecodeError:
            logger.warning("Progress file corrupted, starting fresh")

    def save_progress(self) -> None:
        """Save current progress to file"""
        with open(cfg.PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f)
        
        with open(cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(self.current_output)

    def read_input_file(self, input_file: str) -> str:
        """Read input file with error handling"""
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError as e:
            logger.error(f"Input file {input_file} not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}\n{traceback.format_exc()}")
            raise

    def initialize_output_file(self):
        """Clear or create the output file"""
        with open(cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write('')

    def write_chunk(self, tibetan_text: str, translation: str) -> None:
        """Write only the translation to the output file"""
        self.current_output += f"{translation}\n\n"
        
        with open(cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(self.current_output)

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
            if chunk:
                parts.append(chunk)
            start = end

        return parts
