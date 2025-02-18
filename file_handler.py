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
        """Write translation to the output file with appropriate formatting"""
        if cfg.OUTPUT_FORMAT == "markdown":
            formatted_text = f"{translation}\n\n"
            if not self.current_output:
                self.current_output = "# Tibetan Text Translation\n\n"
        else:  # txt format
            formatted_text = f"{translation}\n\n"
            
        self.current_output += formatted_text
        
        # Get byte count of the new chunk
        chunk_bytes = len(formatted_text.encode('utf-8'))
        total_bytes = len(self.current_output.encode('utf-8'))
        
        with open(cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(self.current_output)
            
        logger.info(f"Wrote chunk: {chunk_bytes} bytes (Total: {total_bytes} bytes)")

    @staticmethod
    def chunk_text(content: str) -> List[str]:
        """
        Split content into chunks of appropriate size, respecting Tibetan word boundaries.
        Primary split on Tibetan sentence marker '།'
        Secondary split on space character
        Never splits within a word
        """
        if not content:
            raise ValueError("Empty content provided")

        parts = []
        start = 0
        
        while start < len(content):
            end = start + cfg.CHUNK_SIZE
            
            if end >= len(content):
                # If we're at the end, just take the rest
                chunk = content[start:].strip()
                if chunk:
                    parts.append(chunk)
                break
                
            # Look for sentence marker within window
            window_start = max(start + cfg.CHUNK_SIZE - 100, start)
            window_end = min(start + cfg.CHUNK_SIZE + 100, len(content))
            last_marker = content.rfind('།', window_start, window_end)
            
            if last_marker != -1 and last_marker > start:
                # Found a sentence marker, split there
                end = last_marker + 1
            else:
                # No sentence marker found, try splitting on space
                last_space = content.rfind(' ', window_start, window_end)
                if last_space != -1 and last_space > start:
                    end = last_space
                else:
                    # If no suitable split point found, extend until we find one
                    next_space = content.find(' ', end)
                    next_marker = content.find('།', end)
                    if next_space != -1 and (next_marker == -1 or next_space < next_marker):
                        end = next_space
                    elif next_marker != -1:
                        end = next_marker + 1
                    else:
                        # If no split point found, take the rest of the text
                        end = len(content)
            
            chunk = content[start:end].strip()
            if chunk:
                parts.append(chunk)
            start = end

        return parts
