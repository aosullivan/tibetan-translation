import logging
import signal
from collections import deque
from typing import List
from .translation_client import TranslationClient
from .file_handler import FileHandler
from .config import cfg

logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, translation_client: TranslationClient, file_handler: FileHandler):
        self.translation_client = translation_client
        self.file_handler = file_handler
        self.previous_translation = ""
        self.context_summary = ""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        self.recent_translations = deque(maxlen=10)
        self.unsummarized_translations = []
        self.current_summary = ""

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle script interruption gracefully"""
        logger.info("Interruption received, saving progress...")
        self.file_handler.save_progress()
        exit(0)

    def translate_file(self, input_file: str = cfg.INPUT_FILE) -> None:
        """Translate entire file chunk by chunk"""
        chunks = self.file_handler.read_chunks(input_file, cfg.CHUNK_SIZE)
        translated_chunks: List[str] = []

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}")
            try:
                # Get both translation and untranslated fragment
                translation, untranslated = self.translation_client.translate_chunk(
                    chunk,
                    prev_translation=self.previous_translation,
                    summary=self.context_summary
                )
                
                # Store only the translated part
                translated_chunks.append(translation)
                self.previous_translation = translation
                
                # Update context summary periodically
                if i % 5 == 0:  # Every 5 chunks
                    recent_text = " ".join(translated_chunks[-5:])
                    self.context_summary = self.translation_client.generate_summary(recent_text)

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                raise

        # Write all translations to output file
        final_translation = "\n".join(translated_chunks)
        self.file_handler.write_translation(final_translation, cfg.OUTPUT_FILE)