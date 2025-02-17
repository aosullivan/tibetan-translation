import logging
import signal
from config import TranslationConfig as cfg
from translation_client import TranslationClient
from file_handler import FileHandler

logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, translator: TranslationClient, file_handler: FileHandler):
        self.translator = translator
        self.file_handler = file_handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle script interruption gracefully"""
        logger.info("Interruption received, saving progress...")
        self.file_handler.save_progress()
        exit(0)

    def translate_file(self, input_file: str = cfg.INPUT_FILE) -> None:
        """Main translation process"""
        content = self.file_handler.read_input_file(input_file)
        parts = self.file_handler.chunk_text(content)
        
        self.file_handler.initialize_output_file()
        
        previous_translation = ""
        translation_summary = ""

        for i, part in enumerate(parts):
            if str(i) in self.file_handler.progress and self.file_handler.progress[str(i)]:
                logger.info(f"Skipping already translated chunk {i+1}/{len(parts)}")
                continue

            logger.info(f"Processing chunk {i+1}/{len(parts)}")
            try:
                translation = self.translator.translate_chunk(
                    part, previous_translation, translation_summary
                )
                previous_translation = translation
                
                if i % cfg.SUMMARY_INTERVAL == 0:
                    translation_summary = self.translator.generate_summary(previous_translation)

                self.file_handler.write_chunk(part, translation)
                self.file_handler.progress[str(i)] = True
                self.file_handler.save_progress()
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                self.file_handler.save_progress()
                raise
