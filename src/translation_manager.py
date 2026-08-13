import logging
import signal
from collections import deque
from .config import TranslationConfig as cfg
from .translation_client import TranslationClient
from .file_handler import FileHandler
from .outline import OutlineEntry, render_window

logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(
        self,
        translator: TranslationClient,
        file_handler: FileHandler,
        outline: list[OutlineEntry] | None = None,
    ):
        self.translator = translator
        self.file_handler = file_handler
        self.outline = outline or []
        self.cursor = self.file_handler.get_cursor() if self.outline else -1
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

    def _extract_translation_text(self, translation_result) -> str:
        """Extract text from translation result, handling both string and tuple returns."""
        if isinstance(translation_result, tuple):
            return translation_result[0]
        return translation_result

    def translate_file(self, input_file: str = cfg.INPUT_FILE) -> None:
        """Main translation process"""
        self.file_handler.bind_input(input_file)
        content = self.file_handler.read_input_file(input_file)
        parts = self.file_handler.chunk_text(content)

        self.file_handler.initialize_output_file()
        
        previous_translation = ""

        for i, part in enumerate(parts):
            if str(i) in self.file_handler.progress and self.file_handler.progress[str(i)]:
                logger.info(f"Skipping already translated chunk {i+1}/{len(parts)}")
                continue

            logger.info(f"Starting translation for chunk {i+1}/{len(parts)}")
            try:
                # Get effective summary by combining current summary with unsummarized translations
                effective_summary = self.current_summary
                if self.unsummarized_translations:
                    # Extract text from all stored translations
                    translation_texts = [self._extract_translation_text(t) for t in self.unsummarized_translations]
                    effective_summary += "\n" + " ".join(translation_texts)

                outline_block = ""
                if self.outline:
                    # Show enough of the outline that the model can locate distant matches.
                    # For modest outlines (< 200 entries), show everything.
                    if len(self.outline) <= 200:
                        window = render_window(
                            self.outline,
                            self.cursor,
                            before=len(self.outline),
                            after=len(self.outline),
                        )
                    else:
                        window = render_window(self.outline, self.cursor, before=5, after=60)
                    outline_block = self.translator.build_outline_section(window)

                translation = self.translator.translate_chunk(
                    part, previous_translation, effective_summary, outline_block
                )

                new_cursor = None
                if isinstance(translation, tuple) and len(translation) >= 3:
                    new_cursor = translation[2]
                if new_cursor is not None and self.outline:
                    if 0 <= new_cursor < len(self.outline) and new_cursor >= self.cursor:
                        self.cursor = new_cursor
                        self.file_handler.set_cursor(self.cursor)
                    else:
                        logger.warning(
                            "Ignoring out-of-range cursor update: %s (current=%s, max=%s)",
                            new_cursor,
                            self.cursor,
                            len(self.outline) - 1,
                        )

                translation_text = self._extract_translation_text(translation)
                previous_translation = translation_text
                self.unsummarized_translations.append(translation)
                
                if len(self.unsummarized_translations) >= 5:
                    # Extract text from all translations before summary generation
                    translation_texts = [self._extract_translation_text(t) for t in self.unsummarized_translations]
                    combined_text = self.current_summary + "\n" + " ".join(translation_texts)
                    self.current_summary = self.translator.generate_summary(combined_text)
                    self.unsummarized_translations = []
                    logger.info("Summary generation completed")

                self.file_handler.write_chunk(part, translation_text)
                self.file_handler.progress[str(i)] = True
                self.file_handler.save_progress()
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                self.file_handler.save_progress()
                raise
