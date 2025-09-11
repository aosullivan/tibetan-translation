import logging
from anthropic import Anthropic
from .translation_client import TranslationClient
from src.config import TranslationConfig

logger = logging.getLogger(__name__)

class TibetanTranslationClient(TranslationClient):
    """
    Specialized translation client for Vajrayana Buddhist Tibetan texts.
    Extends the base TranslationClient with Buddhist-specific prompts and terminology handling.
    """
    
    def __init__(self, client: Anthropic):
        """Initialize Tibetan translation client with Buddhist context."""
        super().__init__(client)
    
    def translate_vajrayana_context(self, wylie_text: str, tibetan_script: str = "", 
                                  prev_translation: str = "", summary: str = "") -> tuple[str, str]:
        """
        Translate Tibetan text with Vajrayana Buddhist expertise.
        
        Args:
            wylie_text: Wylie transliteration of Tibetan text
            tibetan_script: Unicode Tibetan script (optional, for reference)
            prev_translation: Translation of previous chunk for context
            summary: Summary of previous content for context
            
        Returns:
            tuple[str, str]: (translated_text, untranslated_fragment)
        """
        # Prepend any untranslated fragment from previous chunk
        full_text = (self.untranslated_fragment + " " + wylie_text).strip()
        self.untranslated_fragment = ""
        
        max_retries = 5
        initial_delay = 1

        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=TranslationConfig.MODEL,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": self._build_vajrayana_translation_prompt(full_text, tibetan_script)}
                    ]
                )
                
                # Extract raw response and return as pure English translation
                response = ''.join(block.text for block in message.content).strip()
                logger.info(f"Received Vajrayana translation response (length: {len(response)})")
                self.untranslated_fragment = ""
                return response, ""

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Translation failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                continue
    
    def _build_vajrayana_translation_prompt(self, wylie_text: str, tibetan_script: str = "") -> str:
        """Build specialized translation prompt for Vajrayana Buddhist texts."""
        tibetan_reference = ""
        if tibetan_script:
            tibetan_reference = f"\nTibetan Unicode: {tibetan_script}\n"
        
        return f"""
        Translate the following Tibetan Wylie transliteration into English.
        Output ONLY the English translation as plain text. Do not include Tibetan script, Wylie, labels, headers, or any extra commentary.
        
        Wylie transliteration to translate:
        {wylie_text}
        {tibetan_reference}
        """