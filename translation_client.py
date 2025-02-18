import logging
import random
import time
from anthropic import Anthropic, APIError, RateLimitError
from config import TranslationConfig as cfg

logger = logging.getLogger(__name__)

class TranslationClient:
    def __init__(self, client: Anthropic):
        """
        Initialize translation client.
        
        Attributes:
            untranslated_fragment: Stores any incomplete sentence from previous chunk
        """
        self.client = client
        self.untranslated_fragment = ""

    def translate_chunk(self, text: str, prev_translation: str = "", summary: str = "") -> tuple[str, str]:
        """
        Translate a single chunk with retries and error handling.
        
        This method handles sentences that are split across chunks by:
        1. Prepending any untranslated fragment from the previous chunk
        2. Identifying incomplete sentences at the end of current chunk
        3. Storing incomplete sentences to be prepended to the next chunk
        
        Args:
            text: Text to translate
            prev_translation: Translation of previous chunk for context
            summary: Summary of previous content for context
            
        Returns:
            tuple[str, str]: (translated_text, untranslated_fragment)
            The untranslated_fragment will be prepended to the next chunk
        """
        # Prepend any untranslated fragment from previous chunk
        full_text = self.untranslated_fragment + text
        self.untranslated_fragment = ""
        
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
                    model=cfg.MODEL,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": self._build_translation_prompt(context_prompt, full_text)}
                    ]
                )
                
                # Parse response to get translation and any untranslated fragment
                response = ''.join(block.text for block in message.content)
                translation, untranslated = self._parse_translation_response(response)
                self.untranslated_fragment = untranslated
                return translation, untranslated

            except RateLimitError as e:
                self._handle_rate_limit(e, attempt, max_retries, initial_delay)
            except APIError as e:
                self._handle_api_error(e, attempt, max_retries, initial_delay)

    @staticmethod
    def _parse_translation_response(response: str) -> tuple[str, str]:
        """
        Parse the response to extract translation and untranslated fragment.
        
        The response may contain an "UNTRANSLATED:" marker followed by Tibetan text
        that represents an incomplete sentence at the end of the chunk. This text
        will be carried forward to the next chunk for translation.
        
        Returns:
            tuple[str, str]: (translated_text, untranslated_fragment)
        """
        if "UNTRANSLATED:" in response:
            parts = response.split("UNTRANSLATED:", 1)
            return parts[0].strip(), parts[1].strip()
        return response.strip(), ""

    def generate_summary(self, text: str) -> str:
        """Generate summary with error handling"""
        try:
            message = self.client.messages.create(
                model=cfg.MODEL,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": self._build_summary_prompt(text)}
                ]
            )
            return ''.join(block.text for block in message.content)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

    @staticmethod
    def _build_translation_prompt(context: str, text: str) -> str:
        return f"""
        You are an AI agent for translating classical Tibetan texts into English. 
        You will receive parts of a text sequentially. Your task is to translate the current part 
        while maintaining consistency with the previous translations.

        {context}
                     
        Guidelines:
        - Translate as much complete content as possible
        - If the chunk ends mid-sentence, DO NOT complete the partial sentence
        - Instead, if you encounter an incomplete sentence at the end, add "UNTRANSLATED:" followed by the untranslated Tibetan text
        - This untranslated text will be prepended to the next chunk for proper translation
        - If a chunk starts with what appears to be the continuation of a previous sentence, translate it as-is
        - Maintain consistency in terminology with previous translations
        - Use Sanskrit terms if that is appropriate according to the norms of Tibetan translations into English
        - Include original Tibetan terms in brackets if the term is particularly technical or obscure
        - Use enumerations where applicable (e.g., "Second, blah blah" becomes "2. Blah blah")
        - If the text says something like 'There are two parts', list them as '1.' and '2.'
        - If an enumerated part has subparts, enumerate them as 1.1, 1.2, etc.
        - Do not put a dot after subenumerations (use 1.1.2 not 1.1.2.)
        - Create numbered lists for enumerated items only
        - Give ONLY the translated words and any untranslated fragment
        
        Translate the following text:
        {text}
        """

    @staticmethod
    def _build_summary_prompt(text: str) -> str:
        return f"""
        Provide a brief summary of the key points and context from this translated text. 
        Focus on main themes, key terms, and any ongoing topics that would be important 
        for maintaining consistency in subsequent translations:

        {text}
        """

    def _handle_rate_limit(self, e: RateLimitError, attempt: int, max_retries: int, initial_delay: int) -> None:
        if attempt == max_retries - 1:
            logger.error(f"Rate limit exceeded after {max_retries} attempts")
            raise
        
        retry_after = int(e.response.headers.get('retry-after', initial_delay))
        sleep_time = retry_after + random.uniform(0, 0.5)
        logger.warning(f"Rate limited. Retrying in {sleep_time:.1f} seconds...")
        time.sleep(sleep_time)

    def _handle_api_error(self, e: APIError, attempt: int, max_retries: int, initial_delay: int) -> None:
        logger.error(f"API error: {str(e)}")
        if e.status_code and e.status_code >= 500:
            if attempt == max_retries - 1:
                raise
            time.sleep(initial_delay * (2 ** attempt))
        else:
            raise
