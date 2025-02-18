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
        full_text = (self.untranslated_fragment + " " + text).strip()
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
                logger.info(f"Received translation response (length: {len(response)})")
                logger.info(f"Raw response: {response}")
                
                translation, untranslated = self._parse_translation_response(response)
                if untranslated:
                    logger.info(f"Found untranslated fragment (length: {len(untranslated)})")
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
        
        The response should contain an "UNTRANSLATED:" marker followed by Tibetan text
        that represents an incomplete sentence at the end of the chunk.
        
        Returns:
            tuple[str, str]: (translated_text, untranslated_fragment)
        """
        if "UNTRANSLATED:" not in response:
            # If no marker is found, assume the last sentence might be incomplete
            sentences = response.split("། ")  # Split on Tibetan sentence boundary marker
            if len(sentences) > 1:
                complete = "། ".join(sentences[:-1]) + "། "
                incomplete = sentences[-1]
                return complete, incomplete
            return response.strip(), ""
            
        parts = response.split("UNTRANSLATED:", 1)
        return parts[0].strip(), parts[1].strip()

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
    def _build_summary_prompt(text: str) -> str:
        return f"""
        Summarize the following Tibetan text in English. Focus on:
        - Main topics and themes
        - Key arguments or points
        - Important names or terms
        
        Keep the summary concise (2-3 sentences).
        
        Text to summarize:
        {text}
        """

    @staticmethod
    def _build_translation_prompt(context: str, text: str) -> str:
        return f"""
        You are an AI agent for translating classical Tibetan texts into English. 
        You will receive parts of a text sequentially. Your task is to translate the current part 
        while maintaining consistency with the previous translations.

        {context}
                     
        Guidelines:        
        - Translate only complete English sentences. 
        - When you encounter an incomplete sentence at the end of the chunk:
          1. Stop at the last complete sentence
          2. Add "UNTRANSLATED:" followed by the remaining untranslated Tibetan text
        - Do NOT give any commentary or any notes. Give ONLY the translation and any untranslated Tibeten text
        - The untranslated text will be prepended to the next chunk
        - Never attempt to complete partial sentences - they must be translated as a whole
        - If you receive text starting with an incomplete sentence (from previous chunk), translate it as part of the first complete sentence
        - Use Sanskrit terms where appropriate
        - Include original Tibetan terms in brackets if technical or unclear
        - Use enumerations where applicable (e.g., "Second..." becomes "2.")
        - For subparts, use format: 1.1, 1.2, etc.
        
        Translate the following text:
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
