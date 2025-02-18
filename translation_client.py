import logging
import random
import time
from anthropic import Anthropic, APIError, RateLimitError
from config import TranslationConfig as cfg

logger = logging.getLogger(__name__)

class TranslationClient:
    def __init__(self, client: Anthropic):
        self.client = client

    def translate_chunk(self, text: str, prev_translation: str = "", summary: str = "") -> str:
        """Translate a single chunk with retries and error handling"""
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
                        {"role": "user", "content": self._build_translation_prompt(context_prompt, text)}
                    ]
                )
                
                return ''.join(block.text for block in message.content)

            except RateLimitError as e:
                self._handle_rate_limit(e, attempt, max_retries, initial_delay)
            except APIError as e:
                self._handle_api_error(e, attempt, max_retries, initial_delay)

    def generate_summary(self, text: str) -> str:
        """Generate summary with error handling"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-latest",
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
        - Maintain consistency in terminology with previous translations
        - Ensure smooth transitions between chunks
        - Use Sanskrit terms if that is appropriate according to the norms of Tibetan translations into English
        - Include original Tibetan terms in brackets if the term is particularly technical or obscure, or the translation needs clarification
        - Use enumerations where applicable (e.g., "Second, blah blah" becomes "2. Blah blah")
        - If the text says something like 'There are two parts', list them as '1.' and '2.' Remember these enumerations and use them as headings of they are explained in more detail later
        - If an enumerated part has subparts or subsections, enumerate them as 1.1, 1.2, etc. and likewise if there are further sub-enumerations then use 1.1.1, 1.1.2 and so forth 
        - Do not put a dot after subenumerations, e.g. 1.1.2 not 1.1.2. and 1.2 not 1.2. and so forth
        - Create numbered lists for any enumerated items, but otherwise do not use lists
        - Do not say 'Here is the translation' or add any of your own comments or words. Give me ONLY the translated words.
        
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
