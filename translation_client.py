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
        Ensures no partial sentences in the translation.
        """
        if "UNTRANSLATED:" not in response:
            # If no marker found, look for last complete sentence
            sentences = [s.strip() + "." for s in response.split(".") if s.strip()]
            if not sentences:
                return "", response.strip()
            return " ".join(sentences), ""
            
        translation, untranslated = response.split("UNTRANSLATED:", 1)
        # Verify translation doesn't end with partial sentence
        if translation.strip().endswith('...'):
            # Remove the partial sentence
            sentences = translation.split('.')
            translation = '.'.join(sentences[:-1]) + '.'
            
        return translation.strip(), untranslated.strip()

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
        while maintaining logical and semantic consistency and continuity with the previous chunk of translation, and the overall translation summary.

        {context}
                     
        CRITICAL INSTRUCTIONS FOR HANDLING INCOMPLETE SENTENCES:
        1. If the current chunk of translation ends mid-sentence, DO NOT include the partial sentence in the response
        2. For the part of the Tibetan text for that was a partial sentence, i.e. which was left untranslated, take the following action: After the translation, on a new line, write "UNTRANSLATED:" followed by ALL the remaining untranslated Tibetan text
        4. Do not provide ANY English translation for the untranslated portion
        5. If the translated chunk of text ends with a complete sentence, do not include the "UNTRANSLATED:" marker
        6. If the translated chunk of text ends with a partial sentence, check you have followed these rules correctly, since if you have followed them, there will be no partial sentence
        
        Example:
        Tibetan: "rgyud bsam gyis mi khyab pa dang gsang ba gnyis su med pa'i rgyud rnams zhus nas spyan drangs pas skor ne ru pa'i rgyud lnga zhes grags cing*/_de thams cad kyang"
        Correct: 
        Having requested the inconceivable tantras and the non-dual secret tantras, [this collection] became renowned as the "Five Tantras of Nerupa". 
        UNTRANSLATED: de thams cad kyang
        
        Incorrect:
        Having requested the inconceivable tantras and the non-dual secret tantras, [this collection] became renowned as the "Five Tantras of Nerupa".  All of these
        UNTRANSLATED: de thams cad kyang

        Additional Guidelines:
        - Do NOT give any commentary or notes, e.g. do not say "Here is the translation" or "I will translate the text"
        - Use Sanskrit terms and names where appropriate
        - Include original Tibetan terms in brackets if technical or unclear
        - Use enumerations (e.g., "Second..." becomes "2.")
        - Use enumerated lists, e.g. instead of 'There are two parts: foo and bar', say '1. Foo.\n2. Bar'
        - For subparts, use format: 1.1, 1.2, etc.
        - Be on the lookout for the difference between prose and verse. Verse lines have 7 or 9 syllables and should be rendered in verses
        
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
