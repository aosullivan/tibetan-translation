import logging
import random
import re
import time
from anthropic import Anthropic, APIError, RateLimitError
from .config import TranslationConfig as cfg

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

    def translate_chunk(
        self,
        text: str,
        prev_translation: str = "",
        summary: str = "",
        outline_block: str = "",
    ) -> tuple[str, str, int | None]:
        """Translate a single chunk with retries and error handling.

        Args:
            text: Text to translate
            prev_translation: Translation of previous chunk for context
            summary: Summary of previous content for context
            outline_block: Optional outline window + cursor block to include

        Returns:
            (translated_text, untranslated_fragment, new_cursor_or_none)
        """
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
                    max_tokens=cfg.MAX_TOKENS,
                    messages=[
                        {
                            "role": "user",
                            "content": self._build_translation_prompt(
                                context_prompt, full_text, outline_block
                            ),
                        }
                    ],
                )

                response = "".join(block.text for block in message.content)
                logger.info(f"Received translation response (length: {len(response)})")
                logger.info(f"Raw response: {response}")

                translation, untranslated, cursor = self._parse_translation_response(response)
                if untranslated:
                    logger.info(f"Found untranslated fragment (length: {len(untranslated)})")
                if cursor is not None:
                    logger.info(f"Outline cursor advanced to {cursor}")
                self.untranslated_fragment = untranslated
                return translation, untranslated, cursor

            except RateLimitError as e:
                self._handle_rate_limit(e, attempt, max_retries, initial_delay)
            except APIError as e:
                self._handle_api_error(e, attempt, max_retries, initial_delay)

        raise RuntimeError("translate_chunk exhausted retries without returning")

    @staticmethod
    def _parse_translation_response(response: str) -> tuple[str, str, int | None]:
        """Parse the response to extract translation, untranslated fragment, and optional cursor."""
        # Pull out CURSOR: N if present (may appear anywhere near the end)
        cursor: int | None = None
        m = re.search(r"^\s*CURSOR:\s*(\d+)\s*$", response, flags=re.MULTILINE)
        if m:
            cursor = int(m.group(1))
            response = response[: m.start()] + response[m.end():]

        if "UNTRANSLATED:" not in response:
            return response.strip(), "", cursor

        translation, untranslated = response.split("UNTRANSLATED:", 1)
        if translation.strip().endswith("..."):
            sentences = translation.split(".")
            translation = ".".join(sentences[:-1]) + "."

        return translation.strip(), untranslated.strip(), cursor

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
    def build_outline_section(window: str) -> str:
        """Compose the outline instructions + current window into a prompt block."""
        if not window:
            return ""
        return (
            "STRUCTURAL HEADINGS (outline-driven):\n"
            "You have been given an OUTLINE — a flat list of section headings for this text, each shown in the EXACT markdown form to emit (e.g. '## Title' or '**Title**'). The '→' arrow marks the CURRENT CURSOR: the index of the last heading ALREADY EMITTED (in a previous chunk). A cursor of -1 means nothing has been emitted yet.\n\n"
            "Rules:\n"
            "1. CURSOR INVARIANT: every entry at index ≤ cursor has already been emitted and must NOT be emitted again. You may ONLY emit entries with index STRICTLY GREATER than the cursor. This applies even if the current chunk continues the discussion of a section whose heading was emitted in a previous chunk — you do not re-emit its heading.\n"
            "2. EMIT AN OUTLINE ENTRY ONLY when the CURRENT Tibetan chunk text actually introduces that section for the first time. Do NOT emit upstream outline entries as context or framing; the reader has already seen them. If the chunk is entirely inside an already-emitted section, emit NO new headings.\n"
            "3. When you do emit a heading, copy the outline's rendered line VERBATIM — character for character — including the exact number of '#' marks, the '**' markers, and the title text. Do NOT paraphrase, expand, or summarise the title.\n"
            "4. Watch for Tibetan section-marker phrases ('dang po ni', 'gnyis pa ni', 'de la ... ni', 'de yang', 'bshad pa', 'brjod pa', ending particles 'gsum pa'o'). When you see one, scan the outline FORWARD from the cursor: if an upcoming entry's topic matches the phrase semantically, emit that entry verbatim on its own line (with blank lines before and after).\n"
            "5. If the phrase does not match any upcoming outline entry, it is an in-paragraph enumeration (e.g. 'There are six: first X, second Y...') — translate it as ordinary prose or a numbered list. Do NOT promote it to a heading.\n"
            "6. If genuinely uncertain, translate as prose and add an inline editor note like: [?editor: possible new section — candidate \"X\"]. Do not guess.\n"
            "7. You may emit MULTIPLE outline entries in one chunk if the text genuinely crosses multiple section boundaries. Emit each at its point of introduction. Never emit the same outline entry twice across the whole document.\n"
            "8. Do NOT write any preamble, meta-commentary, horizontal rules ('---'), or framing text. Begin the response directly with the first line of translation (or the first emitted heading).\n\n"
            "After the translation (and after UNTRANSLATED if any), on its own line, write:\n"
            "CURSOR: N\n"
            "where N is the index of the LAST outline entry you emitted in this chunk. If you emitted no new headings, repeat the incoming cursor value unchanged. If multiple headings were emitted, N must be the index of the highest-indexed one. Always include the CURSOR line.\n\n"
            f"{window}\n"
        )

    @staticmethod
    def _build_translation_prompt(context: str, text: str, outline_block: str = "") -> str:
        return f"""
        You are an AI agent for translating classical Tibetan texts into English.
        The source text is Wylie (EWTS) transliteration unless you clearly see Unicode Tibetan.
        You will receive parts of a text sequentially. Your task is to translate the current part 
        while maintaining logical and semantic consistency and continuity with the previous chunk of translation, and the overall translation summary.

        {context}
                     
        CRITICAL INSTRUCTIONS FOR HANDLING INCOMPLETE SENTENCES:
        1. If the current chunk of translation ends mid-sentence, DO NOT include that partial sentence in the response
        2. A partial sentence includes sentences that end with an ellipsis "...", conjunctions, etc. For example:
            "Then first the yogin should..." is a partial sentence because it contains ellipsis
            "First he went travelled there and" is a partial sentence because it ends in a conjunction
            "Acting like a magical creation towards activities...\"" is a partial sentence because it contains ellipsis, even though it is in quotation marks
            "The bodhisattvas who practice the perfections engage in what is utterly free from confusion" is a partial sentence, because there is no ending punctuation mark
            "Through words like "blue" and "yellow" and so forth" is a partial sentence, because there is no ending punctuation mark
        3. For the part of the Tibetan text that formed a partial sentence, i.e. the part which was left untranslated, take the following action: 
            After the translation, on a new line, write "UNTRANSLATED:" followed by ALL the remaining untranslated Tibetan text
        4. Do not provide ANY English translation for the untranslated portion
        5. If the translated chunk of text ends with a complete sentence, do not include the "UNTRANSLATED:" marker
        6. If the translated chunk of text ends with a partial sentence, check you have followed these rules correctly, especially making sure there is no partial sentence ending the translation
        
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

        QUOTATIONS (markdown formatting):
        - Tibetan cites scripture with introductory phrases like "X las", "X nas", "X-r gsungs pa", and closes citations with quotative particles "zhes", "ces", "shes", often followed by verbs of speech ("gsungs so", "bshad do", "smras so"). When you translate such a citation, render the quoted material as a markdown blockquote — each line of the quotation prefixed with "> ".
        - The introductory line (e.g. "X says in Y:") must stay as normal prose, followed by a blank line, then the blockquote.
        - If the quoted content is verse, put each verse line on its own line inside the blockquote. End every quoted line except the last with a single backslash ("\\") so markdown renders each as a hard line break.
        - If the quoted content is prose, use a single "> " paragraph with no backslashes.
        - Do NOT include the closing quotative particle ("zhes gsungs so" etc.) inside the blockquote; render it as a short closing line after the blockquote, e.g. "— thus it is said." or omit it if redundant.
        - Do not wrap the blockquote lines in straight quotes ("...") in addition to the "> " prefix.

        Example of a quoted verse:
        Tibetan: slob dpon d+hom+bi pas lhan cig skyes grub las / gang zhig slob ma dngos grub thob gyur kyang / nga rgyal dbang gis bla ma la mi 'dud / bla ma'i bka' las 'das par gyur pa yis / skad cig nyid la mnar med par 'gro'o / zhes gsungs so /
        Correct:
        The ācārya Ḍombipa says in the Sahajasiddhi [lhan cig skyes grub]:

        > Whatever disciple, even having attained accomplishment,\\
        > Out of arrogance does not prostrate to the guru—\\
        > Through transgressing the guru's command,\\
        > In an instant enters the Avīci [hell].

        — thus it is said.

        {outline_block}

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
