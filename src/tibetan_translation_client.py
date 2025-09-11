import logging
from anthropic import Anthropic
from .translation_client import TranslationClient
from .config import TranslationConfig as cfg

logger = logging.getLogger(__name__)

class TibetanTranslationClient(TranslationClient):
    """
    Specialized translation client for Vajrayana Buddhist Tibetan texts.
    Extends the base TranslationClient with Buddhist-specific prompts and terminology handling.
    """
    
    def __init__(self, client: Anthropic):
        """Initialize Tibetan translation client with Buddhist context."""
        super().__init__(client)
        self.buddhist_terms = self._load_buddhist_terms()
    
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
                context_prompt = ""
                if summary:
                    context_prompt = f"Summary of previous content: {summary}\n\n"
                if prev_translation:
                    context_prompt += f"Previous chunk's translation: {prev_translation}\n\n"

                message = self.client.messages.create(
                    model=cfg.MODEL,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": self._build_vajrayana_translation_prompt(context_prompt, full_text, tibetan_script)}
                    ]
                )
                
                # Parse response to get translation and any untranslated fragment
                response = ''.join(block.text for block in message.content)
                logger.info(f"Received Vajrayana translation response (length: {len(response)})")
                
                translation, untranslated = self._parse_translation_response(response)
                if untranslated:
                    logger.info(f"Found untranslated fragment (length: {len(untranslated)})")
                self.untranslated_fragment = untranslated
                return translation, untranslated

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Translation failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                continue
    
    def handle_technical_terms(self, text: str) -> str:
        """
        Process text to identify and preserve Buddhist technical terms.
        
        Args:
            text: Input text containing potential Buddhist terms
            
        Returns:
            str: Text with technical terms marked for preservation
        """
        processed_text = text
        
        # Sort terms by length (longest first) to avoid partial replacements
        sorted_terms = sorted(self.buddhist_terms.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Mark known Buddhist technical terms for special handling
        for term, context in sorted_terms:
            if term in processed_text and f"[BUDDHIST_TERM:{term}" not in processed_text:
                # Add context marker for the translation prompt
                processed_text = processed_text.replace(term, f"[BUDDHIST_TERM:{term}:{context}]")
        
        return processed_text
    
    def preserve_sacred_elements(self, text: str) -> str:
        """
        Identify and mark sacred elements (mantras, seed syllables) for preservation.
        
        Args:
            text: Input text that may contain sacred elements
            
        Returns:
            str: Text with sacred elements marked for preservation
        """
        # Common patterns for mantras and seed syllables
        sacred_patterns = [
            # Complete mantras (check these first)
            'a ra pa ca na dhI',                 # Manjushri mantra
            'oM ma Ni pad+me hUM',              # Chenrezig mantra
            'oM tA re tu tA re tu re svA hA',   # Tara mantra
            'gate gate pAragate pArasamgate bodhi svAhA',  # Heart Sutra mantra
            'oM muni muni mahAmuni',
            # Core mantra syllables
            'hU~M', 'HUM', 'oM', 'AH', 'SVAHA', 'SOHA', 'phaT',
            # Seed syllables
            'hrIH', 'hrI', 'jaH', 'khaM', 'traM', 'hraM',
            # Dharani elements
            'tadyathA',
        ]
        
        processed_text = text
        # Sort by length (longest first) to avoid partial replacements
        for pattern in sorted(sacred_patterns, key=len, reverse=True):
            if pattern in text:
                processed_text = processed_text.replace(pattern, f"[SACRED:{pattern}]")
        
        return processed_text
    
    def identify_content_type(self, text: str) -> str:
        """
        Identify the type of Buddhist content for appropriate translation approach.
        
        Args:
            text: Input Wylie text
            
        Returns:
            str: Content type identifier
        """
        # Check for different content types (more comprehensive patterns)
        if any(marker in text for marker in ['mchod pa', 'gsol ba', 'bstod pa', 'phyag \'tshal', 'me tog', 'bdug spos']):
            return 'liturgical'  # Prayers, praises, offerings
        elif any(marker in text for marker in ['rgyud', 'dbang', 'lung', 'khrid']):
            return 'tantric'     # Tantric texts, empowerments
        elif any(marker in text for marker in ['bsgom pa', 'nyams len', 'gdams ngag', 'sgrub', 'sgrub thabs', '\'khor lo']):
            return 'practice'    # Meditation instructions, sadhanas
        elif any(marker in text for marker in ['\'grel pa', 'mchan bu', 'bshad pa']):
            return 'commentary'  # Commentarial literature
        elif any(marker in text for marker in ['tshigs bcad', 'sku gsum', 'ye shes']):
            return 'philosophical'  # Philosophical treatises
        else:
            return 'general'     # General Buddhist content
    
    def get_specialized_prompt_additions(self, content_type: str) -> str:
        """
        Get additional prompt instructions based on content type.
        
        Args:
            content_type: Type of Buddhist content
            
        Returns:
            str: Additional prompt instructions
        """
        additions = {
            'liturgical': """
        LITURGICAL TEXT GUIDELINES:
        - Preserve the devotional and sacred tone
        - Maintain verse structure and rhythm where present
        - Use traditional English Buddhist liturgical language
        - Handle invocations and dedications with appropriate reverence
        """,
            'tantric': """
        TANTRIC TEXT GUIDELINES:
        - Handle empowerment procedures with technical precision
        - Preserve visualization instructions exactly
        - Maintain the sacred and esoteric nature of the content
        - Use established tantric terminology (mandala, deity yoga, etc.)
        """,
            'practice': """
        MEDITATION PRACTICE GUIDELINES:
        - Translate instructions clearly and practically
        - Maintain the progressive structure of meditation stages
        - Use precise terminology for different meditation techniques
        - Preserve the experiential and practical nature of the instructions
        """,
            'commentary': """
        COMMENTARY GUIDELINES:
        - Distinguish between root text and commentary
        - Maintain scholarly precision and analytical structure
        - Preserve philosophical arguments and their logical flow
        - Use appropriate academic Buddhist terminology
        """,
            'philosophical': """
        PHILOSOPHICAL TEXT GUIDELINES:
        - Maintain precise philosophical distinctions and terminology
        - Preserve logical arguments and their structure
        - Use established philosophical vocabulary
        - Handle different schools' terminology appropriately
        """
        }
        
        return additions.get(content_type, "")
    
    def _build_vajrayana_translation_prompt(self, context: str, wylie_text: str, tibetan_script: str = "") -> str:
        """Build specialized translation prompt for Vajrayana Buddhist texts."""
        tibetan_reference = ""
        if tibetan_script:
            tibetan_reference = f"\nTibetan Unicode: {tibetan_script}\n"
        
        return f"""
        You are a highly qualified translator specializing in Vajrayana Buddhist texts with expertise equivalent to a scholar who has:
        - Completed traditional Buddhist studies in Tibetan monasteries
        - Mastered Sanskrit and Tibetan classical languages
        - Deep understanding of Indo-Tibetan Buddhist philosophy across all schools
        - Specialized knowledge of tantric practices, empowerments, and ritual procedures
        - Familiarity with the complete Buddhist canon and major commentarial literature

        SPECIALIZED KNOWLEDGE AREAS:
        - Madhyamaka (Middle Way) philosophy and its Tibetan interpretations
        - Yogacara (Mind-Only) school teachings and terminology
        - Tantric Buddhism: Kriya, Charya, Yoga, and Anuttarayoga Tantra classifications
        - Tibetan Buddhist schools: Nyingma, Kagyu, Sakya, Gelug traditions
        - Bon religious terminology and practices
        - Sanskrit-Tibetan translation conventions established by Lotsawas (translators)

        {context}
        
        VAJRAYANA BUDDHIST TRANSLATION PRINCIPLES:
        
        1. DOCTRINAL ACCURACY:
        - Maintain precise philosophical distinctions (e.g., conventional vs. ultimate truth)
        - Recognize technical terms specific to different Buddhist schools
        - Preserve the graduated path (lamrim) structure and terminology
        - Understand empowerment (abhisheka) classifications and procedures
        
        2. TANTRIC TERMINOLOGY:
        - Recognize deity yoga practices and visualization instructions
        - Handle mandala descriptions with appropriate technical vocabulary
        - Preserve mantra syllables and seed syllables (bija) without translation
        - Understand completion stage (sampannakrama) and generation stage (utpattikrama) practices
        
        3. RITUAL AND LITURGICAL ELEMENTS:
        - Identify offering sequences (puja) and their proper terminology
        - Recognize prayer structures and invocation formulas
        - Handle refuge formulas and bodhisattva vows appropriately
        - Preserve the sacred nature of dedication prayers and aspiration prayers
        
        4. SANSKRIT PRESERVATION:
        - Keep established Sanskrit terms: dharma, sangha, buddha, bodhisattva, etc.
        - Use Sanskrit for technical philosophical terms: sunyata, prajna, upaya, etc.
        - Preserve Sanskrit names of deities, texts, and historical figures
        - Include Tibetan terms in brackets only when clarification is essential
        
        5. CONTEXTUAL SENSITIVITY:
        - Recognize quotations from sutras, tantras, and commentaries
        - Identify different literary genres (root text, commentary, practice manual)
        - Understand the difference between scholarly exposition and practice instructions
        - Maintain appropriate register for different types of content
        
        TECHNICAL TERM HANDLING:
        - [BUDDHIST_TERM:term:context] markers indicate important concepts requiring special attention
        - [SACRED:element] markers indicate mantras, dharanis, or seed syllables to preserve exactly
        - Use established English Buddhist terminology from authoritative translations
        - When uncertain about technical terms, prefer Sanskrit over invented English equivalents
        - Maintain consistency with translations by established scholars (Thurman, Hopkins, etc.)
        
        CRITICAL INSTRUCTIONS FOR INCOMPLETE SENTENCES:
        1. If the translation ends mid-sentence, DO NOT include that partial sentence in the response
        2. Mark incomplete Tibetan text with "UNTRANSLATED:" followed by the remaining Wylie transliteration
        3. Do not provide any English translation for the untranslated portion
        4. Complete sentences must end with proper punctuation
        5. Verify that no partial thoughts or incomplete clauses appear at the end of translations
        
        FORMATTING REQUIREMENTS:
        - Use numbered lists (1., 2., etc.) for enumerations and structured teachings
        - Preserve verse structure for liturgical texts and root verses
        - Use appropriate subheadings for major sections in longer texts
        - Include essential Tibetan terms in brackets only when absolutely necessary for clarity
        - Maintain paragraph structure that reflects the logical flow of the original
        
        EXAMPLES OF PROPER HANDLING:
        
        Mantra preservation:
        Tibetan: "oM a ra pa ca na dhI"
        Translation: "oM a ra pa ca na dhI" (preserve exactly, do not translate)
        
        Technical term handling:
        Tibetan: "byang chub sems dpa'"
        Translation: "bodhisattva" (use established Sanskrit term)
        
        Philosophical precision:
        Tibetan: "kun rdzob bden pa"
        Translation: "conventional truth" (not "relative truth" or "apparent truth")
        
        Do NOT provide commentary, explanatory notes, or scholarly apparatus. Translate directly and accurately while maintaining the sacred and precise nature of the Buddhist teachings.
        
        Wylie transliteration to translate:
        {wylie_text}
        {tibetan_reference}
        """
    
    def _load_buddhist_terms(self) -> dict:
        """Load dictionary of Buddhist technical terms with context."""
        return {
            # Core Buddhist concepts
            'byang chub sems': 'bodhicitta - the mind of enlightenment',
            'byang chub sems dpa\'': 'bodhisattva - enlightenment being',
            'sangs rgyas': 'buddha - awakened one',
            'chos': 'dharma - Buddhist teaching/phenomenon',
            'dge ba\'i bshes gnyen': 'spiritual friend/teacher',
            'bla ma': 'lama - spiritual teacher',
            'rgyud': 'tantra - continuum/tantric text',
            
            # Vajrayana specific terms
            'dbang bskur': 'empowerment/initiation',
            'sgrub thabs': 'sadhana - meditation practice',
            'yi dam': 'yidam - meditation deity',
            'mkha\' \'gro': 'dakini - sky-goer',
            'chos skyong': 'dharma protector',
            'gsang sngags': 'secret mantra/Vajrayana',
            
            # Meditation and practice
            'bsgom pa': 'meditation/cultivation',
            'ting nge \'dzin': 'samadhi - meditative absorption',
            'shes rab': 'prajna - wisdom',
            'thabs': 'upaya - skillful means',
            'zhi gnas': 'shamatha - calm abiding',
            'lhag mthong': 'vipashyana - special insight',
            
            # Philosophical terms
            'stong pa nyid': 'emptiness/sunyata',
            'rten \'brel': 'dependent origination',
            'rang bzhin': 'inherent nature/svabhava',
            'kun rdzob': 'conventional truth',
            'don dam': 'ultimate truth',
        }