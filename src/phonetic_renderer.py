"""
Phonetic renderer for Wylie transliteration to readable phonetic approximations.
Handles Tibetan pronunciation rules, tone markers, and Sanskrit terms.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PhoneticRenderer:
    """Converts Wylie transliteration to phonetic approximations."""
    
    def __init__(self, style: str = 'default'):
        """Initialize the phonetic renderer with pronunciation rules."""
        self.style = style
        
        # Basic Wylie to phonetic consonant mappings
        self.consonant_map = {
            # Basic consonants
            'k': 'k', 'kh': 'kh', 'g': 'g', 'ng': 'ng',
            'c': 'ch', 'ch': 'ch', 'j': 'j', 'ny': 'ny',
            't': 't', 'th': 'th', 'd': 'd', 'n': 'n',
            'p': 'p', 'ph': 'ph', 'b': 'b', 'm': 'm',
            'ts': 'ts', 'tsh': 'tsh', 'dz': 'dz', 'w': 'w',
            'zh': 'zh', 'z': 'z', "'": '', 'y': 'y',
            'r': 'r', 'l': 'l', 'sh': 'sh', 's': 's', 'h': 'h'
        }
        
        # Vowel mappings
        self.vowel_map = {
            'a': 'a', 'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o',
            'A': 'aa', 'I': 'ii', 'U': 'uu', 'E': 'ee', 'O': 'oo'
        }
        
        # Special Tibetan phonetic rules
        self.phonetic_rules = [
            # Final consonants are often silent or modified
            (r'([aeiou])g$', r'\1'),  # Final 'g' often silent
            (r'([aeiou])d$', r'\1'),  # Final 'd' often silent
            (r'([aeiou])s$', r'\1'),  # Final 's' often silent
            
            # Consonant clusters simplification
            (r'([kg])r', r'\1r'),     # Keep kr, gr clusters
            (r'([pb])r', r'\1r'),     # Keep pr, br clusters
            (r'([td])r', r'tr'),      # Simplify dr to tr
            
            # Aspirated consonants
            (r'kh', 'kh'), (r'ph', 'ph'), (r'th', 'th'), (r'tsh', 'tsh'),
            
            # Nasalization
            (r'ng([aeiou])', r'ng\1'),
            (r'ny([aeiou])', r'ny\1'),
            
            # Common Tibetan sound changes
            (r'wa', 'wa'), (r'ya', 'ya'), (r'ra', 'ra'), (r'la', 'la'),
        ]
        
        # Sanskrit term phonetic mappings
        self.sanskrit_phonetics = {
            'om': 'ohm', 'aum': 'ohm',
            'ah': 'ah', 'hum': 'hoom', 'hung': 'hoong',
            'svaha': 'so-ha', 'soha': 'so-ha',
            'phat': 'peht', 'hrih': 'hreeh', 'hri': 'hree',
            'mantra': 'man-tra', 'dharani': 'dha-ra-nee',
            'mudra': 'mu-dra', 'yantra': 'yan-tra',
            'mandala': 'man-da-la', 'stupa': 'stoo-pa',
            'buddha': 'bud-dha', 'dharma': 'dhar-ma',
            'sangha': 'sang-ha', 'bodhisattva': 'bo-dhi-sat-tva',
            'prajna': 'praj-nya', 'sutra': 'soo-tra',
            'tantra': 'tan-tra', 'sadhana': 'sa-dha-na',
            'guru': 'gu-ru', 'lama': 'la-ma'
        }
        
        # Mantra and seed syllable phonetics (special handling)
        self.mantra_phonetics = {
            'oM': 'OHM', 'aH': 'AH', 'hUM': 'HOOM',
            'hrIH': 'HREEH', 'hrI': 'HREE',
            'phaT': 'PEHT', 'swa': 'swa', 'ha': 'ha',
            'so': 'so', 'hung': 'hoong',
            'mani': 'ma-ni', 'padme': 'pad-me',
            'vajra': 'vaj-ra', 'sattva': 'sat-tva',
            'tare': 'ta-re', 'tuttare': 'tut-ta-re',
            'ture': 'tu-re', 'mama': 'ma-ma',
            'ayur': 'a-yur', 'jnana': 'gya-na',
            'punye': 'pun-ye', 'pushti': 'push-ti',
            'kuru': 'ku-ru', 'ye': 'ye'
        }
        
        # Special Tibetan Buddhist terms
        self.tibetan_buddhist_terms = {
            'bcom': 'chom', 'ldan': 'den', 'das': 'de',
            'sangs': 'sang', 'rgyas': 'gye',
            'byang': 'chang', 'chub': 'chub',
            'sems': 'sem', 'dpa': 'pa',
            'rje': 'je', 'btsun': 'tsun',
            'sku': 'ku', 'gsung': 'sung', 'thugs': 'thuk',
            'yon': 'yon', 'tan': 'ten',
            'bka': 'ka', 'gdams': 'dam',
            'rgyud': 'gyü', 'sde': 'de',
            'snying': 'nying', 'thig': 'thik', 'le': 'le',
            'rdzogs': 'dzok', 'chen': 'chen',
            'khyung': 'khyung', 'chen': 'chen',
            'mkha': 'kha', 'gro': 'dro'
        }
        
        # Tone and emphasis markers
        self.tone_markers = {
            '~': ' (approx)', '*': ' (emph)', '+': '', '?': ' (?)',
            '_': ' ', '/': ' | ', '\\': ' \\ '
        }
    
    def handle_mantras_and_seed_syllables(self, text: str) -> str:
        """
        Handle mantras and seed syllables with special phonetic rendering.
        
        Args:
            text: Input text that may contain mantras or seed syllables
            
        Returns:
            Text with mantras and seed syllables processed
        """
        if not text:
            return ""
        
        words = text.split()
        processed_words = []
        
        for word in words:
            # Clean word for comparison
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check mantra phonetics first (case-sensitive for mantras)
            if clean_word in self.mantra_phonetics:
                phonetic = self.mantra_phonetics[clean_word]
                processed_word = self._preserve_word_format(word, phonetic)
                processed_words.append(processed_word)
            # Check Tibetan Buddhist terms
            elif clean_word.lower() in self.tibetan_buddhist_terms:
                phonetic = self.tibetan_buddhist_terms[clean_word.lower()]
                processed_word = self._preserve_word_format(word, phonetic)
                processed_words.append(processed_word)
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def format_phonetic_output(self, phonetic_text: str) -> str:
        """
        Create consistent phonetic output formatting.
        
        Args:
            phonetic_text: Raw phonetic text
            
        Returns:
            Formatted phonetic text with consistent styling
        """
        if not phonetic_text:
            return ""
        
        # Clean up spacing
        result = re.sub(r'\s+', ' ', phonetic_text).strip()
        
        # Ensure consistent hyphenation for multi-syllable words
        result = re.sub(r'([a-z])([A-Z])', r'\1-\2', result)  # Add hyphens between case changes
        
        # Standardize mantra formatting (keep uppercase for sacred syllables)
        mantra_patterns = [
            (r'\bOHM\b', 'OHM'),
            (r'\bAH\b', 'AH'),
            (r'\bHOOM\b', 'HOOM'),
            (r'\bHREEH?\b', 'HREEH'),
            (r'\bPEHT?\b', 'PEHT')
        ]
        
        for pattern, replacement in mantra_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def process_complex_tibetan_text(self, wylie_text: str) -> str:
        """
        Process complex Tibetan texts with comprehensive phonetic handling.
        
        Args:
            wylie_text: Complex Wylie transliteration text
            
        Returns:
            Comprehensive phonetic rendering
        """
        if not wylie_text:
            return ""
        
        # Split into manageable chunks (sentences or phrases)
        chunks = re.split(r'[.!?/]+', wylie_text)
        processed_chunks = []
        
        for chunk in chunks:
            if chunk.strip():
                # Apply all phonetic processing steps
                processed = self.generate_phonetic(chunk.strip())
                processed = self.handle_mantras_and_seed_syllables(processed)
                processed = self.format_phonetic_output(processed)
                processed_chunks.append(processed)
        
        return ' | '.join(processed_chunks)
    
    def generate_phonetic(self, wylie_text: str) -> str:
        """
        Generate phonetic approximation from Wylie transliteration.
        
        Args:
            wylie_text: Input Wylie transliteration
            
        Returns:
            Phonetic approximation string
        """
        if not wylie_text or not wylie_text.strip():
            return ""
        
        try:
            # Clean and preprocess the input
            processed_text = self._preprocess_wylie(wylie_text)
            
            # Handle Sanskrit terms, mantras, and special terms first
            words = processed_text.split()
            result_words = []
            
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                clean_lower = clean_word.lower()
                
                # Check mantras first (case-sensitive)
                if clean_word in self.mantra_phonetics:
                    phonetic = self.mantra_phonetics[clean_word]
                    result_words.append(self._preserve_word_format(word, phonetic))
                # Check Sanskrit terms
                elif clean_lower in self.sanskrit_phonetics:
                    phonetic = self.sanskrit_phonetics[clean_lower]
                    result_words.append(self._preserve_word_format(word, phonetic))
                # Check Tibetan Buddhist terms
                elif clean_lower in self.tibetan_buddhist_terms:
                    phonetic = self.tibetan_buddhist_terms[clean_lower]
                    result_words.append(self._preserve_word_format(word, phonetic))
                else:
                    # Apply regular phonetic conversion
                    phonetic = self._convert_syllable_to_phonetic(word)
                    result_words.append(phonetic)
            
            # Join words and apply pronunciation rules
            result = ' '.join(result_words)
            result = self.apply_pronunciation_rules(result)
            result = self.handle_tone_markers(result)
            result = self.format_phonetic_output(result)
            
            logger.debug(f"Phonetic conversion: '{wylie_text}' -> '{result}'")
            return result.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate phonetic for '{wylie_text}': {str(e)}")
            return wylie_text  # Fallback to original
    
    def apply_pronunciation_rules(self, phonetic_text: str) -> str:
        """
        Apply Tibetan pronunciation rules to phonetic text.
        
        Args:
            phonetic_text: Input phonetic text
            
        Returns:
            Text with pronunciation rules applied
        """
        if not phonetic_text:
            return ""
        
        result = phonetic_text
        
        # Apply each phonetic rule
        for pattern, replacement in self.phonetic_rules:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def handle_tone_markers(self, text: str) -> str:
        """
        Handle tone and emphasis markers in phonetic text.
        
        Args:
            text: Input text with tone markers
            
        Returns:
            Text with tone markers processed
        """
        if not text:
            return ""
        
        result = text
        
        # Process tone markers
        for marker, replacement in self.tone_markers.items():
            result = result.replace(marker, replacement)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _preprocess_wylie(self, wylie_text: str) -> str:
        """Preprocess Wylie text for phonetic conversion."""
        # Remove common Wylie markup
        text = wylie_text.strip()
        
        # Handle common Wylie conventions
        text = re.sub(r'\+', '', text)  # Remove stacking markers
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text
    

    
    def _split_into_syllables(self, text: str) -> List[str]:
        """Split Wylie text into syllables for processing."""
        # Simple syllable splitting based on spaces and common patterns
        # This is a simplified approach - real Tibetan syllable detection is complex
        
        # Split on spaces first
        words = text.split()
        syllables = []
        
        for word in words:
            # Further split long words that might contain multiple syllables
            if len(word) > 6:  # Arbitrary threshold for long words
                # Split on vowel-consonant boundaries (simplified)
                parts = re.split(r'([aeiouAEIOU][^aeiouAEIOU]+)', word)
                syllables.extend([part for part in parts if part.strip()])
            else:
                syllables.append(word)
        
        return syllables
    
    def _convert_syllable_to_phonetic(self, syllable: str) -> str:
        """Convert a single syllable to phonetic representation."""
        if not syllable:
            return ""
        
        # Preserve original case and punctuation
        original = syllable
        result = syllable  # Don't lowercase yet - need to preserve case for vowel mappings
        
        # Remove punctuation for processing, but remember it
        punctuation = re.findall(r'[^\w\s]', result)
        result = re.sub(r'[^\w\s]', '', result)
        
        # Apply vowel mappings first (case-sensitive, longest first)
        vowels = sorted(self.vowel_map.keys(), key=len, reverse=True)
        for wylie_vowel in vowels:
            if wylie_vowel in result:
                result = result.replace(wylie_vowel, self.vowel_map[wylie_vowel])
        
        # Now convert to lowercase for consonant processing
        result = result.lower()
        
        # Apply consonant mappings (longest first to avoid partial matches)
        consonants = sorted(self.consonant_map.keys(), key=len, reverse=True)
        for wylie_cons in consonants:
            if wylie_cons in result:
                result = result.replace(wylie_cons, self.consonant_map[wylie_cons])
        
        # Restore punctuation
        if punctuation:
            result += ''.join(punctuation)
        
        # Preserve original capitalization for first letter
        if original and original[0].isupper() and result:
            result = result[0].upper() + result[1:]
        
        return result
    
    def _preserve_word_format(self, original: str, replacement: str) -> str:
        """Preserve capitalization and punctuation when replacing words."""
        # Simple preservation - capitalize if original was capitalized
        if original and original[0].isupper():
            replacement = replacement.capitalize()
        
        # Preserve trailing punctuation
        trailing_punct = re.search(r'[^\w\s]*$', original)
        if trailing_punct:
            replacement += trailing_punct.group()
        
        return replacement
    
    def validate_phonetic_output(self, phonetic_text: str) -> Tuple[bool, str]:
        """
        Validate phonetic output for quality and correctness.
        
        Args:
            phonetic_text: Generated phonetic text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not phonetic_text:
            return False, "Empty phonetic output"
        
        if not phonetic_text.strip():
            return False, "Phonetic output contains only whitespace"
        
        # Check for reasonable length (not too short or too long)
        if len(phonetic_text) < 1:
            return False, "Phonetic output too short"
        
        if len(phonetic_text) > 1000:
            return False, "Phonetic output unusually long"
        
        # Check for reasonable character set (basic Latin + common punctuation)
        valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -()[]{}.,;:!?\'"/\\|~*+_')
        invalid_chars = set(phonetic_text) - valid_chars
        if invalid_chars:
            return False, f"Contains invalid characters: {', '.join(sorted(invalid_chars))}"
        
        return True, ""
    
    def generate_phonetic_with_validation(self, wylie_text: str) -> Tuple[str, bool, str]:
        """
        Generate phonetic with comprehensive validation.
        
        Args:
            wylie_text: Input Wylie text
            
        Returns:
            Tuple of (phonetic_text, success, error_message)
        """
        if not wylie_text:
            return "", False, "Empty input"
        
        try:
            phonetic = self.generate_phonetic(wylie_text)
            
            # Validate output
            is_valid, error_msg = self.validate_phonetic_output(phonetic)
            if not is_valid:
                logger.warning(f"Invalid phonetic output: {error_msg} - Input: '{wylie_text[:50]}...' - Output: '{phonetic[:50]}...'")
                return wylie_text, False, f"Validation failed: {error_msg}"
            
            return phonetic, True, ""
            
        except Exception as e:
            error_msg = f"Phonetic generation error: {str(e)}"
            logger.error(f"{error_msg} - Input: '{wylie_text[:50]}...'")
            return wylie_text, False, error_msg