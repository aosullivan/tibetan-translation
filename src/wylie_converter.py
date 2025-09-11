"""
Wylie to Tibetan Unicode converter using pyewts library.
Handles standard Wylie transliteration, Sanskrit terms, and special characters.
"""

import logging
import re
from typing import Optional, Dict, Set
from pyewts import pyewts

logger = logging.getLogger(__name__)


class WylieConverter:
    """Converter for Wylie transliteration to Tibetan Unicode script."""
    
    def __init__(self):
        """Initialize the converter with pyewts and special character mappings."""
        self.converter = pyewts()
        
        # Sanskrit terms that should be handled specially
        self.sanskrit_terms = {
            'om', 'ah', 'hum', 'svaha', 'soha', 'phat', 'hrih', 'hri',
            'mantra', 'dharani', 'mudra', 'yantra', 'mandala', 'stupa',
            'buddha', 'dharma', 'sangha', 'bodhisattva', 'arhat', 'prajna',
            'sutra', 'tantra', 'sadhana', 'puja', 'guru', 'lama'
        }
        
        # Special Wylie characters and their handling
        self.special_chars = {
            '+': '',  # Wylie stacking marker - remove for display
            '*': '',  # Wylie emphasis marker - remove for display
            '~': '',  # Wylie approximation marker - remove for display
            '?': '',  # Wylie uncertainty marker - remove for display
        }
        
        # Common Wylie patterns that need preprocessing
        self.wylie_patterns = {
            r'\s+': ' ',  # Normalize whitespace
            r'([a-z])\+([a-z])': r'\1\2',  # Handle stacked consonants
            r'([a-z])\*([a-z])': r'\1\2',  # Handle emphasized characters
        }
    
    def convert_to_tibetan(self, wylie_text: str) -> str:
        """
        Convert Wylie transliteration to Tibetan Unicode script.
        
        Args:
            wylie_text: Input text in Wylie transliteration
            
        Returns:
            Tibetan Unicode text, or original text if conversion fails
        """
        if not wylie_text or not wylie_text.strip():
            return ""
        
        try:
            # Preprocess the text
            processed_text = self.process_special_characters(wylie_text)
            processed_text = self.handle_sanskrit_terms(processed_text)
            
            # Convert using pyewts
            tibetan_text = self.converter.toUnicode(processed_text)
            
            # Post-process if needed
            tibetan_text = self._post_process_tibetan(tibetan_text)
            
            logger.debug(f"Converted '{wylie_text}' to '{tibetan_text}'")
            return tibetan_text
            
        except Exception as e:
            logger.warning(f"Failed to convert Wylie text '{wylie_text}': {str(e)}")
            # Return original text as fallback
            return wylie_text
    
    def handle_sanskrit_terms(self, text: str) -> str:
        """
        Handle Sanskrit transliterations appropriately for Tibetan context.
        
        Args:
            text: Input text that may contain Sanskrit terms
            
        Returns:
            Text with Sanskrit terms processed for Tibetan conversion
        """
        if not text:
            return ""
        
        # Convert to lowercase for comparison
        words = text.split()
        processed_words = []
        
        for word in words:
            # Clean word for comparison (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.sanskrit_terms:
                # For Sanskrit terms, we might want to preserve them as-is
                # or apply special Tibetan phonetic rules
                processed_word = self._process_sanskrit_term(word, clean_word)
                processed_words.append(processed_word)
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def process_special_characters(self, text: str) -> str:
        """
        Process special Wylie characters and conventions.
        
        Args:
            text: Input Wylie text with special characters
            
        Returns:
            Processed text ready for conversion
        """
        if not text:
            return ""
        
        processed = text
        
        # Apply Wylie pattern preprocessing
        for pattern, replacement in self.wylie_patterns.items():
            processed = re.sub(pattern, replacement, processed)
        
        # Handle special characters
        for char, replacement in self.special_chars.items():
            processed = processed.replace(char, replacement)
        
        # Clean up extra spaces
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _process_sanskrit_term(self, original_word: str, clean_word: str) -> str:
        """
        Process a specific Sanskrit term for Tibetan conversion.
        
        Args:
            original_word: Original word with punctuation
            clean_word: Cleaned lowercase word
            
        Returns:
            Processed word for Tibetan conversion
        """
        # Special handling for common mantric syllables
        mantra_mappings = {
            'om': 'oM',
            'ah': 'aH',
            'hum': 'hUM',
            'svaha': 'swa ha',
            'soha': 'so ha',
            'phat': 'phaT',
            'hrih': 'hrI',
            'hri': 'hrI'
        }
        
        if clean_word in mantra_mappings:
            # Replace the clean part while preserving punctuation
            return original_word.replace(clean_word, mantra_mappings[clean_word])
        
        # For other Sanskrit terms, return as-is
        return original_word
    
    def _post_process_tibetan(self, tibetan_text: str) -> str:
        """
        Post-process converted Tibetan text for display.
        
        Args:
            tibetan_text: Converted Tibetan Unicode text
            
        Returns:
            Post-processed Tibetan text
        """
        if not tibetan_text:
            return ""
        
        # Remove any remaining conversion artifacts
        processed = tibetan_text
        
        # Clean up extra spaces
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def validate_conversion(self, wylie_text: str, tibetan_text: str) -> bool:
        """
        Validate that a Wylie to Tibetan conversion is reasonable.
        
        Args:
            wylie_text: Original Wylie text
            tibetan_text: Converted Tibetan text
            
        Returns:
            True if conversion appears valid
        """
        if not wylie_text or not tibetan_text:
            return False
        
        # Basic validation checks
        checks = [
            # Tibetan text should contain Tibetan Unicode characters
            any('\u0f00' <= char <= '\u0fff' for char in tibetan_text),
            # Converted text shouldn't be identical to input (unless very short)
            len(wylie_text) <= 3 or wylie_text != tibetan_text,
            # Shouldn't be empty after conversion
            tibetan_text.strip() != ""
        ]
        
        return all(checks)
    
    def validate_wylie_input(self, wylie_text: str) -> tuple[bool, str]:
        """
        Validate Wylie input text for common issues.
        
        Args:
            wylie_text: Input Wylie text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not wylie_text:
            return False, "Empty input text"
        
        if not wylie_text.strip():
            return False, "Input contains only whitespace"
        
        # Check for invalid characters that might cause conversion issues
        invalid_chars = set(wylie_text) - set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +-*~?/()[]{}.,;:!@#$%^&_=|\\`"\'')
        if invalid_chars:
            return False, f"Contains invalid characters: {', '.join(sorted(invalid_chars))}"
        
        # Check for suspicious patterns
        if len(wylie_text) > 1000:
            return False, "Input text is unusually long (>1000 characters)"
        
        return True, ""
    
    def validate_tibetan_output(self, tibetan_text: str) -> tuple[bool, str]:
        """
        Validate Tibetan Unicode output for correctness.
        
        Args:
            tibetan_text: Tibetan Unicode text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not tibetan_text:
            return False, "Empty output text"
        
        if not tibetan_text.strip():
            return False, "Output contains only whitespace"
        
        # Check for Tibetan Unicode range
        has_tibetan = any('\u0f00' <= char <= '\u0fff' for char in tibetan_text)
        if not has_tibetan:
            return False, "Output does not contain Tibetan Unicode characters"
        
        # Check for common conversion artifacts
        artifacts = ['?', '???', 'ERROR', 'INVALID']
        for artifact in artifacts:
            if artifact in tibetan_text:
                return False, f"Output contains conversion artifact: {artifact}"
        
        return True, ""
    
    def convert_with_fallback(self, wylie_text: str, fallback_strategy: str = 'original') -> tuple[str, bool, str]:
        """
        Convert Wylie to Tibetan with comprehensive error handling and fallback.
        
        Args:
            wylie_text: Input Wylie text
            fallback_strategy: Strategy for handling failures ('original', 'empty', 'phonetic')
            
        Returns:
            Tuple of (result_text, success, error_message)
        """
        if not wylie_text:
            return "", False, "Empty input"
        
        # Validate input
        input_valid, input_error = self.validate_wylie_input(wylie_text)
        if not input_valid:
            logger.warning(f"Invalid Wylie input: {input_error} - Text: '{wylie_text[:50]}...'")
            return self._apply_fallback(wylie_text, fallback_strategy), False, f"Input validation failed: {input_error}"
        
        try:
            # Attempt conversion
            result = self.convert_to_tibetan(wylie_text)
            
            # Validate output
            output_valid, output_error = self.validate_tibetan_output(result)
            if not output_valid:
                logger.warning(f"Invalid Tibetan output: {output_error} - Input: '{wylie_text[:50]}...' - Output: '{result[:50]}...'")
                return self._apply_fallback(wylie_text, fallback_strategy), False, f"Output validation failed: {output_error}"
            
            # Additional conversion quality check
            if not self.validate_conversion(wylie_text, result):
                logger.warning(f"Conversion quality check failed - Input: '{wylie_text[:50]}...' - Output: '{result[:50]}...'")
                return self._apply_fallback(wylie_text, fallback_strategy), False, "Conversion quality check failed"
            
            logger.debug(f"Successfully converted: '{wylie_text}' -> '{result}'")
            return result, True, ""
            
        except Exception as e:
            error_msg = f"Conversion error: {str(e)}"
            logger.error(f"{error_msg} - Input: '{wylie_text[:50]}...'")
            return self._apply_fallback(wylie_text, fallback_strategy), False, error_msg
    
    def _apply_fallback(self, wylie_text: str, strategy: str) -> str:
        """
        Apply fallback strategy for failed conversions.
        
        Args:
            wylie_text: Original Wylie text
            strategy: Fallback strategy to apply
            
        Returns:
            Fallback text result
        """
        if strategy == 'original':
            return wylie_text
        elif strategy == 'empty':
            return ""
        elif strategy == 'phonetic':
            # Simple phonetic approximation as fallback
            return self._simple_phonetic_fallback(wylie_text)
        else:
            logger.warning(f"Unknown fallback strategy: {strategy}, using 'original'")
            return wylie_text
    
    def _simple_phonetic_fallback(self, wylie_text: str) -> str:
        """
        Create a simple phonetic approximation as fallback.
        
        Args:
            wylie_text: Wylie text to convert phonetically
            
        Returns:
            Simple phonetic approximation
        """
        # Basic Wylie to phonetic mappings for fallback
        phonetic_map = {
            'kh': 'kh', 'gh': 'gh', 'ng': 'ng', 'ch': 'ch', 'jh': 'jh', 'ny': 'ny',
            'th': 'th', 'dh': 'dh', 'ph': 'ph', 'bh': 'bh', 'ts': 'ts', 'tsh': 'tsh',
            'dz': 'dz', 'zh': 'zh', 'sh': 'sh', 'k': 'k', 'g': 'g', 'c': 'ch',
            'j': 'j', 't': 't', 'd': 'd', 'n': 'n', 'p': 'p', 'b': 'b', 'm': 'm',
            'y': 'y', 'r': 'r', 'l': 'l', 'w': 'w', 's': 's', 'h': 'h', 'a': 'a',
            'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o'
        }
        
        result = wylie_text
        for wylie, phonetic in phonetic_map.items():
            result = result.replace(wylie, phonetic)
        
        return result