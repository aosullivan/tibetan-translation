"""
Unit tests for WylieConverter class.
Tests Wylie to Tibetan conversion, Sanskrit handling, and special characters.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wylie_converter import WylieConverter


class TestWylieConverter(unittest.TestCase):
    """Test cases for WylieConverter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = WylieConverter()
    
    def test_init(self):
        """Test converter initialization."""
        self.assertIsNotNone(self.converter.converter)
        self.assertIsInstance(self.converter.sanskrit_terms, set)
        self.assertIsInstance(self.converter.special_chars, dict)
        self.assertIsInstance(self.converter.wylie_patterns, dict)
    
    def test_convert_to_tibetan_basic(self):
        """Test basic Wylie to Tibetan conversion."""
        # Test with simple Wylie syllables
        test_cases = [
            ("ka", "ཀ"),
            ("kha", "ཁ"),
            ("ga", "ག"),
            ("nga", "ང"),
            ("tshogs", "ཚོགས"),
        ]
        
        for wylie, expected_tibetan in test_cases:
            with self.subTest(wylie=wylie):
                result = self.converter.convert_to_tibetan(wylie)
                # Check that we get Tibetan Unicode characters
                self.assertTrue(any('\u0f00' <= char <= '\u0fff' for char in result))
    
    def test_convert_empty_input(self):
        """Test conversion with empty or None input."""
        self.assertEqual(self.converter.convert_to_tibetan(""), "")
        self.assertEqual(self.converter.convert_to_tibetan("   "), "")
        self.assertEqual(self.converter.convert_to_tibetan(None), "")
    
    @patch('wylie_converter.logger')
    def test_convert_with_error(self, mock_logger):
        """Test conversion error handling."""
        # Mock pyewts to raise an exception
        with patch.object(self.converter.converter, 'toUnicode', side_effect=Exception("Test error")):
            result = self.converter.convert_to_tibetan("test")
            # Should return original text as fallback
            self.assertEqual(result, "test")
            mock_logger.warning.assert_called_once()
    
    def test_handle_sanskrit_terms(self):
        """Test Sanskrit term handling."""
        test_cases = [
            ("om mani padme hum", "oM mani padme hUM"),
            ("svaha", "swa ha"),
            ("regular tibetan text", "regular tibetan text"),
            ("om ah hum", "oM aH hUM"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.converter.handle_sanskrit_terms(input_text)
                self.assertEqual(result, expected)
    
    def test_handle_sanskrit_terms_empty(self):
        """Test Sanskrit handling with empty input."""
        self.assertEqual(self.converter.handle_sanskrit_terms(""), "")
        self.assertEqual(self.converter.handle_sanskrit_terms(None), "")
    
    def test_process_special_characters(self):
        """Test special character processing."""
        test_cases = [
            ("ka+ga", "kaga"),  # Stacking marker removal
            ("ka*ga", "kaga"),  # Emphasis marker removal
            ("ka~ga", "kaga"),  # Approximation marker removal
            ("ka?ga", "kaga"),  # Uncertainty marker removal
            ("ka  ga", "ka ga"),  # Whitespace normalization
            ("ka+ga*na", "kagana"),  # Multiple special chars
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.converter.process_special_characters(input_text)
                self.assertEqual(result, expected)
    
    def test_process_special_characters_empty(self):
        """Test special character processing with empty input."""
        self.assertEqual(self.converter.process_special_characters(""), "")
        self.assertEqual(self.converter.process_special_characters(None), "")
    
    def test_process_sanskrit_term(self):
        """Test individual Sanskrit term processing."""
        # Test mantra mappings
        self.assertEqual(self.converter._process_sanskrit_term("om", "om"), "oM")
        self.assertEqual(self.converter._process_sanskrit_term("hum", "hum"), "hUM")
        self.assertEqual(self.converter._process_sanskrit_term("svaha", "svaha"), "swa ha")
        
        # Test non-mantra terms (should return as-is)
        self.assertEqual(self.converter._process_sanskrit_term("buddha", "buddha"), "buddha")
        self.assertEqual(self.converter._process_sanskrit_term("dharma", "dharma"), "dharma")
    
    def test_post_process_tibetan(self):
        """Test Tibetan text post-processing."""
        test_cases = [
            ("ཀ  ག", "ཀ ག"),  # Whitespace normalization
            ("  ཀག  ", "ཀག"),  # Trim spaces
            ("", ""),  # Empty input
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.converter._post_process_tibetan(input_text)
                self.assertEqual(result, expected)
    
    def test_validate_conversion(self):
        """Test conversion validation."""
        # Valid conversions
        self.assertTrue(self.converter.validate_conversion("ka", "ཀ"))
        self.assertTrue(self.converter.validate_conversion("tshogs", "ཚོགས"))
        
        # Invalid conversions
        self.assertFalse(self.converter.validate_conversion("", ""))
        self.assertFalse(self.converter.validate_conversion("test", ""))
        self.assertFalse(self.converter.validate_conversion("", "ཀ"))
        self.assertFalse(self.converter.validate_conversion("longtext", "longtext"))  # No conversion
        self.assertFalse(self.converter.validate_conversion("test", "test"))  # No Tibetan chars
    
    def test_integration_full_conversion(self):
        """Test full conversion pipeline with realistic input."""
        # Test with a phrase that includes Sanskrit and special characters
        input_text = "om+mani*padme~hum"
        result = self.converter.convert_to_tibetan(input_text)
        
        # Should contain Tibetan Unicode characters
        self.assertTrue(any('\u0f00' <= char <= '\u0fff' for char in result))
        
        # Should be different from input (conversion occurred)
        self.assertNotEqual(result, input_text)
        
        # Validation should pass
        self.assertTrue(self.converter.validate_conversion(input_text, result))
    
    def test_validate_wylie_input(self):
        """Test Wylie input validation."""
        # Valid inputs
        valid_cases = [
            "ka ga nga",
            "tshogs",
            "om mani padme hum",
            "bde+ba chen*po"
        ]
        
        for case in valid_cases:
            with self.subTest(case=case):
                is_valid, error = self.converter.validate_wylie_input(case)
                self.assertTrue(is_valid, f"Should be valid: {case}")
                self.assertEqual(error, "")
        
        # Invalid inputs
        invalid_cases = [
            ("", "Empty input text"),
            ("   ", "Input contains only whitespace"),
            ("a" * 1001, "Input text is unusually long"),
        ]
        
        for case, expected_error in invalid_cases:
            with self.subTest(case=case):
                is_valid, error = self.converter.validate_wylie_input(case)
                self.assertFalse(is_valid)
                self.assertIn(expected_error.split()[0], error)  # Check first word of error
    
    def test_validate_tibetan_output(self):
        """Test Tibetan output validation."""
        # Valid outputs (with Tibetan Unicode)
        valid_cases = [
            "ཀ",
            "ཀ་ག་ང",
            "ཚོགས",
            "ཨོཾ་མནི་པདམེ་ཧཱུཾ"
        ]
        
        for case in valid_cases:
            with self.subTest(case=case):
                is_valid, error = self.converter.validate_tibetan_output(case)
                self.assertTrue(is_valid, f"Should be valid: {case}")
                self.assertEqual(error, "")
        
        # Invalid outputs
        invalid_cases = [
            ("", "Empty output text"),
            ("   ", "Output contains only whitespace"),
            ("no tibetan here", "Output does not contain Tibetan Unicode characters"),
            ("ཀ ERROR ག", "Output contains conversion artifact"),
            ("ཀ ??? ག", "Output contains conversion artifact"),
        ]
        
        for case, expected_error in invalid_cases:
            with self.subTest(case=case):
                is_valid, error = self.converter.validate_tibetan_output(case)
                self.assertFalse(is_valid)
                self.assertIn(expected_error.split()[0], error)  # Check first word of error
    
    def test_convert_with_fallback_success(self):
        """Test successful conversion with fallback method."""
        result, success, error = self.converter.convert_with_fallback("ka ga")
        
        self.assertTrue(success)
        self.assertEqual(error, "")
        self.assertTrue(any('\u0f00' <= char <= '\u0fff' for char in result))
    
    def test_convert_with_fallback_empty_input(self):
        """Test fallback with empty input."""
        result, success, error = self.converter.convert_with_fallback("")
        
        self.assertFalse(success)
        self.assertEqual(result, "")
        self.assertIn("Empty input", error)
    
    def test_convert_with_fallback_invalid_input(self):
        """Test fallback with invalid input."""
        # Test with whitespace-only input
        result, success, error = self.converter.convert_with_fallback("   ")
        
        self.assertFalse(success)
        self.assertEqual(result, "   ")  # Original fallback
        self.assertIn("Input validation failed", error)
    
    @patch('wylie_converter.logger')
    def test_convert_with_fallback_conversion_error(self, mock_logger):
        """Test fallback when conversion fails."""
        # Mock the entire convert_to_tibetan method to raise an exception
        with patch.object(self.converter, 'convert_to_tibetan', side_effect=Exception("Test error")):
            result, success, error = self.converter.convert_with_fallback("test", "original")
            
            self.assertFalse(success)
            self.assertEqual(result, "test")  # Original fallback
            self.assertIn("Conversion error", error)
            mock_logger.error.assert_called_once()
    
    def test_apply_fallback_strategies(self):
        """Test different fallback strategies."""
        test_text = "test wylie"
        
        # Original strategy
        result = self.converter._apply_fallback(test_text, "original")
        self.assertEqual(result, test_text)
        
        # Empty strategy
        result = self.converter._apply_fallback(test_text, "empty")
        self.assertEqual(result, "")
        
        # Phonetic strategy
        result = self.converter._apply_fallback(test_text, "phonetic")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        
        # Unknown strategy (should default to original)
        with patch('wylie_converter.logger') as mock_logger:
            result = self.converter._apply_fallback(test_text, "unknown")
            self.assertEqual(result, test_text)
            mock_logger.warning.assert_called_once()
    
    def test_simple_phonetic_fallback(self):
        """Test simple phonetic fallback conversion."""
        test_cases = [
            ("ka", "ka"),  # 'a' is preserved in the mapping
            ("kha", "kha"), # 'kh' maps to 'kh', 'a' maps to 'a'
            ("tshogs", "tshogs"),
            ("nga", "nga"), # 'ng' maps to 'ng', 'a' maps to 'a'
        ]
        
        for wylie, expected_phonetic in test_cases:
            with self.subTest(wylie=wylie):
                result = self.converter._simple_phonetic_fallback(wylie)
                self.assertEqual(result, expected_phonetic)
    
    def test_error_handling_edge_cases(self):
        """Test error handling with various edge cases."""
        edge_cases = [
            None,  # None input
            "",    # Empty string
            "   ", # Whitespace only
            "a" * 1001,  # Very long string
            "invalid\x00chars",  # Null characters
            "mixed བོད་ wylie",  # Mixed script
        ]
        
        for case in edge_cases:
            with self.subTest(case=repr(case)):
                # Should not raise exceptions
                try:
                    if case is not None:
                        result, success, error = self.converter.convert_with_fallback(case)
                        self.assertIsInstance(result, str)
                        self.assertIsInstance(success, bool)
                        self.assertIsInstance(error, str)
                except Exception as e:
                    self.fail(f"Unexpected exception for case {repr(case)}: {e}")


if __name__ == '__main__':
    unittest.main()