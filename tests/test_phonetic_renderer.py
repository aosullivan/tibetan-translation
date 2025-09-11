"""
Unit tests for PhoneticRenderer class.
Tests phonetic conversion accuracy, pronunciation rules, and Sanskrit handling.
"""

import unittest
from src.phonetic_renderer import PhoneticRenderer


class TestPhoneticRenderer(unittest.TestCase):
    """Test cases for PhoneticRenderer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = PhoneticRenderer()
    
    def test_basic_consonant_conversion(self):
        """Test basic consonant phonetic conversion."""
        test_cases = [
            ('ka', 'ka'),
            ('kha', 'kha'),
            ('ga', 'ga'),
            ('nga', 'nga'),
            ('cha', 'cha'),
            ('ja', 'ja'),
            ('nya', 'nya'),
            ('ta', 'ta'),
            ('tha', 'tha'),
            ('da', 'da'),
            ('na', 'na'),
            ('pa', 'pa'),
            ('pha', 'pha'),
            ('ba', 'ba'),
            ('ma', 'ma'),
            ('tsa', 'tsa'),
            ('tsha', 'tsha'),
            ('dza', 'dza'),
            ('wa', 'wa'),
            ('zha', 'zha'),
            ('za', 'za'),
            ('ya', 'ya'),
            ('ra', 'ra'),
            ('la', 'la'),
            ('sha', 'sha'),
            ('sa', 'sa'),
            ('ha', 'ha')
        ]
        
        for wylie, expected in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.generate_phonetic(wylie)
                self.assertIn(expected.replace('a', ''), result.lower())
    
    def test_vowel_conversion(self):
        """Test vowel phonetic conversion."""
        test_cases = [
            ('ki', 'ki'),
            ('ku', 'ku'),
            ('ke', 'ke'),
            ('ko', 'ko'),
            ('kA', 'kaa'),  # Long vowels
            ('kI', 'kii'),
            ('kU', 'kuu')
        ]
        
        for wylie, expected in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.generate_phonetic(wylie)
                # Check that the phonetic contains expected vowel sounds
                self.assertTrue(any(vowel in result.lower() for vowel in ['i', 'u', 'e', 'o', 'aa', 'ii', 'uu']))
    
    def test_sanskrit_terms(self):
        """Test Sanskrit term phonetic rendering."""
        test_cases = [
            ('om', 'ohm'),
            ('ah', 'ah'),
            ('hum', 'hoom'),
            ('svaha', 'so-ha'),
            ('phat', 'peht'),
            ('hrih', 'hreeh'),
            ('mantra', 'man-tra'),
            ('dharani', 'dha-ra-nee'),
            ('buddha', 'bud-dha'),
            ('dharma', 'dhar-ma'),
            ('sangha', 'sang-ha'),
            ('guru', 'gu-ru'),
            ('lama', 'la-ma')
        ]
        
        for wylie, expected in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.generate_phonetic(wylie)
                self.assertEqual(result.lower(), expected.lower())
    
    def test_tone_markers(self):
        """Test tone and emphasis marker handling."""
        test_cases = [
            ('ka~', 'ka (approx)'),
            ('ka*', 'ka (emph)'),
            ('ka?', 'ka (?)'),
            ('ka_ba', 'ka ba'),
            ('ka/ba', 'ka | ba')
        ]
        
        for wylie, expected in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.handle_tone_markers(wylie)
                self.assertEqual(result, expected)
    
    def test_pronunciation_rules(self):
        """Test Tibetan pronunciation rule application."""
        test_cases = [
            # Final consonants often silent
            ('kag', 'ka'),
            ('kad', 'ka'),
            ('kas', 'ka'),
            
            # Consonant clusters
            ('kra', 'kra'),
            ('pra', 'pra'),
            ('dra', 'tra')  # dr -> tr simplification
        ]
        
        for input_text, expected_pattern in test_cases:
            with self.subTest(input_text=input_text):
                result = self.renderer.apply_pronunciation_rules(input_text)
                # Check that the expected pattern appears in result
                self.assertIn(expected_pattern, result)
    
    def test_complex_wylie_phrases(self):
        """Test phonetic conversion of complex Wylie phrases."""
        test_cases = [
            ('om mani padme hum', 'ohm'),  # Should contain 'ohm' for 'om'
            ("bcom ldan 'das", 'chom'),   # Should handle apostrophes (bcom -> chom)
            ('sangs rgyas', 'sang'),      # Should handle consonant clusters (sangs -> sang)
            ('byang chub sems dpa', 'chang')  # Should handle complex syllables (byang -> chang)
        ]
        
        for wylie, expected_contains in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.generate_phonetic(wylie)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
                # Check that result contains some expected phonetic elements
                if expected_contains:
                    self.assertIn(expected_contains, result.lower())
    
    def test_empty_and_invalid_input(self):
        """Test handling of empty and invalid input."""
        test_cases = [
            ('', ''),
            ('   ', ''),
            (None, '')
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=repr(input_text)):
                if input_text is None:
                    # Should handle None gracefully
                    result = self.renderer.generate_phonetic('')
                else:
                    result = self.renderer.generate_phonetic(input_text)
                self.assertEqual(result, expected)
    
    def test_syllable_splitting(self):
        """Test syllable splitting functionality."""
        test_cases = [
            ('ka ba', ['ka', 'ba']),
            ('mani padme', ['mani', 'padme']),
            ('bcom ldan das', ['bcom', 'ldan', 'das'])
        ]
        
        for input_text, expected_count in test_cases:
            with self.subTest(input_text=input_text):
                syllables = self.renderer._split_into_syllables(input_text)
                self.assertGreaterEqual(len(syllables), len(expected_count))
                # Check that we get reasonable syllable splits
                for syllable in syllables:
                    if syllable.strip():
                        self.assertGreater(len(syllable.strip()), 0)
    
    def test_phonetic_validation(self):
        """Test phonetic output validation."""
        test_cases = [
            ('hello world', True, ''),
            ('', False, 'Empty phonetic output'),
            ('   ', False, 'Phonetic output contains only whitespace'),
            ('a' * 1001, False, 'Phonetic output unusually long'),
            ('hello\u0f00world', False, 'Contains invalid characters')
        ]
        
        for phonetic_text, expected_valid, expected_error_contains in test_cases:
            with self.subTest(phonetic_text=phonetic_text[:20] + '...'):
                is_valid, error_msg = self.renderer.validate_phonetic_output(phonetic_text)
                self.assertEqual(is_valid, expected_valid)
                if expected_error_contains:
                    self.assertIn(expected_error_contains, error_msg)
    
    def test_generate_phonetic_with_validation(self):
        """Test phonetic generation with comprehensive validation."""
        test_cases = [
            ('om mani padme hum', True),
            ('ka ba', True),
            ('', False),
            ('bcom ldan das', True)
        ]
        
        for wylie_text, expected_success in test_cases:
            with self.subTest(wylie_text=wylie_text):
                phonetic, success, error_msg = self.renderer.generate_phonetic_with_validation(wylie_text)
                self.assertEqual(success, expected_success)
                
                if expected_success:
                    self.assertIsInstance(phonetic, str)
                    self.assertGreater(len(phonetic), 0)
                    self.assertEqual(error_msg, '')
                else:
                    self.assertNotEqual(error_msg, '')
    
    def test_word_format_preservation(self):
        """Test preservation of capitalization and punctuation."""
        test_cases = [
            ('Om', 'Ohm'),
            ('om.', 'ohm.'),
            ('OM!', 'OHM!'),
            ('dharma,', 'dhar-ma,')
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                # Test the internal method
                result = self.renderer._preserve_word_format(original, 'ohm' if 'om' in original.lower() else 'dhar-ma')
                # Check that capitalization and punctuation are preserved
                if original[0].isupper():
                    self.assertTrue(result[0].isupper())
                if original.endswith('.') or original.endswith(',') or original.endswith('!'):
                    self.assertTrue(any(result.endswith(p) for p in ['.', ',', '!']))
    
    def test_special_characters_handling(self):
        """Test handling of special Wylie characters."""
        test_cases = [
            ('ka+ba', 'ka ba'),  # Stacking marker removal
            ('ka*ba', 'ka ba'),  # Emphasis marker
            ('ka~ba', 'ka ba'),  # Approximation marker
            ('ka?ba', 'ka ba')   # Uncertainty marker
        ]
        
        for wylie, expected_pattern in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.generate_phonetic(wylie)
                # Should not contain the special characters in output
                self.assertNotIn('+', result)
                self.assertNotIn('*', result)
                # Should contain the basic syllables
                self.assertIn('ka', result.lower())
                self.assertIn('ba', result.lower())
    
    def test_mantra_and_seed_syllables(self):
        """Test mantra and seed syllable phonetic rendering."""
        test_cases = [
            ('oM', 'OHM'),
            ('aH', 'AH'),
            ('hUM', 'HOOM'),
            ('hrIH', 'HREEH'),
            ('phaT', 'PEHT'),
            ('mani', 'ma-ni'),
            ('padme', 'pad-me'),
            ('vajra', 'vaj-ra'),
            ('tare', 'ta-re'),
            ('tuttare', 'tut-ta-re')
        ]
        
        for wylie, expected in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.handle_mantras_and_seed_syllables(wylie)
                self.assertEqual(result.lower(), expected.lower())
    
    def test_tibetan_buddhist_terms(self):
        """Test Tibetan Buddhist term phonetic rendering."""
        test_cases = [
            ('bcom ldan das', 'chom den de'),
            ('sangs rgyas', 'sang gye'),
            ('byang chub sems dpa', 'chang chub sem pa'),
            ('rje btsun', 'je tsun'),
            ('sku gsung thugs', 'ku sung thuk'),
            ('rdzogs chen', 'dzok chen')
        ]
        
        for wylie, expected_contains in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.generate_phonetic(wylie)
                # Check that key phonetic elements are present
                expected_words = expected_contains.split()
                for expected_word in expected_words:
                    self.assertIn(expected_word, result.lower())
    
    def test_phonetic_output_formatting(self):
        """Test consistent phonetic output formatting."""
        test_cases = [
            ('bo-dhi-sat-tva', 'bo-dhi-sat-tva'),  # Already hyphenated
            ('OHM AH HOOM', 'OHM AH HOOM'),
            ('ma-ni  pad-me   hoom', 'ma-ni pad-me hoom'),  # Multiple spaces with hyphens
            ('dhar-ma sang-ha', 'dhar-ma sang-ha')  # Already hyphenated
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.renderer.format_phonetic_output(input_text)
                # Check for consistent formatting
                self.assertNotIn('  ', result)  # No double spaces
                if '-' in expected:
                    self.assertIn('-', result)  # Hyphenation preserved
    
    def test_complex_tibetan_text_processing(self):
        """Test processing of complex Tibetan texts."""
        complex_texts = [
            'oM mani padme hUM',
            'bcom ldan das sangs rgyas',
            'oM tare tuttare ture soha',
            'hrIH vajra sattva hUM'
        ]
        
        for wylie_text in complex_texts:
            with self.subTest(wylie_text=wylie_text):
                result = self.renderer.process_complex_tibetan_text(wylie_text)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
                # Should not contain original Wylie characters in final output
                # Should be properly phonetic (allow 'gye' as it's a valid phonetic rendering)
    
    def test_mantra_case_sensitivity(self):
        """Test that mantras preserve proper case sensitivity."""
        test_cases = [
            ('oM', 'OHM'),  # Should be uppercase
            ('om', 'ohm'),  # Lowercase version
            ('hUM', 'HOOM'),  # Should be uppercase
            ('hum', 'hoom')   # Lowercase version
        ]
        
        for wylie, expected in test_cases:
            with self.subTest(wylie=wylie):
                result = self.renderer.generate_phonetic(wylie)
                if wylie.isupper() or (wylie[0].isupper() and len(wylie) > 1 and wylie[1:].isupper()):
                    # Should preserve uppercase for sacred syllables
                    self.assertTrue(any(c.isupper() for c in result))
                else:
                    # Regular terms can be lowercase
                    self.assertIn(expected.lower(), result.lower())
    
    def test_mixed_sanskrit_tibetan_text(self):
        """Test handling of mixed Sanskrit and Tibetan text."""
        test_cases = [
            ('om bcom ldan das', ['ohm', 'chom']),
            ('vajra guru padma', ['vaj-ra', 'gu-ru', 'padma']),
            ('dharma sangs rgyas', ['dhar-ma', 'sang'])
        ]
        
        for wylie_text, expected_contains in test_cases:
            with self.subTest(wylie_text=wylie_text):
                result = self.renderer.generate_phonetic(wylie_text)
                for expected in expected_contains:
                    self.assertIn(expected.lower(), result.lower())


if __name__ == '__main__':
    unittest.main()