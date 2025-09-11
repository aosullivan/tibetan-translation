import unittest
from unittest.mock import Mock, MagicMock
from anthropic import Anthropic
from src.tibetan_translation_client import TibetanTranslationClient

class TestTibetanTranslationClient(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_anthropic = Mock(spec=Anthropic)
        self.client = TibetanTranslationClient(self.mock_anthropic)
    
    def test_initialization(self):
        """Test that TibetanTranslationClient initializes correctly."""
        self.assertIsInstance(self.client, TibetanTranslationClient)
        self.assertEqual(self.client.client, self.mock_anthropic)
        self.assertIsInstance(self.client.buddhist_terms, dict)
        self.assertIn('byang chub sems', self.client.buddhist_terms)
    
    def test_handle_technical_terms(self):
        """Test Buddhist technical term handling."""
        text = "The byang chub sems is important for sangs rgyas."
        result = self.client.handle_technical_terms(text)
        
        self.assertIn('[BUDDHIST_TERM:byang chub sems:', result)
        self.assertIn('[BUDDHIST_TERM:sangs rgyas:', result)
    
    def test_preserve_sacred_elements(self):
        """Test sacred element preservation."""
        text = "Recite oM ma Ni pad+me hUM and then hrI."
        result = self.client.preserve_sacred_elements(text)
        
        # The method replaces individual sacred patterns, not the full mantra
        self.assertIn('[SACRED:oM]', result)
        self.assertIn('[SACRED:hrI]', result)
    
    def test_vajrayana_translation_prompt_building(self):
        """Test that Vajrayana-specific prompts are built correctly."""
        context = "Previous context"
        wylie_text = "byang chub sems dpa'"
        tibetan_script = "བྱང་ཆུབ་སེམས་དཔའ།"
        
        prompt = self.client._build_vajrayana_translation_prompt(context, wylie_text, tibetan_script)
        
        self.assertIn("Vajrayana Buddhist", prompt)
        self.assertIn("TANTRIC TERMINOLOGY", prompt)  # All caps as in the actual prompt
        self.assertIn("Sanskrit terms", prompt)
        self.assertIn(wylie_text, prompt)
        self.assertIn(tibetan_script, prompt)
        self.assertIn(context, prompt)

if __name__ == '__main__':
    unittest.main()