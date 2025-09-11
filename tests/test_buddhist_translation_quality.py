import unittest
from unittest.mock import Mock, MagicMock
from anthropic import Anthropic
from src.tibetan_translation_client import TibetanTranslationClient

class TestBuddhistTranslationQuality(unittest.TestCase):
    """Test Buddhist-specific translation prompts and quality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_anthropic = Mock(spec=Anthropic)
        self.client = TibetanTranslationClient(self.mock_anthropic)
    
    def test_content_type_identification(self):
        """Test identification of different Buddhist content types."""
        # Liturgical content
        liturgical_text = "sangs rgyas chos dang tshogs kyi mchod pa la"
        self.assertEqual(self.client.identify_content_type(liturgical_text), 'liturgical')
        
        # Tantric content
        tantric_text = "rgyud kyi dbang bskur ba'i cho ga"
        self.assertEqual(self.client.identify_content_type(tantric_text), 'tantric')
        
        # Practice instructions
        practice_text = "zhi gnas kyi bsgom pa'i gdams ngag"
        self.assertEqual(self.client.identify_content_type(practice_text), 'practice')
        
        # Commentary
        commentary_text = "shes rab kyi pha rol tu phyin pa'i 'grel pa"
        self.assertEqual(self.client.identify_content_type(commentary_text), 'commentary')
        
        # Philosophical
        philosophical_text = "stong pa nyid kyi tshigs bcad"
        self.assertEqual(self.client.identify_content_type(philosophical_text), 'philosophical')
    
    def test_specialized_prompt_additions(self):
        """Test that specialized prompts are generated for different content types."""
        liturgical_addition = self.client.get_specialized_prompt_additions('liturgical')
        self.assertIn('LITURGICAL TEXT GUIDELINES', liturgical_addition)
        self.assertIn('devotional', liturgical_addition)
        
        tantric_addition = self.client.get_specialized_prompt_additions('tantric')
        self.assertIn('TANTRIC TEXT GUIDELINES', tantric_addition)
        self.assertIn('empowerment', tantric_addition)
        
        practice_addition = self.client.get_specialized_prompt_additions('practice')
        self.assertIn('MEDITATION PRACTICE GUIDELINES', practice_addition)
        self.assertIn('meditation stages', practice_addition)
    
    def test_enhanced_sacred_element_preservation(self):
        """Test enhanced sacred element preservation with longer mantras."""
        text = "oM a ra pa ca na dhI and gate gate pAragate pArasamgate bodhi svAhA"
        result = self.client.preserve_sacred_elements(text)
        
        # Should preserve complete mantras
        self.assertIn('[SACRED:a ra pa ca na dhI]', result)
        self.assertIn('[SACRED:gate gate pAragate pArasamgate bodhi svAhA]', result)
    
    def test_enhanced_buddhist_terms_dictionary(self):
        """Test that the Buddhist terms dictionary includes comprehensive terminology."""
        terms = self.client.buddhist_terms
        
        # Core Buddhist concepts
        self.assertIn('byang chub sems', terms)
        self.assertIn('sangs rgyas', terms)
        self.assertIn('chos', terms)
        
        # Vajrayana specific
        self.assertIn('dbang bskur', terms)
        self.assertIn('sgrub thabs', terms)
        self.assertIn('yi dam', terms)
        
        # Meditation terms
        self.assertIn('bsgom pa', terms)
        self.assertIn('ting nge \'dzin', terms)
        self.assertIn('zhi gnas', terms)
        
        # Philosophical terms
        self.assertIn('stong pa nyid', terms)
        self.assertIn('rten \'brel', terms)
    
    def test_vajrayana_prompt_comprehensiveness(self):
        """Test that the Vajrayana prompt includes comprehensive Buddhist knowledge."""
        context = "Test context"
        wylie_text = "byang chub sems dpa' chen po"
        
        prompt = self.client._build_vajrayana_translation_prompt(context, wylie_text)
        
        # Check for comprehensive Buddhist knowledge areas
        self.assertIn("Madhyamaka", prompt)
        self.assertIn("Yogacara", prompt)
        self.assertIn("Anuttarayoga Tantra", prompt)
        self.assertIn("Nyingma, Kagyu, Sakya, Gelug", prompt)
        
        # Check for translation principles
        self.assertIn("DOCTRINAL ACCURACY", prompt)
        self.assertIn("TANTRIC TERMINOLOGY", prompt)
        self.assertIn("RITUAL AND LITURGICAL", prompt)
        self.assertIn("SANSKRIT PRESERVATION", prompt)
        
        # Check for examples
        self.assertIn("bodhisattva", prompt)
        self.assertIn("conventional truth", prompt)
    
    def test_technical_term_processing_with_context(self):
        """Test technical term processing includes contextual information."""
        text = "The byang chub sems is essential for sangs rgyas."
        result = self.client.handle_technical_terms(text)
        
        # Should include context information in the markers
        self.assertIn('[BUDDHIST_TERM:byang chub sems:bodhicitta', result)
        self.assertIn('[BUDDHIST_TERM:sangs rgyas:buddha', result)

if __name__ == '__main__':
    unittest.main()