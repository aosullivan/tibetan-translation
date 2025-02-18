import pytest
from unittest.mock import Mock, create_autospec, MagicMock
from anthropic import Anthropic
from translation_client import TranslationClient

class TestTranslationClient:
    @pytest.fixture
    def mock_anthropic(self):
        mock = MagicMock(spec=Anthropic)
        # Set up the nested structure
        mock.messages = MagicMock()
        mock.messages.create = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_anthropic):
        return TranslationClient(mock_anthropic)

    def test_untranslated_fragment_handling(self, mock_anthropic, client):
        # Setup mock responses
        # First chunk response with untranslated fragment
        first_response = Mock()
        first_response.content = [Mock(text="This is the first part of UNTRANSLATED:རྒྱུད་འདི་")]
        
        # Second chunk response completing the sentence
        second_response = Mock()
        second_response.content = [Mock(text="the sacred tantra.")]
        
        mock_anthropic.messages.create.side_effect = [first_response, second_response]

        # Test first chunk
        translation1, untranslated1 = client.translate_chunk("This is the first part of རྒྱུད་འདི་ འདི་ནི་གསང་སྔགས་ཡིན་")
        
        assert translation1 == "This is the first part of"
        assert untranslated1 == "རྒྱུད་འདི་"
        assert client.untranslated_fragment == "རྒྱུད་འདི་"

        # Test second chunk
        translation2, untranslated2 = client.translate_chunk("འདི་ནི་གསང་སྔགས་ཡིན་")
        
        # Verify the fragment was carried forward
        calls = mock_anthropic.messages.create.call_args_list
        assert len(calls) == 2
        
        # Check that second chunk included the untranslated fragment
        second_call_text = calls[1][1]['messages'][0]['content']
        assert "རྒྱུད་འདི་" in second_call_text
        
        assert translation2 == "the sacred tantra."
        assert untranslated2 == ""
