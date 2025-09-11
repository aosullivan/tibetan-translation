"""
Unit tests for TibetanLineParser class.
"""

import unittest
import tempfile
import os
from src.tibetan_line_parser import TibetanLineParser, TibetanLine


class TestTibetanLineParser(unittest.TestCase):
    """Test cases for TibetanLineParser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = TibetanLineParser()
    
    def test_extract_wylie_content_basic(self):
        """Test basic Wylie content extraction."""
        # Test normal Wylie text
        result = self.parser.extract_wylie_content("bo dong paN chen phyogs las rnam rgyal")
        self.assertEqual(result, "bo dong paN chen phyogs las rnam rgyal")
        
        # Test with extra whitespace
        result = self.parser.extract_wylie_content("  bo dong   paN chen  ")
        self.assertEqual(result, "bo dong paN chen")
    
    def test_extract_wylie_content_with_markers(self):
        """Test Wylie extraction with structural markers."""
        # Test with parenthetical marker
        result = self.parser.extract_wylie_content("(kha)@#/[ ]/bo dong paN chen")
        self.assertEqual(result, "bo dong paN chen")
        
        # Test with bracket marker
        result = self.parser.extract_wylie_content("[ ]/dbyangs can ma sgrub par 'dod pas")
        self.assertEqual(result, "dbyangs can ma sgrub par 'dod pas")
        
        # Test with underscores
        result = self.parser.extract_wylie_content("_tshangs par spyad_")
        self.assertEqual(result, "tshangs par spyad")
    
    def test_is_structural_marker_detection(self):
        """Test structural marker detection."""
        # Test structural markers
        self.assertTrue(self.parser.is_structural_marker("(kha)@#"))
        self.assertTrue(self.parser.is_structural_marker("[ ]"))
        self.assertTrue(self.parser.is_structural_marker("_@#"))
        self.assertTrue(self.parser.is_structural_marker("//_@#"))
        self.assertTrue(self.parser.is_structural_marker("_!_"))
        
        # Test lines starting/ending with underscores
        self.assertTrue(self.parser.is_structural_marker("_some content_"))
        self.assertTrue(self.parser.is_structural_marker("_content"))
        
        # Test normal content
        self.assertFalse(self.parser.is_structural_marker("bo dong paN chen"))
        self.assertFalse(self.parser.is_structural_marker("dbyangs can ma"))
    
    def test_parse_file_with_sample_data(self):
        """Test file parsing with sample Tibetan data."""
        # Create temporary file with sample content
        sample_content = """(kha)@#/[ ]/bo dong paN chen phyogs las rnam rgyal/dbyangs can ma sgrub par 'dod pas/_tshangs par spyad_/khyad par rdzun mi smra/"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            f.write(sample_content)
            temp_file = f.name
        
        try:
            # Parse the file
            lines = self.parser.parse_file(temp_file)
            
            # Should have 6 non-empty lines after splitting on "/"
            self.assertEqual(len(lines), 6)
            
            # Check first line (structural marker)
            self.assertEqual(lines[0].original_content, "(kha)@#")
            self.assertTrue(lines[0].is_structural)
            self.assertEqual(lines[0].wylie_content, "")
            
            # Check second line (bracket marker)
            self.assertEqual(lines[1].original_content, "[ ]")
            self.assertTrue(lines[1].is_structural)
            
            # Check third line (normal content)
            self.assertEqual(lines[2].original_content, "bo dong paN chen phyogs las rnam rgyal")
            self.assertFalse(lines[2].is_structural)
            self.assertEqual(lines[2].wylie_content, "bo dong paN chen phyogs las rnam rgyal")
            
            # Check fourth line (normal content)
            self.assertEqual(lines[3].original_content, "dbyangs can ma sgrub par 'dod pas")
            self.assertFalse(lines[3].is_structural)
            
            # Check fifth line (content with underscores - structural)
            self.assertEqual(lines[4].original_content, "_tshangs par spyad_")
            self.assertTrue(lines[4].is_structural)
            self.assertEqual(lines[4].wylie_content, "tshangs par spyad")
            
            # Check sixth line (normal content)
            self.assertEqual(lines[5].original_content, "khyad par rdzun mi smra")
            self.assertFalse(lines[5].is_structural)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def test_extract_markers(self):
        """Test marker extraction functionality."""
        # Test line with multiple markers
        line = "(kha)@#/[ ]/some content/_@#"
        markers = self.parser._extract_markers(line)
        
        # Should find the structural markers
        self.assertIn("(kha)@#", markers)
        self.assertIn("[ ]", markers)
        self.assertIn("_@#", markers)
    
    def test_empty_and_whitespace_lines(self):
        """Test handling of empty and whitespace-only lines."""
        # Test empty content extraction
        self.assertEqual(self.parser.extract_wylie_content(""), "")
        self.assertEqual(self.parser.extract_wylie_content("   "), "")
        
        # Test structural marker detection for empty content
        self.assertFalse(self.parser.is_structural_marker(""))
        self.assertTrue(self.parser.is_structural_marker("   _   "))
    
    def test_complex_wylie_content(self):
        """Test extraction of complex Wylie content with special characters."""
        # Test content with apostrophes and special Wylie characters
        content = "dbyangs can ma'i sgrub thabs gegs sel rjes gnang"
        result = self.parser.extract_wylie_content(content)
        self.assertEqual(result, content)
        
        # Test content with plus signs (Sanskrit)
        content = "pad+ma dkar po"
        result = self.parser.extract_wylie_content(content)
        self.assertEqual(result, content)
        
        # Test content with asterisks
        content = "spang*/_tshangs par spyad*"
        result = self.parser.extract_wylie_content(content)
        self.assertEqual(result, "spang*/_tshangs par spyad*")


if __name__ == '__main__':
    unittest.main()