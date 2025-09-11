"""
Unit tests for TibetanFileHandler class.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.tibetan_file_handler import TibetanFileHandler
from src.tibetan_line_parser import TibetanLine
from src.tibetan_output_formatter import ProcessedLine, ProcessingStatus


class TestTibetanFileHandler(unittest.TestCase):
    """Test cases for TibetanFileHandler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = TibetanFileHandler()
    
    def test_read_wylie_file(self):
        """Test reading and parsing a Wylie file."""
        # Create temporary file with sample content
        sample_content = "(kha)@#/[ ]/bo dong paN chen/dbyangs can ma/"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            f.write(sample_content)
            temp_file = f.name
        
        try:
            # Read the file
            lines = self.handler.read_wylie_file(temp_file)
            
            # Should have parsed lines
            self.assertGreater(len(lines), 0)
            
            # Check that we got TibetanLine objects
            for line in lines:
                self.assertIsInstance(line, TibetanLine)
            
            # Check specific content
            content_lines = [line for line in lines if not line.is_structural]
            self.assertGreater(len(content_lines), 0)
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_read_wylie_file_not_found(self):
        """Test handling of missing file."""
        with self.assertRaises(FileNotFoundError):
            self.handler.read_wylie_file("nonexistent_file.txt")
    
    def test_ensure_unicode_support(self):
        """Test Unicode content handling."""
        # Test normal content
        normal_text = "bo dong paN chen"
        result = self.handler.ensure_unicode_support(normal_text)
        self.assertEqual(result, normal_text)
        
        # Test empty content
        result = self.handler.ensure_unicode_support("")
        self.assertEqual(result, "")
        
        # Test None content
        result = self.handler.ensure_unicode_support(None)
        self.assertIsNone(result)
        
        # Test Tibetan Unicode content
        tibetan_text = "བོད་སྐད།"
        result = self.handler.ensure_unicode_support(tibetan_text)
        self.assertEqual(result, tibetan_text)
    
    def test_write_tibetan_output(self):
        """Test writing processed lines to output file."""
        # Create sample processed lines
        source_line1 = TibetanLine(
            line_number=1,
            original_content="bo dong paN chen",
            wylie_content="bo dong paN chen",
            is_structural=False,
            markers=[]
        )
        
        source_line2 = TibetanLine(
            line_number=2,
            original_content="(kha)@#",
            wylie_content="",
            is_structural=True,
            markers=["(kha)@#"]
        )
        
        processed_line1 = ProcessedLine(source_line1)
        processed_line1.tibetan_script = "བོད་སྐད།"
        processed_line1.phonetic_rendering = "bo dong pan chen"
        processed_line1.english_translation = "Tibetan language"
        processed_line1.processing_status = ProcessingStatus.SUCCESS
        
        processed_line2 = ProcessedLine(source_line2)
        processed_line2.processing_status = ProcessingStatus.SKIPPED
        
        processed_lines = [processed_line1, processed_line2]
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            temp_file = f.name
        
        try:
            self.handler.write_tibetan_output(temp_file, processed_lines)
            
            # Verify file was created and has content
            self.assertTrue(os.path.exists(temp_file))
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that content includes expected elements
            self.assertIn("Tibetan Text Processing Results", content)
            self.assertIn("bo dong paN chen", content)
            self.assertIn("བོད་སྐད།", content)
            self.assertIn("bo dong pan chen", content)
            self.assertIn("Tibetan language", content)
            self.assertIn("Processing Summary", content)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    @patch('src.tibetan_file_handler.cfg')
    def test_initialize_tibetan_output_files(self, mock_cfg):
        """Test initialization of output files."""
        # Mock configuration
        mock_cfg.OUTPUT_FILE = "test_output.md"
        mock_cfg.WYLIE_OUTPUT_FILE = "test_wylie.txt"
        mock_cfg.PHONETIC_OUTPUT_FILE = "test_phonetic.txt"
        mock_cfg.TRANSLATION_OUTPUT_FILE = "test_translation.txt"
        mock_cfg.UNICODE_ENCODING = "utf-8"
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update paths to use temp directory
            mock_cfg.OUTPUT_FILE = os.path.join(temp_dir, "test_output.md")
            mock_cfg.WYLIE_OUTPUT_FILE = os.path.join(temp_dir, "test_wylie.txt")
            mock_cfg.PHONETIC_OUTPUT_FILE = os.path.join(temp_dir, "test_phonetic.txt")
            mock_cfg.TRANSLATION_OUTPUT_FILE = os.path.join(temp_dir, "test_translation.txt")
            
            # Initialize files
            self.handler.initialize_tibetan_output_files()
            
            # Check that files were created
            self.assertTrue(os.path.exists(mock_cfg.OUTPUT_FILE))
            self.assertTrue(os.path.exists(mock_cfg.WYLIE_OUTPUT_FILE))
            self.assertTrue(os.path.exists(mock_cfg.PHONETIC_OUTPUT_FILE))
            self.assertTrue(os.path.exists(mock_cfg.TRANSLATION_OUTPUT_FILE))
            
            # Check content of main output file
            with open(mock_cfg.OUTPUT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("Tibetan Text Processing Output", content)
    
    def test_save_tibetan_progress(self):
        """Test saving processing progress."""
        # Create sample processed line
        source_line = TibetanLine(
            line_number=1,
            original_content="bo dong paN chen",
            wylie_content="bo dong paN chen",
            is_structural=False,
            markers=[]
        )
        
        processed_line = ProcessedLine(source_line)
        processed_line.processing_status = ProcessingStatus.SUCCESS
        
        processed_lines = [processed_line]
        
        # Mock the save_progress method to avoid file operations
        with patch.object(self.handler, 'save_progress') as mock_save:
            with patch.object(self.handler, 'write_tibetan_output') as mock_write:
                self.handler.save_tibetan_progress(processed_lines)
                
                # Verify methods were called
                mock_save.assert_called_once()
                
                # Check that progress was updated
                self.assertIn("line_1", self.handler.progress)
                self.assertTrue(self.handler.progress["line_1"])
    
    def test_processed_line_creation(self):
        """Test ProcessedLine object creation and initialization."""
        source_line = TibetanLine(
            line_number=1,
            original_content="test content",
            wylie_content="test content",
            is_structural=False,
            markers=[]
        )
        
        processed_line = ProcessedLine(source_line)
        
        # Check initial state
        self.assertEqual(processed_line.source_line, source_line)
        self.assertEqual(processed_line.tibetan_script, "")
        self.assertEqual(processed_line.phonetic_rendering, "")
        self.assertEqual(processed_line.english_translation, "")
        self.assertEqual(processed_line.processing_status, ProcessingStatus.SUCCESS)  # Default value
        self.assertIsNone(processed_line.error_message)


if __name__ == '__main__':
    unittest.main()