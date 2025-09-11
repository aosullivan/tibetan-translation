"""
Tests for TibetanOutputFormatter class.
Tests output format consistency, Unicode encoding, and summary report generation.
"""

import unittest
from unittest.mock import Mock, patch
from src.tibetan_output_formatter import TibetanOutputFormatter, ProcessedLine, ProcessingStatus
from src.tibetan_line_parser import TibetanLine


class TestTibetanOutputFormatter(unittest.TestCase):
    """Test cases for TibetanOutputFormatter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = TibetanOutputFormatter()
        
        # Create sample TibetanLine objects
        self.sample_line = TibetanLine(
            line_number=1,
            original_content="om mani padme hum/",
            wylie_content="om ma ni pad+me hUM",
            is_structural=False,
            markers=[]
        )
        
        self.structural_line = TibetanLine(
            line_number=2,
            original_content="(kha)@#/[ ]/",
            wylie_content="",
            is_structural=True,
            markers=["(kha)@#", "[ ]"]
        )
        
        # Create sample ProcessedLine objects
        self.processed_line = ProcessedLine(
            source_line=self.sample_line,
            tibetan_script="ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ",
            phonetic_rendering="om ma ni pad-me hung",
            english_translation="The jewel in the lotus",
            processing_status=ProcessingStatus.SUCCESS,
            error_message=None
        )
        
        self.failed_line = ProcessedLine(
            source_line=self.sample_line,
            tibetan_script="",
            phonetic_rendering="",
            english_translation="",
            processing_status=ProcessingStatus.FAILED,
            error_message="Wylie conversion failed"
        )
    
    def test_format_triple_output_success(self):
        """Test formatting a successfully processed line."""
        result = self.formatter.format_triple_output(self.processed_line)
        
        # Check that all expected sections are present
        self.assertIn("## Line 1", result)
        self.assertIn("**Original:** om mani padme hum/", result)
        self.assertIn("**Wylie:** om ma ni pad+me hUM", result)
        self.assertIn("**Tibetan:** ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ", result)
        self.assertIn("**Phonetic:** om ma ni pad-me hung", result)
        self.assertIn("**Translation:** The jewel in the lotus", result)
        self.assertIn("**Status:** ✅ success", result)
        self.assertIn("---", result)
    
    def test_format_triple_output_failed(self):
        """Test formatting a failed processed line."""
        result = self.formatter.format_triple_output(self.failed_line)
        
        # Check that error information is included
        self.assertIn("## Line 1", result)
        self.assertIn("**Original:** om mani padme hum/", result)
        self.assertIn("**Status:** ❌ failed", result)
        self.assertIn("**Error:** Wylie conversion failed", result)
    
    def test_format_triple_output_structural_preserved(self):
        """Test formatting structural markers when preservation is enabled."""
        # Create processed structural line
        processed_structural = ProcessedLine(
            source_line=self.structural_line,
            processing_status=ProcessingStatus.SKIPPED
        )
        
        # Test with preservation enabled (default)
        result = self.formatter.format_triple_output(processed_structural)
        self.assertIn("## Line 2", result)
        self.assertIn("**Markers:** (kha)@#, [ ]", result)
    
    @patch('src.tibetan_output_formatter.cfg.PRESERVE_STRUCTURAL_MARKERS', False)
    def test_format_triple_output_structural_skipped(self):
        """Test skipping structural markers when preservation is disabled."""
        # Create processed structural line
        processed_structural = ProcessedLine(
            source_line=self.structural_line,
            processing_status=ProcessingStatus.SKIPPED
        )
        
        # Create new formatter instance to pick up the mocked config
        formatter = TibetanOutputFormatter()
        result = formatter.format_triple_output(processed_structural)
        self.assertEqual(result, "")
    
    def test_ensure_unicode_encoding_valid(self):
        """Test Unicode encoding with valid Tibetan text."""
        tibetan_text = "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ"
        result = self.formatter.ensure_unicode_encoding(tibetan_text)
        self.assertEqual(result, tibetan_text)
    
    def test_ensure_unicode_encoding_empty(self):
        """Test Unicode encoding with empty string."""
        result = self.formatter.ensure_unicode_encoding("")
        self.assertEqual(result, "")
    
    def test_ensure_unicode_encoding_none(self):
        """Test Unicode encoding with None input."""
        result = self.formatter.ensure_unicode_encoding(None)
        self.assertIsNone(result)
    
    @patch('src.tibetan_output_formatter.logger')
    def test_ensure_unicode_encoding_error_handling(self, mock_logger):
        """Test Unicode encoding error handling."""
        # Create a string that will cause encoding issues when using a restrictive encoding
        problematic_text = "test\udcff"  # Invalid surrogate
        
        # Test with a restrictive encoding that will cause issues
        original_encoding = self.formatter.unicode_encoding
        self.formatter.unicode_encoding = 'ascii'  # This will cause encoding errors
        
        try:
            result = self.formatter.ensure_unicode_encoding(problematic_text)
            
            # Should log a warning and return fallback
            mock_logger.warning.assert_called()
            self.assertIsInstance(result, str)
        finally:
            # Restore original encoding
            self.formatter.unicode_encoding = original_encoding
    
    def test_generate_summary_report_basic(self):
        """Test basic summary report generation."""
        processed_lines = [self.processed_line, self.failed_line]
        
        result = self.formatter.generate_summary_report(processed_lines)
        
        # Check report structure
        self.assertIn("# Processing Summary Report", result)
        self.assertIn("## Overall Statistics", result)
        self.assertIn("## Content Analysis", result)
        self.assertIn("## Processing Details", result)
        self.assertIn("## Quality Metrics", result)
        
        # Check statistics
        self.assertIn("**Total lines processed:** 2", result)
        self.assertIn("**Successfully completed:** 1 (50.0%)", result)
        self.assertIn("**Failed:** 1 (50.0%)", result)
    
    def test_generate_summary_report_empty(self):
        """Test summary report with empty input."""
        result = self.formatter.generate_summary_report([])
        
        self.assertIn("# Processing Summary Report", result)
        self.assertIn("**Total lines processed:** 0", result)
    
    def test_generate_summary_report_with_errors(self):
        """Test summary report includes error analysis."""
        # Create lines with different error types
        unicode_error_line = ProcessedLine(
            source_line=self.sample_line,
            processing_status=ProcessingStatus.FAILED,
            error_message="Unicode encoding error"
        )
        
        wylie_error_line = ProcessedLine(
            source_line=self.sample_line,
            processing_status=ProcessingStatus.FAILED,
            error_message="Wylie conversion failed"
        )
        
        processed_lines = [unicode_error_line, wylie_error_line]
        result = self.formatter.generate_summary_report(processed_lines)
        
        # Should include error analysis
        self.assertIn("## Issues and Errors", result)
        self.assertIn("Unicode/Encoding errors", result)
        self.assertIn("Wylie conversion errors", result)
        self.assertIn("### Failed/Partial Lines", result)
    
    def test_calculate_statistics_comprehensive(self):
        """Test comprehensive statistics calculation."""
        # Create diverse set of processed lines
        success_line = ProcessedLine(
            source_line=self.sample_line,
            tibetan_script="ཨོཾ",
            phonetic_rendering="om",
            english_translation="om",
            processing_status=ProcessingStatus.SUCCESS
        )
        
        partial_line = ProcessedLine(
            source_line=self.sample_line,
            tibetan_script="ཨོཾ",
            processing_status=ProcessingStatus.PARTIAL_SUCCESS
        )
        
        skipped_line = ProcessedLine(
            source_line=self.structural_line,
            processing_status=ProcessingStatus.SKIPPED
        )
        
        processed_lines = [success_line, partial_line, self.failed_line, skipped_line]
        stats = self.formatter._calculate_statistics(processed_lines)
        
        # Verify statistics
        self.assertEqual(stats['total_lines'], 4)
        self.assertEqual(stats['success_count'], 1)
        self.assertEqual(stats['partial_count'], 1)
        self.assertEqual(stats['failed_count'], 1)
        self.assertEqual(stats['skipped_count'], 1)
        self.assertEqual(stats['structural_count'], 1)
        self.assertEqual(stats['content_count'], 3)
        self.assertEqual(stats['tibetan_script_count'], 2)
        self.assertEqual(stats['phonetic_count'], 1)
        self.assertEqual(stats['translation_count'], 1)
    
    def test_summarize_errors(self):
        """Test error summarization by type."""
        lines_with_errors = [
            ProcessedLine(
                source_line=self.sample_line,
                processing_status=ProcessingStatus.FAILED,
                error_message="Unicode encoding failed"
            ),
            ProcessedLine(
                source_line=self.sample_line,
                processing_status=ProcessingStatus.FAILED,
                error_message="Wylie conversion error"
            ),
            ProcessedLine(
                source_line=self.sample_line,
                processing_status=ProcessingStatus.FAILED,
                error_message="API translation timeout"
            ),
            ProcessedLine(
                source_line=self.sample_line,
                processing_status=ProcessingStatus.FAILED,
                error_message="Unknown error occurred"
            )
        ]
        
        error_summary = self.formatter._summarize_errors(lines_with_errors)
        
        # Check error categorization
        self.assertEqual(error_summary.get("Unicode/Encoding errors", 0), 1)
        self.assertEqual(error_summary.get("Wylie conversion errors", 0), 1)
        self.assertEqual(error_summary.get("Translation API errors", 0), 1)
        self.assertEqual(error_summary.get("Other errors", 0), 1)
    
    def test_get_status_icon(self):
        """Test status icon mapping."""
        self.assertEqual(self.formatter._get_status_icon(ProcessingStatus.SUCCESS), "✅")
        self.assertEqual(self.formatter._get_status_icon(ProcessingStatus.PARTIAL_SUCCESS), "⚠️")
        self.assertEqual(self.formatter._get_status_icon(ProcessingStatus.FAILED), "❌")
        self.assertEqual(self.formatter._get_status_icon(ProcessingStatus.SKIPPED), "⏭️")
    
    def test_generate_structured_output(self):
        """Test complete structured output generation."""
        processed_lines = [self.processed_line, self.failed_line]
        
        result = self.formatter.generate_structured_output(processed_lines, "Test Document")
        
        # Check document structure
        self.assertIn("# Test Document", result)
        self.assertIn("# Table of Contents", result)
        self.assertIn("# Processing Overview", result)
        self.assertIn("# Main Content", result)
        self.assertIn("# Failed Processing", result)
        self.assertIn("# Processing Summary Report", result)
        self.assertIn("# Line Correspondence Index", result)
    
    def test_maintain_line_correspondence(self):
        """Test line correspondence tracking."""
        self.formatter.maintain_line_correspondence(1, 5, ProcessingStatus.SUCCESS)
        self.formatter.maintain_line_correspondence(2, 6, ProcessingStatus.FAILED)
        
        correspondence = self.formatter.line_correspondence
        self.assertEqual(correspondence[1]['output_line'], 5)
        self.assertEqual(correspondence[1]['status'], ProcessingStatus.SUCCESS)
        self.assertEqual(correspondence[2]['output_line'], 6)
        self.assertEqual(correspondence[2]['status'], ProcessingStatus.FAILED)
    
    def test_organize_content_by_sections(self):
        """Test content organization into sections."""
        # Create structural line
        structural_processed = ProcessedLine(
            source_line=self.structural_line,
            processing_status=ProcessingStatus.SKIPPED
        )
        
        processed_lines = [self.processed_line, self.failed_line, structural_processed]
        sections = self.formatter._organize_content_by_sections(processed_lines)
        
        # Check section organization
        self.assertIn("Main Content", sections)
        self.assertIn("Failed Processing", sections)
        self.assertIn("Structural Elements", sections)
        
        self.assertEqual(len(sections["Main Content"]), 1)
        self.assertEqual(len(sections["Failed Processing"]), 1)
        self.assertEqual(len(sections["Structural Elements"]), 1)
    
    def test_generate_document_header(self):
        """Test document header generation."""
        processed_lines = [self.processed_line, self.failed_line]
        
        header = self.formatter._generate_document_header("Test Title", processed_lines)
        
        self.assertIn("# Test Title", header)
        self.assertIn("Total lines processed: 2", header)
        self.assertIn("Successfully processed: 1", header)
        self.assertIn("Success rate: 50.0%", header)
        self.assertIn("Unicode encoding:", header)
    
    def test_generate_table_of_contents(self):
        """Test table of contents generation."""
        processed_lines = [self.processed_line, self.failed_line]
        
        toc = self.formatter._generate_table_of_contents(processed_lines)
        
        self.assertIn("# Table of Contents", toc)
        self.assertIn("Processing Overview", toc)
        self.assertIn("Content Sections", toc)
        self.assertIn("Processing Summary", toc)
        self.assertIn("Line Correspondence Index", toc)
    
    def test_generate_processing_overview(self):
        """Test processing overview generation."""
        processed_lines = [self.processed_line, self.failed_line]
        
        overview = self.formatter._generate_processing_overview(processed_lines)
        
        self.assertIn("# Processing Overview", overview)
        self.assertIn("## Status Distribution", overview)
        self.assertIn("## Content Breakdown", overview)
        self.assertIn("| Status | Count | Percentage |", overview)
        self.assertIn("✅ Success | 1 | 50.0%", overview)
        self.assertIn("❌ Failed | 1 | 50.0%", overview)
    
    def test_generate_content_section(self):
        """Test content section generation."""
        section_lines = [self.processed_line]
        
        section = self.formatter._generate_content_section("Test Section", section_lines)
        
        self.assertIn("# Test Section", section)
        self.assertIn("*1 lines in this section*", section)
        self.assertIn("## Line 1", section)  # From the formatted line content
    
    def test_generate_line_correspondence_index(self):
        """Test line correspondence index generation."""
        # Add some correspondence data
        self.formatter.maintain_line_correspondence(1, 5, ProcessingStatus.SUCCESS)
        self.formatter.maintain_line_correspondence(2, 6, ProcessingStatus.FAILED)
        
        index = self.formatter._generate_line_correspondence_index()
        
        self.assertIn("# Line Correspondence Index", index)
        self.assertIn("| Input Line | Output Section | Status | Notes |", index)
        self.assertIn("| 1 | Line 5 | ✅ success", index)
        self.assertIn("| 2 | Line 6 | ❌ failed", index)
    
    def test_add_processing_status_indicators(self):
        """Test adding status indicators to content."""
        content = "# Test Header\n\nSome content here"
        
        result = self.formatter.add_processing_status_indicators(
            content, ProcessingStatus.SUCCESS, "All good"
        )
        
        self.assertIn("✅ **SUCCESS** - All good", result)
        self.assertIn("# Test Header", result)
        self.assertIn("Some content here", result)
    
    def test_create_section_headers(self):
        """Test section header creation."""
        header = self.formatter.create_section_headers("Test Section", 5, "content")
        
        self.assertIn("# 📄 Test Section", header)
        self.assertIn("*Section contains 5 lines*", header)
        
        # Test error section
        error_header = self.formatter.create_section_headers("Errors", 2, "errors")
        self.assertIn("# ⚠️ Errors", error_header)
        self.assertIn("⚠️ **Note:** This section contains lines that encountered processing errors.", error_header)


if __name__ == '__main__':
    unittest.main()