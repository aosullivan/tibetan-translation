"""
Tibetan Output Formatter for generating structured three-column output.
Handles Unicode encoding and creates formatted reports for Tibetan text processing.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from .config import TibetanConfig as cfg
from .tibetan_models import ProcessedLine, ProcessingStatus, TibetanLine

logger = logging.getLogger(__name__)


class TibetanOutputFormatter:
    """
    Formatter for creating structured three-column Tibetan text output.
    Handles Unicode encoding and generates processing reports.
    """
    
    def __init__(self):
        self.unicode_encoding = cfg.UNICODE_ENCODING
        self.preserve_structural = cfg.PRESERVE_STRUCTURAL_MARKERS
        self.line_correspondence = {}  # Track input-output line mapping
    
    def format_triple_output(self, processed_line: ProcessedLine) -> str:
        """
        Format a processed line into three-column layout.
        
        Args:
            processed_line: ProcessedLine object with all conversions
            
        Returns:
            Formatted string with three-column layout
        """
        try:
            source = processed_line.source_line
            
            # Skip structural markers unless configured to preserve them
            if source.is_structural and not self.preserve_structural:
                return ""
            
            # Build the formatted output
            output_lines = []
            
            # Tibetan script (if converted)
            if processed_line.tibetan_script:
                tibetan_safe = self.ensure_unicode_encoding(processed_line.tibetan_script)
                output_lines.append(f"{tibetan_safe}")
            
            # Wylie transliteration
            output_lines.append(f"{source.wylie_content}")
            
            # English translation (if available)
            if processed_line.english_translation:
                translation_safe = self.ensure_unicode_encoding(processed_line.english_translation)
                output_lines.append(f"{translation_safe}")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error formatting line {processed_line.source_line.line_number}: {str(e)}")
            return ""
    
    def ensure_unicode_encoding(self, content: str) -> str:
        """
        Ensure content is properly encoded for Unicode Tibetan display.
        
        Args:
            content: Text content to encode safely
            
        Returns:
            Unicode-safe content
        """
        if not content:
            return content
        
        try:
            # Test encoding/decoding to ensure Unicode safety
            encoded = content.encode(self.unicode_encoding)
            decoded = encoded.decode(self.unicode_encoding)
            return decoded
        except UnicodeError as e:
            logger.warning(f"Unicode encoding issue: {str(e)}")
            # Fallback: replace problematic characters
            safe_content = content.encode(self.unicode_encoding, errors='replace').decode(self.unicode_encoding)
            return safe_content
    
    def generate_summary_report(self, processed_lines: List[ProcessedLine]) -> str:
        """
        Generate a comprehensive summary report of processing statistics.
        
        Args:
            processed_lines: List of all processed lines
            
        Returns:
            Formatted summary report string
        """
        # No longer needed for simplified output
        return ""
    
    def _calculate_statistics(self, processed_lines: List[ProcessedLine]) -> Dict[str, Any]:
        """Calculate comprehensive statistics from processed lines."""
        total_lines = len(processed_lines)
        
        if total_lines == 0:
            return {key: 0 for key in [
                'total_lines', 'success_count', 'partial_count', 'failed_count', 'skipped_count',
                'success_percentage', 'partial_percentage', 'failed_percentage', 'skipped_percentage',
                'structural_count', 'content_count', 'empty_count', 'tibetan_script_count',
                'translation_count', 'avg_wylie_length', 'avg_translation_length',
                'conversion_success_rate', 'translation_coverage'
            ]}
        
        # Count by status
        success_count = sum(1 for line in processed_lines if line.processing_status == ProcessingStatus.SUCCESS)
        partial_count = sum(1 for line in processed_lines if line.processing_status == ProcessingStatus.PARTIAL_SUCCESS)
        failed_count = sum(1 for line in processed_lines if line.processing_status == ProcessingStatus.FAILED)
        skipped_count = sum(1 for line in processed_lines if line.processing_status == ProcessingStatus.SKIPPED)
        
        # Content analysis
        structural_count = sum(1 for line in processed_lines if line.source_line.is_structural)
        content_count = total_lines - structural_count
        empty_count = sum(1 for line in processed_lines if not line.source_line.wylie_content.strip())
        
        # Conversion counts
        tibetan_script_count = sum(1 for line in processed_lines if line.tibetan_script)
        translation_count = sum(1 for line in processed_lines if line.english_translation)
        
        # Length calculations
        wylie_lengths = [len(line.source_line.wylie_content) for line in processed_lines if line.source_line.wylie_content]
        translation_lengths = [len(line.english_translation) for line in processed_lines if line.english_translation]
        
        avg_wylie_length = sum(wylie_lengths) / len(wylie_lengths) if wylie_lengths else 0
        avg_translation_length = sum(translation_lengths) / len(translation_lengths) if translation_lengths else 0
        
        return {
            'total_lines': total_lines,
            'success_count': success_count,
            'partial_count': partial_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'success_percentage': (success_count / total_lines) * 100,
            'partial_percentage': (partial_count / total_lines) * 100,
            'failed_percentage': (failed_count / total_lines) * 100,
            'skipped_percentage': (skipped_count / total_lines) * 100,
            'structural_count': structural_count,
            'content_count': content_count,
            'empty_count': empty_count,
            'tibetan_script_count': tibetan_script_count,
            'translation_count': translation_count,
            'avg_wylie_length': avg_wylie_length,
            'avg_translation_length': avg_translation_length,
            'conversion_success_rate': ((success_count + partial_count) / total_lines) * 100,
            'translation_coverage': (translation_count / content_count) * 100 if content_count > 0 else 0,
        }
    
    def _summarize_errors(self, processed_lines: List[ProcessedLine]) -> Dict[str, int]:
        """Summarize error types and their frequencies."""
        error_summary = {}
        
        for line in processed_lines:
            if line.error_message:
                # Categorize errors by common patterns
                error_msg = line.error_message.lower()
                
                if 'unicode' in error_msg or 'encoding' in error_msg:
                    error_type = "Unicode/Encoding errors"
                elif 'wylie' in error_msg or 'conversion' in error_msg:
                    error_type = "Wylie conversion errors"
                elif 'api' in error_msg or 'translation' in error_msg:
                    error_type = "Translation API errors"
                else:
                    error_type = "Other errors"
                
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        return error_summary
    
    def _get_status_icon(self, status: ProcessingStatus) -> str:
        """
        Get an icon/symbol for the processing status."""
        icons = {
            ProcessingStatus.SUCCESS: "✅",
            ProcessingStatus.PARTIAL_SUCCESS: "⚠️",
            ProcessingStatus.FAILED: "❌",
            ProcessingStatus.SKIPPED: "⏭️"
        }
        return icons.get(status, "❓")
    
    def generate_structured_output(self, processed_lines: List[ProcessedLine], 
                                 output_title: str = "Tibetan Text Processing Results") -> str:
        """
        Generate complete structured output with headers, line correspondence, and status indicators.
        
        Args:
            processed_lines: List of all processed lines
            output_title: Title for the output document
            
        Returns:
            Complete formatted output string
        """
        try:
            output_lines = []
            
            for line in processed_lines:
                formatted_line = self.format_triple_output(line)
                if formatted_line.strip():
                    output_lines.append(formatted_line)
            
            return "\n\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error generating structured output: {str(e)}")
            return ""
    
    def maintain_line_correspondence(self, input_line_num: int, output_line_num: int, 
                                   processing_status: ProcessingStatus) -> None:
        """
        Maintain correspondence between input and output line numbers.
        
        Args:
            input_line_num: Original input line number
            output_line_num: Corresponding output line number
            processing_status: Status of the line processing
        """
        # This function is no longer needed in the simplified output
        pass
    
    def _generate_document_header(self, title: str, processed_lines: List[ProcessedLine]) -> str:
        """Generate document header with metadata."""
        return "" # No longer needed
    
    def _organize_content_by_sections(self, processed_lines: List[ProcessedLine]) -> Dict[str, List[ProcessedLine]]:
        """
        Organize processed lines into logical sections based on content and structure.
        
        Args:
            processed_lines: List of processed lines
            
        Returns:
            Dictionary mapping section names to lists of lines
        """
        # For simplified output, only return content lines
        return {"Main Content": [line for line in processed_lines if not line.source_line.is_structural]}
    
    def _generate_content_section(self, section_name: str, section_lines: List[ProcessedLine]) -> str:
        """
        Generate a formatted content section.
        
        Args:
            section_name: Name of the section
            section_lines: Lines belonging to this section
            
        Returns:
            Formatted section string
        """
        section_parts = []
        
        for line in section_lines:
            formatted_line = self.format_triple_output(line)
            if formatted_line.strip():
                section_parts.append(formatted_line)
        
        return "\n".join(section_parts)
    
    def _summarize_errors(self, processed_lines: List[ProcessedLine]) -> Dict[str, int]:
        """
        Summarize error types and their frequencies."""
        # No longer needed for simplified output
        return {}
    
    def _get_status_icon(self, status: ProcessingStatus) -> str:
        """
        Get an icon/symbol for the processing status."""
        # No longer needed for simplified output
        return "" # Return empty string as status icons are not required.