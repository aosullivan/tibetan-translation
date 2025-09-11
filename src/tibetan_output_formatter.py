"""
Tibetan Output Formatter for generating structured three-column output.
Handles Unicode encoding and creates formatted reports for Tibetan text processing.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from .config import TibetanConfig as cfg

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status enumeration for processed lines."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessedLine:
    """Represents a fully processed Tibetan line with all conversions."""
    source_line: Any  # TibetanLine object
    tibetan_script: str = ""
    phonetic_rendering: str = ""
    english_translation: str = ""
    processing_status: ProcessingStatus = ProcessingStatus.SUCCESS
    error_message: Optional[str] = None


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
            
            # Line header with number
            output_lines.append(f"## Line {source.line_number}")
            output_lines.append("")
            
            # Original content (always show for reference)
            output_lines.append(f"**Original:** {source.original_content}")
            output_lines.append("")
            
            # Three-column content
            if source.wylie_content.strip():
                # Wylie transliteration
                output_lines.append(f"**Wylie:** {source.wylie_content}")
                output_lines.append("")
                
                # Tibetan script (if converted)
                if processed_line.tibetan_script:
                    tibetan_safe = self.ensure_unicode_encoding(processed_line.tibetan_script)
                    output_lines.append(f"**Tibetan:** {tibetan_safe}")
                    output_lines.append("")
                
                # Phonetic rendering (if available)
                if processed_line.phonetic_rendering:
                    output_lines.append(f"**Phonetic:** {processed_line.phonetic_rendering}")
                    output_lines.append("")
                
                # English translation (if available)
                if processed_line.english_translation:
                    translation_safe = self.ensure_unicode_encoding(processed_line.english_translation)
                    output_lines.append(f"**Translation:** {translation_safe}")
                    output_lines.append("")
            
            # Processing status and metadata
            status_icon = self._get_status_icon(processed_line.processing_status)
            output_lines.append(f"**Status:** {status_icon} {processed_line.processing_status.value}")
            
            # Error information if present
            if processed_line.error_message:
                output_lines.append(f"**Error:** {processed_line.error_message}")
            
            # Structural markers if present
            if source.markers:
                markers_str = ", ".join(source.markers)
                output_lines.append(f"**Markers:** {markers_str}")
            
            output_lines.append("")
            output_lines.append("---")
            output_lines.append("")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error formatting line {processed_line.source_line.line_number}: {str(e)}")
            return f"## Line {processed_line.source_line.line_number}\n\n**Error:** Failed to format line\n\n---\n\n"
    
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
        try:
            # Calculate statistics
            stats = self._calculate_statistics(processed_lines)
            
            # Build summary report
            report_lines = []
            
            # Header
            report_lines.append("# Processing Summary Report")
            report_lines.append("")
            report_lines.append("Generated by Tibetan Text Processor")
            report_lines.append("")
            
            # Overall statistics
            report_lines.append("## Overall Statistics")
            report_lines.append("")
            report_lines.append(f"- **Total lines processed:** {stats['total_lines']}")
            report_lines.append(f"- **Successfully completed:** {stats['success_count']} ({stats['success_percentage']:.1f}%)")
            report_lines.append(f"- **Partially successful:** {stats['partial_count']} ({stats['partial_percentage']:.1f}%)")
            report_lines.append(f"- **Failed:** {stats['failed_count']} ({stats['failed_percentage']:.1f}%)")
            report_lines.append(f"- **Skipped:** {stats['skipped_count']} ({stats['skipped_percentage']:.1f}%)")
            report_lines.append("")
            
            # Content breakdown
            report_lines.append("## Content Analysis")
            report_lines.append("")
            report_lines.append(f"- **Structural markers:** {stats['structural_count']}")
            report_lines.append(f"- **Content lines:** {stats['content_count']}")
            report_lines.append(f"- **Empty lines:** {stats['empty_count']}")
            report_lines.append(f"- **Lines with Tibetan script:** {stats['tibetan_script_count']}")
            report_lines.append(f"- **Lines with phonetic rendering:** {stats['phonetic_count']}")
            report_lines.append(f"- **Lines with translations:** {stats['translation_count']}")
            report_lines.append("")
            
            # Processing details
            report_lines.append("## Processing Details")
            report_lines.append("")
            report_lines.append(f"- **Unicode encoding:** {self.unicode_encoding}")
            report_lines.append(f"- **Structural markers preserved:** {self.preserve_structural}")
            report_lines.append(f"- **Average Wylie length:** {stats['avg_wylie_length']:.1f} characters")
            report_lines.append(f"- **Average translation length:** {stats['avg_translation_length']:.1f} characters")
            report_lines.append("")
            
            # Error summary (if any failures)
            if stats['failed_count'] > 0 or stats['partial_count'] > 0:
                report_lines.append("## Issues and Errors")
                report_lines.append("")
                
                # Group errors by type
                error_summary = self._summarize_errors(processed_lines)
                for error_type, count in error_summary.items():
                    report_lines.append(f"- **{error_type}:** {count} occurrences")
                
                report_lines.append("")
                
                # List failed lines
                failed_lines = [line for line in processed_lines 
                              if line.processing_status in [ProcessingStatus.FAILED, ProcessingStatus.PARTIAL_SUCCESS]]
                
                if failed_lines:
                    report_lines.append("### Failed/Partial Lines")
                    report_lines.append("")
                    for line in failed_lines[:10]:  # Show first 10 failed lines
                        status_icon = self._get_status_icon(line.processing_status)
                        error_msg = line.error_message or "No error message"
                        report_lines.append(f"- Line {line.source_line.line_number}: {status_icon} {error_msg}")
                    
                    if len(failed_lines) > 10:
                        report_lines.append(f"- ... and {len(failed_lines) - 10} more")
                    
                    report_lines.append("")
            
            # Quality metrics
            report_lines.append("## Quality Metrics")
            report_lines.append("")
            report_lines.append(f"- **Conversion success rate:** {stats['conversion_success_rate']:.1f}%")
            report_lines.append(f"- **Translation coverage:** {stats['translation_coverage']:.1f}%")
            report_lines.append(f"- **Phonetic coverage:** {stats['phonetic_coverage']:.1f}%")
            report_lines.append("")
            
            # Footer
            report_lines.append("---")
            report_lines.append("")
            report_lines.append(f"*Report generated with {self.unicode_encoding} encoding*")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return "# Processing Summary Report\n\nError: Failed to generate summary report\n"
    
    def _calculate_statistics(self, processed_lines: List[ProcessedLine]) -> Dict[str, Any]:
        """Calculate comprehensive statistics from processed lines."""
        total_lines = len(processed_lines)
        
        if total_lines == 0:
            return {key: 0 for key in [
                'total_lines', 'success_count', 'partial_count', 'failed_count', 'skipped_count',
                'success_percentage', 'partial_percentage', 'failed_percentage', 'skipped_percentage',
                'structural_count', 'content_count', 'empty_count', 'tibetan_script_count',
                'phonetic_count', 'translation_count', 'avg_wylie_length', 'avg_translation_length',
                'conversion_success_rate', 'translation_coverage', 'phonetic_coverage'
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
        phonetic_count = sum(1 for line in processed_lines if line.phonetic_rendering)
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
            'phonetic_count': phonetic_count,
            'translation_count': translation_count,
            'avg_wylie_length': avg_wylie_length,
            'avg_translation_length': avg_translation_length,
            'conversion_success_rate': ((success_count + partial_count) / total_lines) * 100,
            'translation_coverage': (translation_count / content_count) * 100 if content_count > 0 else 0,
            'phonetic_coverage': (phonetic_count / content_count) * 100 if content_count > 0 else 0,
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
                elif 'phonetic' in error_msg:
                    error_type = "Phonetic rendering errors"
                else:
                    error_type = "Other errors"
                
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        return error_summary
    
    def _get_status_icon(self, status: ProcessingStatus) -> str:
        """Get an icon/symbol for the processing status."""
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
            output_sections = []
            
            # Document header
            output_sections.append(self._generate_document_header(output_title, processed_lines))
            
            # Table of contents
            output_sections.append(self._generate_table_of_contents(processed_lines))
            
            # Processing overview
            output_sections.append(self._generate_processing_overview(processed_lines))
            
            # Main content sections
            content_sections = self._organize_content_by_sections(processed_lines)
            for section_name, section_lines in content_sections.items():
                output_sections.append(self._generate_content_section(section_name, section_lines))
            
            # Summary and statistics
            output_sections.append(self.generate_summary_report(processed_lines))
            
            # Line correspondence index
            output_sections.append(self._generate_line_correspondence_index())
            
            return "\n\n".join(output_sections)
            
        except Exception as e:
            logger.error(f"Error generating structured output: {str(e)}")
            return f"# {output_title}\n\nError: Failed to generate structured output\n"
    
    def maintain_line_correspondence(self, input_line_num: int, output_line_num: int, 
                                   processing_status: ProcessingStatus) -> None:
        """
        Maintain correspondence between input and output line numbers.
        
        Args:
            input_line_num: Original input line number
            output_line_num: Corresponding output line number
            processing_status: Status of the line processing
        """
        self.line_correspondence[input_line_num] = {
            'output_line': output_line_num,
            'status': processing_status,
            'timestamp': None  # Could add timestamp if needed
        }
    
    def _generate_document_header(self, title: str, processed_lines: List[ProcessedLine]) -> str:
        """Generate document header with metadata."""
        total_lines = len(processed_lines)
        success_count = sum(1 for line in processed_lines if line.processing_status == ProcessingStatus.SUCCESS)
        
        header_lines = [
            f"# {title}",
            "",
            "---",
            "",
            "**Document Information:**",
            f"- Generated by: Tibetan Text Processor",
            f"- Total lines processed: {total_lines}",
            f"- Successfully processed: {success_count}",
            f"- Success rate: {(success_count/total_lines*100):.1f}%" if total_lines > 0 else "- Success rate: 0%",
            f"- Unicode encoding: {self.unicode_encoding}",
            "",
            "---"
        ]
        
        return "\n".join(header_lines)
    
    def _generate_table_of_contents(self, processed_lines: List[ProcessedLine]) -> str:
        """Generate table of contents for the document."""
        toc_lines = [
            "# Table of Contents",
            "",
            "1. [Processing Overview](#processing-overview)",
            "2. [Content Sections](#content-sections)"
        ]
        
        # Add section links based on content organization
        content_sections = self._organize_content_by_sections(processed_lines)
        section_num = 3
        for section_name in content_sections.keys():
            section_link = section_name.lower().replace(' ', '-')
            toc_lines.append(f"   - [{section_name}](#{section_link})")
        
        toc_lines.extend([
            f"{section_num}. [Processing Summary](#processing-summary-report)",
            f"{section_num + 1}. [Line Correspondence Index](#line-correspondence-index)"
        ])
        
        return "\n".join(toc_lines)
    
    def _generate_processing_overview(self, processed_lines: List[ProcessedLine]) -> str:
        """Generate processing overview section."""
        stats = self._calculate_statistics(processed_lines)
        
        overview_lines = [
            "# Processing Overview",
            "",
            "## Status Distribution",
            "",
            f"| Status | Count | Percentage |",
            f"|--------|-------|------------|",
            f"| ✅ Success | {stats['success_count']} | {stats['success_percentage']:.1f}% |",
            f"| ⚠️ Partial | {stats['partial_count']} | {stats['partial_percentage']:.1f}% |",
            f"| ❌ Failed | {stats['failed_count']} | {stats['failed_percentage']:.1f}% |",
            f"| ⏭️ Skipped | {stats['skipped_count']} | {stats['skipped_percentage']:.1f}% |",
            "",
            "## Content Breakdown",
            "",
            f"- **Structural markers:** {stats['structural_count']} lines",
            f"- **Content lines:** {stats['content_count']} lines",
            f"- **Lines with Tibetan script:** {stats['tibetan_script_count']} lines",
            f"- **Lines with phonetic rendering:** {stats['phonetic_count']} lines",
            f"- **Lines with translations:** {stats['translation_count']} lines"
        ]
        
        return "\n".join(overview_lines)
    
    def _organize_content_by_sections(self, processed_lines: List[ProcessedLine]) -> Dict[str, List[ProcessedLine]]:
        """
        Organize processed lines into logical sections based on content and structure.
        
        Args:
            processed_lines: List of processed lines
            
        Returns:
            Dictionary mapping section names to lists of lines
        """
        sections = {
            "Main Content": [],
            "Structural Elements": [],
            "Failed Processing": []
        }
        
        for line in processed_lines:
            if line.processing_status == ProcessingStatus.FAILED:
                sections["Failed Processing"].append(line)
            elif line.source_line.is_structural:
                sections["Structural Elements"].append(line)
            else:
                sections["Main Content"].append(line)
        
        # Remove empty sections
        return {name: lines for name, lines in sections.items() if lines}
    
    def _generate_content_section(self, section_name: str, section_lines: List[ProcessedLine]) -> str:
        """
        Generate a formatted content section.
        
        Args:
            section_name: Name of the section
            section_lines: Lines belonging to this section
            
        Returns:
            Formatted section string
        """
        section_parts = [
            f"# {section_name}",
            "",
            f"*{len(section_lines)} lines in this section*",
            ""
        ]
        
        # Add section-specific formatting
        if section_name == "Failed Processing":
            section_parts.append("⚠️ **Note:** These lines encountered processing errors and may need manual review.")
            section_parts.append("")
        elif section_name == "Structural Elements":
            section_parts.append("ℹ️ **Note:** These are structural markup elements from the original text.")
            section_parts.append("")
        
        # Format each line in the section
        for i, line in enumerate(section_lines, 1):
            # Update line correspondence
            self.maintain_line_correspondence(
                line.source_line.line_number, 
                i, 
                line.processing_status
            )
            
            formatted_line = self.format_triple_output(line)
            if formatted_line.strip():  # Only add non-empty formatted lines
                section_parts.append(formatted_line)
        
        return "\n".join(section_parts)
    
    def _generate_line_correspondence_index(self) -> str:
        """Generate an index showing input-output line correspondence."""
        if not self.line_correspondence:
            return "# Line Correspondence Index\n\nNo line correspondence data available."
        
        index_lines = [
            "# Line Correspondence Index",
            "",
            "This index shows the mapping between original input lines and their corresponding output locations.",
            "",
            "| Input Line | Output Section | Status | Notes |",
            "|------------|----------------|--------|-------|"
        ]
        
        for input_line, correspondence in sorted(self.line_correspondence.items()):
            status_icon = self._get_status_icon(correspondence['status'])
            output_ref = f"Line {correspondence['output_line']}"
            status_text = correspondence['status'].value
            notes = "Processed successfully" if correspondence['status'] == ProcessingStatus.SUCCESS else "See details above"
            
            index_lines.append(f"| {input_line} | {output_ref} | {status_icon} {status_text} | {notes} |")
        
        return "\n".join(index_lines)
    
    def add_processing_status_indicators(self, content: str, status: ProcessingStatus, 
                                       additional_info: str = "") -> str:
        """
        Add visual processing status indicators to content.
        
        Args:
            content: Original content string
            status: Processing status
            additional_info: Additional status information
            
        Returns:
            Content with status indicators added
        """
        status_icon = self._get_status_icon(status)
        status_badge = f"{status_icon} **{status.value.upper()}**"
        
        if additional_info:
            status_badge += f" - {additional_info}"
        
        # Add status indicator at the beginning of content
        lines = content.split('\n')
        if lines and lines[0].startswith('#'):
            # Insert after header
            lines.insert(1, f"\n{status_badge}\n")
        else:
            # Insert at beginning
            lines.insert(0, f"{status_badge}\n")
        
        return '\n'.join(lines)
    
    def create_section_headers(self, section_title: str, line_count: int, 
                             section_type: str = "content") -> str:
        """
        Create formatted section headers with metadata.
        
        Args:
            section_title: Title of the section
            line_count: Number of lines in the section
            section_type: Type of section (content, structural, errors, etc.)
            
        Returns:
            Formatted section header
        """
        # Choose appropriate icon based on section type
        type_icons = {
            "content": "📄",
            "structural": "🏗️",
            "errors": "⚠️",
            "summary": "📊",
            "index": "📇"
        }
        
        icon = type_icons.get(section_type, "📄")
        
        header_lines = [
            f"# {icon} {section_title}",
            "",
            f"*Section contains {line_count} {'line' if line_count == 1 else 'lines'}*",
            ""
        ]
        
        # Add section-specific notes
        if section_type == "errors":
            header_lines.insert(-1, "⚠️ **Note:** This section contains lines that encountered processing errors.")
        elif section_type == "structural":
            header_lines.insert(-1, "ℹ️ **Note:** This section contains structural markup from the original text.")
        
        return "\n".join(header_lines)