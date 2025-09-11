"""
Tibetan Line Parser for processing Wylie transliterated text with "/" delimiters.
Handles structural markers and extracts clean Wylie content for conversion.
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TibetanLine:
    """Represents a parsed line of Tibetan text with metadata."""
    line_number: int
    original_content: str
    wylie_content: str
    is_structural: bool
    markers: List[str]


class TibetanLineParser:
    """Parser for Wylie transliterated Tibetan text with "/" delimiters."""
    
    def __init__(self):
        # Regex patterns for structural markers
        self.structural_patterns = [
            r'\([^)]*\)@#',  # Pattern like (kha)@#
            r'\[[^\]]*\]',   # Pattern like [ ]
            r'_@#',          # Pattern like _@#
            r'//_@#',        # Pattern like //_@#
            r'_!_',          # Pattern like _!_
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.structural_patterns]
    
    def parse_file(self, file_path: str) -> List[TibetanLine]:
        """
        Parse a Tibetan text file, splitting on "/" delimiters.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            List of TibetanLine objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split on "/" delimiter
            raw_lines = content.split('/')
            
            parsed_lines = []
            line_counter = 0
            for line in raw_lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                line_counter += 1
                
                wylie_content = self.extract_wylie_content(line)
                is_structural = self.is_structural_marker(line)
                markers = self._extract_markers(line)
                
                tibetan_line = TibetanLine(
                    line_number=line_counter,
                    original_content=line,
                    wylie_content=wylie_content,
                    is_structural=is_structural,
                    markers=markers
                )
                
                parsed_lines.append(tibetan_line)
                
            logger.info(f"Parsed {len(parsed_lines)} lines from {file_path}")
            return parsed_lines
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise
    
    def extract_wylie_content(self, line: str) -> str:
        """
        Extract clean Wylie content from a line, removing structural markers.
        
        Args:
            line: Raw line content
            
        Returns:
            Cleaned Wylie text
        """
        if not line:
            return ""
        
        # Remove structural markers
        cleaned = line
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Remove leading slashes that might remain
        cleaned = cleaned.lstrip('/')
        
        # Remove extra whitespace and common non-Wylie characters
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()
        
        # Remove leading/trailing underscores and other markup
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def is_structural_marker(self, line: str) -> bool:
        """
        Determine if a line contains structural markup elements.
        
        Args:
            line: Line content to check
            
        Returns:
            True if line contains structural markers
        """
        if not line:
            return False
        
        # Check for any structural patterns
        for pattern in self.compiled_patterns:
            if pattern.search(line):
                return True
        
        # Additional checks for other structural indicators
        structural_indicators = [
            line.startswith('_'),
            line.endswith('_'),
            '//' in line,
            line.strip() == '',
            re.search(r'^[\s_@#\[\]()]+$', line)  # Only contains structural chars
        ]
        
        return any(structural_indicators)
    
    def _extract_markers(self, line: str) -> List[str]:
        """
        Extract all structural markers from a line.
        
        Args:
            line: Line content
            
        Returns:
            List of found markers
        """
        markers = []
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(line)
            markers.extend(matches)
        
        return markers