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
        self.structural_patterns = [
            r'^\[[^\]]*\]$',   # Pattern like [ ] (entire line)
            r'^\([^)]*\)$',    # Pattern like (kha) (entire line)
            r'^!_.*_!_$',       # Pattern like !_..._!_ (entire line)
            r'^[\s]*$',        # Empty or whitespace only line
        ]
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
            
            # Split on core and internal markers with CAPTURED delimiters to preserve them
            delimiter_pattern = r'(////|//_!_|/\[\u00A0\]/|/\[ \]/|//|/\_|\*/\_|_)'
            pieces = re.split(delimiter_pattern, content)

            # Build (segment, delimiter) pairs while preserving delimiters
            segments: List[str] = []
            i = 0
            while i < len(pieces):
                segment = (pieces[i] or '').strip()
                next_delim = ''
                if i + 1 < len(pieces) and pieces[i + 1] is not None:
                    next_delim = pieces[i + 1]
                if segment:
                    segments.append(segment + next_delim)
                elif next_delim:
                    # If there's no segment but a delimiter (unlikely), keep the delimiter
                    segments.append(next_delim)
                i += 2

            # Coalesce adjacent short segments into chunks of at least ~12 Wylie tokens
            MIN_TOKENS = 12
            coalesced: List[str] = []
            buffer_parts: List[str] = []
            buffer_tokens = 0

            for raw in segments:
                segment = raw.strip()
                if not segment:
                    continue

                # Structural lines flush the buffer and stand alone
                if self.is_structural_marker(segment):
                    if buffer_parts:
                        coalesced.append(' '.join(buffer_parts).strip())
                        buffer_parts = []
                        buffer_tokens = 0
                    coalesced.append(segment)
                    continue

                # Non-structural: accumulate until reaching MIN_TOKENS
                # Count tokens ignoring delimiters by replacing slashes with spaces for counting only
                token_count = len(segment.replace('/', ' ').split())
                buffer_parts.append(segment)
                buffer_tokens += token_count

                if buffer_tokens >= MIN_TOKENS:
                    coalesced.append(' '.join(buffer_parts).strip())
                    buffer_parts = []
                    buffer_tokens = 0

            # Flush any remaining buffer
            if buffer_parts:
                coalesced.append(' '.join(buffer_parts).strip())

            parsed_lines = []
            line_counter = 0
            for line in coalesced:
                line = line.strip()
                if not line:
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
        
        # Remove extra whitespace and common non-Wylie characters
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()
        
        # Remove leading/trailing underscores and other markup
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def is_structural_marker(self, line: str) -> bool:
        if not line.strip():
            return True
        
        for pattern in self.compiled_patterns:
            if pattern.fullmatch(line): # Use fullmatch to ensure the entire line matches the structural pattern
                return True
        return False
    
    def _extract_markers(self, line: str) -> List[str]:
        markers = []
        for pattern in self.compiled_patterns:
            matches = pattern.findall(line)
            markers.extend(matches)
        return markers