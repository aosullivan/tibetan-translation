from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class ProcessingStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TibetanLine:
    line_number: int
    original_content: str
    wylie_content: str
    is_structural: bool
    markers: List[str]

@dataclass
class ProcessedLine:
    source_line: TibetanLine
    tibetan_script: str
    phonetic_rendering: str
    english_translation: str
    processing_status: ProcessingStatus
    error_message: Optional[str] = None