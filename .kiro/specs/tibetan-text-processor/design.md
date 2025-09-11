# Tibetan Text Processor Design

## Overview

The Tibetan Text Processor will be built as a specialized version of the existing translate.py system, focusing on structured processing of Wylie transliterated Tibetan text. The system will process input line by line, generating three distinct outputs: Tibetan Unicode script, phonetic rendering, and Vajrayana Buddhist translation.

## Architecture

### High-Level Architecture

```
Input File (Wylie) → Line Parser → Processing Pipeline → Output Generator
                                        ↓
                                  API Integration
                                  (Anthropic Claude)
```

### Component Structure

The system will follow the existing translate.py architecture pattern with specialized components:

**Root Level:**
1. **Main Script** (`tibetan_processor.py`) - Single entry point script in project root

**src/ Directory:**
2. **Tibetan Line Parser** (`src/tibetan_line_parser.py`) - Handles "/" delimited line parsing
3. **Wylie Converter** (`src/wylie_converter.py`) - Converts Wylie to Tibetan Unicode
4. **Phonetic Renderer** (`src/phonetic_renderer.py`) - Generates phonetic approximations
5. **Tibetan Translation Client** (`src/tibetan_translation_client.py`) - Extends existing TranslationClient
6. **Tibetan Output Formatter** (`src/tibetan_output_formatter.py`) - Creates structured output files

**Reused Components:**
- `translation_client.py` - Base API client (reused as-is)
- `file_handler.py` - Base file operations (extended for Tibetan needs)
- `config.py` - Configuration management (extended with Tibetan settings)

## Components and Interfaces

### TibetanLineParser

```python
class TibetanLineParser:
    def parse_file(self, file_path: str) -> List[TibetanLine]
    def extract_wylie_content(self, line: str) -> str
    def is_structural_marker(self, line: str) -> bool
```

**Responsibilities:**
- Parse input file using "/" as line delimiters
- Extract Wylie content from each line
- Handle structural markers like "(kha)@#/[ ]/"
- Maintain line numbering and structure

### WylieConverter

```python
class WylieConverter:
    def convert_to_tibetan(self, wylie_text: str) -> str
    def handle_sanskrit_terms(self, text: str) -> str
    def process_special_characters(self, text: str) -> str
```

**Responsibilities:**
- Convert Wylie transliteration to Unicode Tibetan script
- Handle special Wylie conventions ("+", "*", etc.)
- Process Sanskrit transliterations appropriately
- Maintain character mapping accuracy

### PhoneticRenderer

```python
class PhoneticRenderer:
    def generate_phonetic(self, wylie_text: str) -> str
    def apply_pronunciation_rules(self, syllables: List[str]) -> List[str]
    def handle_tone_markers(self, text: str) -> str
```

**Responsibilities:**
- Generate readable phonetic approximations
- Apply Tibetan pronunciation rules
- Handle tonal and aspiration markers
- Provide consistent phonetic output

### TibetanTranslationClient

```python
class TibetanTranslationClient(TranslationClient):
    def translate_vajrayana_context(self, wylie_text: str, tibetan_script: str) -> str
    def handle_technical_terms(self, text: str) -> str
    def preserve_sacred_elements(self, text: str) -> str
```

**Responsibilities:**
- **Inherit from existing TranslationClient** - Reuse all API handling, retry logic, error handling
- Provide Vajrayana Buddhist expert translations through specialized prompts
- Handle technical Buddhist terminology
- Preserve sacred and ritual elements

### TibetanFileHandler

```python
class TibetanFileHandler(FileHandler):
    def read_wylie_file(self, file_path: str) -> List[str]
    def write_tibetan_output(self, output_path: str, processed_lines: List[ProcessedLine]) -> None
    def ensure_unicode_support(self, content: str) -> str
```

**Responsibilities:**
- **Extend existing FileHandler** - Reuse base file operations and error handling
- Add Tibetan-specific file reading with "/" delimiter parsing
- Handle Unicode Tibetan script output formatting
- Maintain existing logging and error reporting patterns

### OutputFormatter

```python
class OutputFormatter:
    def format_triple_output(self, line_data: ProcessedLine) -> str
    def generate_summary_report(self, processed_lines: List[ProcessedLine]) -> str
    def ensure_unicode_encoding(self, content: str) -> str
```

**Responsibilities:**
- Format three-column output (Tibetan, Phonetic, Translation)
- Maintain line correspondence
- Ensure proper Unicode encoding
- Generate processing summaries

## Data Models

### TibetanLine

```python
@dataclass
class TibetanLine:
    line_number: int
    original_content: str
    wylie_content: str
    is_structural: bool
    markers: List[str]
```

### ProcessedLine

```python
@dataclass
class ProcessedLine:
    source_line: TibetanLine
    tibetan_script: str
    phonetic_rendering: str
    english_translation: str
    processing_status: ProcessingStatus
    error_message: Optional[str]
```

### ProcessingStatus

```python
class ProcessingStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
```

## Error Handling

### Error Categories

1. **File Processing Errors**
   - File not found or unreadable
   - Encoding issues
   - Malformed input structure

2. **Conversion Errors**
   - Invalid Wylie sequences
   - Unicode conversion failures
   - Phonetic rendering issues

3. **API Errors**
   - Network connectivity issues
   - Rate limiting
   - Authentication failures
   - Translation service errors

4. **Output Generation Errors**
   - File write permissions
   - Disk space issues
   - Unicode encoding problems

### Error Handling Strategy

- **Graceful Degradation**: Continue processing other lines when individual lines fail
- **Detailed Logging**: Log all errors with context for debugging
- **Retry Logic**: Implement exponential backoff for API failures
- **Progress Preservation**: Save intermediate results to allow resumption
- **User Feedback**: Provide clear error messages and suggestions

## Testing Strategy

### Unit Testing

1. **TibetanLineParser Tests**
   - Test "/" delimiter parsing
   - Verify structural marker detection
   - Validate Wylie content extraction

2. **WylieConverter Tests**
   - Test standard Wylie to Tibetan conversion
   - Verify special character handling
   - Test Sanskrit term processing

3. **PhoneticRenderer Tests**
   - Test pronunciation rule application
   - Verify phonetic output consistency
   - Test tone and aspiration handling

4. **API Integration Tests**
   - Test translation accuracy with sample texts
   - Verify Buddhist terminology handling
   - Test error handling and retries

### Integration Testing

1. **End-to-End Processing**
   - Process sample Tibetan texts
   - Verify output format and structure
   - Test with various input complexities

2. **Performance Testing**
   - Test with large input files
   - Measure API call efficiency
   - Verify memory usage patterns

### Validation Testing

1. **Linguistic Accuracy**
   - Validate Tibetan script conversion with experts
   - Verify phonetic rendering accuracy
   - Review translation quality with Buddhist scholars

2. **Output Quality**
   - Test Unicode rendering in various environments
   - Verify output file readability
   - Test with different Tibetan fonts

## Implementation Approach

### Phase 1: Core Infrastructure and Code Reuse
1. **Create tibetan_processor.py in root** - Single entry point following translate.py pattern
2. **Set up src/ directory structure** - Organize supporting modules
3. **Extend existing classes** - TibetanTranslationClient(TranslationClient), TibetanFileHandler(FileHandler)
4. **Reuse configuration system** - Extend config.py with Tibetan-specific settings
5. **Implement TibetanLineParser** - Handle "/" delimiter parsing in src/

### Phase 2: Tibetan Processing Pipeline
1. **Create WylieConverter** - Basic Wylie to Tibetan Unicode conversion
2. **Implement PhoneticRenderer** - Generate pronunciation from Wylie
3. **Extend translation prompts** - Add Vajrayana Buddhist context to existing API calls
4. **Create TibetanOutputFormatter** - Structure three-column output format

### Phase 3: Integration and Error Handling
1. **Integrate all components** - Wire together in main tibetan_processor.py
2. **Reuse existing error handling** - Extend logging and retry patterns from translate.py
3. **Add Tibetan-specific validation** - Unicode handling, conversion accuracy
4. **Implement progress reporting** - Follow existing patterns from translate.py

### Phase 4: Testing and Refinement
1. **Test with saras-tibetan.txt** - Validate against actual input file
2. **Verify output quality** - Check Tibetan script, phonetics, translations
3. **Optimize performance** - Batch processing, API efficiency
4. **Gather linguistic feedback** - Refine conversion and translation accuracy

## Dependencies and Integration

### Existing Dependencies (from translate.py)
- `anthropic` - API client for Claude
- `python-dotenv` - Environment variable management
- `argparse` - Command line interface
- Standard library modules for logging and file handling

### New Dependencies
- `pyewts` - Wylie to Tibetan Unicode conversion
- `tibetan-phonetics` - Tibetan phonetic rendering (if available)
- Alternative: Custom phonetic conversion based on linguistic rules

### API Integration
- Reuse existing Anthropic API configuration
- Extend prompts for Tibetan Buddhist context
- Maintain existing rate limiting and error handling
- Use same logging and monitoring patterns

## Configuration and Deployment

### Project Structure
```
project_root/
├── tibetan_processor.py          # Single entry point script
├── translate.py                  # Existing translation script
├── config.py                     # Extended with Tibetan settings
├── translation_client.py         # Reused as-is
├── file_handler.py              # Reused as base class
├── translation_manager.py       # Existing (may be reused/extended)
├── src/
│   ├── tibetan_line_parser.py
│   ├── wylie_converter.py
│   ├── phonetic_renderer.py
│   ├── tibetan_translation_client.py
│   ├── tibetan_file_handler.py
│   └── tibetan_output_formatter.py
└── requirements.txt             # Extended with Tibetan dependencies
```

### Configuration Files
- **Extend existing config.py** - Add TibetanConfig class inheriting base settings
- **Reuse API credentials** - Same ANTHROPIC_API_KEY and client configuration
- **Add Tibetan-specific options** - Wylie conversion rules, phonetic preferences
- **Maintain existing patterns** - Same configuration loading and validation

### Environment Setup
- Same virtual environment as translate.py
- Additional dependencies for Tibetan processing
- Unicode font requirements for output verification
- API key management (reused from existing setup)

### Output Specifications
- UTF-8 encoded output files
- Structured format with clear section headers
- Line-by-line correspondence maintained
- Summary statistics and processing reports