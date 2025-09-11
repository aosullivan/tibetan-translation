# Tibetan Text Processor 🕉️

A comprehensive tool for processing classical Tibetan texts with Wylie transliteration. Converts Wylie to Tibetan Unicode script, generates phonetic renderings, and provides Vajrayana Buddhist expert translations using the Anthropic Claude API.

## Features

- **Wylie to Tibetan Script Conversion**: Converts Wylie transliteration to Unicode Tibetan script using the `pyewts` library
- **Phonetic Rendering**: Generates readable phonetic approximations of Tibetan pronunciation
- **Buddhist Expert Translation**: Provides contextually accurate translations with Vajrayana Buddhist expertise
- **Three-Column Output**: Structured format showing Tibetan script, phonetics, and English translation side-by-side
- **Progress Tracking**: Resumable processing with automatic progress saving
- **Error Handling**: Graceful degradation with detailed error reporting
- **Performance Optimization**: Batch processing and memory monitoring for large texts

## Quick Start

1. **Install dependencies**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set up API credentials**:
```bash
# Create .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

3. **Process Tibetan text**:
```bash
# Basic usage with default settings
python tibetan_processor.py

# Custom input/output files
python tibetan_processor.py -i your_wylie_file.txt -o output.md

# Adjust batch size and phonetic style
python tibetan_processor.py -b 5 -p detailed
```

## Input Format

The processor expects Wylie transliterated Tibetan text with specific formatting:

- **Line delimiter**: Forward slash `/` separates logical lines
- **Structural markers**: Elements like `(kha)@#/[ ]/` mark document structure
- **Wylie content**: Standard Wylie transliteration between delimiters

**Example input format**:
```
(kha)@#/[ ]/bo dong paN chen phyogs las rnam rgyal nas brgyud pa'i/
dbyangs can dkar mo'i sgrub thabs gegs sel rjes gnang/
bstod pa dang bcas pa'i skor rnams bzhugs so//_@#/[ ]/
lha mo dbyangs can ma la phyag 'tshal lo/[ ]/
```

## Output Format

The processor generates a structured three-column markdown format:

```markdown
## Line 1

**Original:** (kha)@#/[ ]/bo dong paN chen phyogs las rnam rgyal nas brgyud pa'i/

**Wylie:** bo dong paN chen phyogs las rnam rgyal nas brgyud pa'i

**Tibetan:** བོ་དོང་པཎ་ཆེན་ཕྱོགས་ལས་རྣམ་རྒྱལ་ནས་བརྒྱུད་པའི

**Phonetic:** bo dong pan chen chok le nam gyal ne gyü pé

**Translation:** From the lineage transmitted by Bodong Panchen Chokle Namgyal

**Status:** ✅ success

---
```

## Command Line Options

```bash
python tibetan_processor.py [OPTIONS]

Options:
  -i, --input FILE          Input file path (default: saras-tibetan.txt)
  -o, --output FILE         Output file path (default: tibetan_output.md)
  -b, --batch-size N        Lines to process per batch (default: 10)
  -p, --phonetic-style STY  Phonetic style: simplified|detailed|academic (default: simplified)
  -h, --help               Show help message
```

## Configuration

Customize processing behavior by editing `config.py`:

### Basic Settings
```python
# Input/Output files
INPUT_FILE = "saras-tibetan.txt"
OUTPUT_FILE = "tibetan_output.md"

# Processing options
BATCH_SIZE = 10
PHONETIC_STYLE = "simplified"  # simplified, detailed, academic
SKIP_EMPTY_LINES = True
```

### Advanced Settings
```python
# Performance optimization
ENABLE_CACHING = True
MAX_CONCURRENT_API_CALLS = 3
MEMORY_MONITORING_INTERVAL = 10

# Buddhist translation context
TRANSLATION_CONTEXT = "vajrayana_buddhist"
PRESERVE_SANSKRIT_TERMS = True
HANDLE_MANTRAS_SPECIALLY = True

# Progress and error handling
ENABLE_PROGRESS_SAVING = True
AUTO_SAVE_INTERVAL = 50
```

## Processing Workflow

1. **Line Parsing**: Splits input on `/` delimiters and extracts Wylie content
2. **Wylie Conversion**: Converts to Unicode Tibetan script using linguistic rules
3. **Phonetic Rendering**: Generates pronunciation based on Tibetan phonetic rules
4. **Translation**: Provides contextually accurate Buddhist translations via Claude API
5. **Output Formatting**: Creates structured three-column markdown output
6. **Progress Tracking**: Saves intermediate results for resumption capability

## Examples

### Basic Processing
```bash
# Process the default input file
python tibetan_processor.py

# Output: tibetan_output.md with three-column format
```

### Custom Configuration
```bash
# Process with detailed phonetics and smaller batches
python tibetan_processor.py \
  --input my_tibetan_text.txt \
  --output detailed_output.md \
  --batch-size 5 \
  --phonetic-style detailed
```

### Resume Interrupted Processing
```bash
# If processing was interrupted, simply run again
# Progress will be automatically resumed from the last saved state
python tibetan_processor.py
```

## Output Examples

### Successful Processing
```markdown
## Line 15

**Original:** lha mo dbyangs can ma la phyag 'tshal lo/[ ]/

**Wylie:** lha mo dbyangs can ma la phyag 'tshal lo

**Tibetan:** ལྷ་མོ་དབྱངས་ཅན་མ་ལ་ཕྱག་འཚལ་ལོ

**Phonetic:** lha mo yang chen ma la chak tsal lo

**Translation:** I prostrate to the goddess Saraswati

**Status:** ✅ success

---
```

### Partial Success (with warnings)
```markdown
## Line 23

**Original:** complex_wylie_with_errors/

**Wylie:** complex_wylie_with_errors

**Tibetan:** [Conversion failed: complex_wylie_with_errors]

**Phonetic:** complex wylie with errors

**Translation:** [Translation of uncertain Wylie transliteration]

**Status:** ⚠️ partial

**Error:** Wylie conversion encountered unknown sequences

---
```

## Performance and Optimization

### Memory Usage
- **Batch Processing**: Processes lines in configurable batches to manage memory
- **Progress Saving**: Automatically saves progress every 50 lines (configurable)
- **Memory Monitoring**: Tracks memory usage and provides warnings

### API Efficiency
- **Rate Limiting**: Respects API rate limits with exponential backoff
- **Concurrent Calls**: Configurable concurrent API calls (default: 3)
- **Caching**: Optional caching of translation results
- **Retry Logic**: Automatic retry on transient failures

### Large File Processing
```bash
# For very large files, use smaller batches and enable all optimizations
python tibetan_processor.py \
  --batch-size 5 \
  --input large_tibetan_corpus.txt \
  --output large_output.md

# Monitor progress in tibetan_processing.log
tail -f tibetan_processing.log
```

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
Error: ANTHROPIC_API_KEY environment variable not set
```
**Solution**: Create `.env` file with your API key or set environment variable

**2. Input File Format Issues**
```
Warning: No lines found to process in input file
```
**Solution**: Ensure input uses `/` as line delimiters and contains Wylie content

**3. Unicode Display Problems**
```
Warning: Unicode encoding issue
```
**Solution**: Ensure your terminal/editor supports Unicode Tibetan fonts

**4. Memory Issues with Large Files**
```
Warning: High memory usage detected
```
**Solution**: Reduce batch size and enable memory monitoring:
```bash
python tibetan_processor.py --batch-size 3
```

### Debug Mode
Enable detailed logging by setting log level in the script:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Progress Recovery
If processing is interrupted:
1. Check for `.progress` file in output directory
2. Simply re-run the same command - processing will resume automatically
3. Delete `.progress` file to start fresh if needed

## Dependencies

- **anthropic**: Claude API client for translations
- **python-dotenv**: Environment variable management
- **pyewts**: Wylie to Tibetan Unicode conversion
- **pynacl**: Cryptographic functions (dependency)

## File Structure

```
tibetan-text-processor/
├── tibetan_processor.py          # Main entry point
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── .env                         # API credentials (create this)
├── saras-tibetan.txt            # Default input file
├── src/                         # Processing modules
│   ├── tibetan_line_parser.py   # Line parsing and structure
│   ├── wylie_converter.py       # Wylie to Tibetan conversion
│   ├── phonetic_renderer.py     # Phonetic rendering
│   ├── tibetan_translation_client.py  # Buddhist translation
│   ├── tibetan_file_handler.py  # File I/O operations
│   └── tibetan_output_formatter.py    # Output formatting
└── tests/                       # Unit tests
    └── test_*.py               # Test modules
```

## Advanced Usage

### Custom Translation Prompts
Modify `src/tibetan_translation_client.py` to customize translation behavior for specific text types or terminology.

### Phonetic Style Customization
The phonetic renderer supports three styles:
- **simplified**: Basic readable approximation
- **detailed**: Includes tone and aspiration markers
- **academic**: Scholarly transcription with diacritics

### Batch Processing Multiple Files
```bash
# Process multiple files in sequence
for file in *.txt; do
    python tibetan_processor.py -i "$file" -o "${file%.txt}_output.md"
done
```

## Contributing

1. Run tests: `python -m pytest tests/`
2. Check code style: Follow existing patterns
3. Add tests for new features
4. Update documentation for changes

## License

This tool is designed for academic and religious study of Tibetan Buddhist texts. Please use respectfully and in accordance with traditional guidelines for handling sacred texts.
