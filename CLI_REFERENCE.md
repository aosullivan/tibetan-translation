# Tibetan Text Processor - Command Line Reference

Complete reference for all command-line options and usage patterns.

## Synopsis

```bash
python tibetan_processor.py [OPTIONS]
```

## Description

The Tibetan Text Processor converts Wylie transliterated Tibetan text into Unicode Tibetan script, generates phonetic renderings, and provides Vajrayana Buddhist expert translations. It processes input line-by-line and generates structured three-column output.

## Options

### Input/Output Options

#### `-i, --input FILE`
**Description**: Specify the input file containing Wylie transliterated text  
**Default**: `saras-tibetan.txt` (from config)  
**Format**: File should use `/` as line delimiters  
**Example**:
```bash
python tibetan_processor.py -i my_text.txt
python tibetan_processor.py --input /path/to/tibetan_manuscript.txt
```

#### `-o, --output FILE`
**Description**: Specify the output file for processed results  
**Default**: `tibetan_output.md` (from config)  
**Format**: Markdown file with three-column layout  
**Example**:
```bash
python tibetan_processor.py -o results.md
python tibetan_processor.py --output /path/to/output.md
```

### Processing Options

#### `-b, --batch-size N`
**Description**: Number of lines to process in each batch  
**Default**: `10` (from config)  
**Range**: 1-50 (recommended)  
**Impact**: 
- Smaller values: Lower memory usage, slower processing
- Larger values: Higher memory usage, faster processing
**Example**:
```bash
python tibetan_processor.py -b 5      # Conservative memory usage
python tibetan_processor.py -b 20     # Faster processing
python tibetan_processor.py --batch-size 15
```

#### `-p, --phonetic-style STYLE`
**Description**: Style of phonetic rendering to generate  
**Default**: `simplified` (from config)  
**Options**:
- `simplified`: Basic readable approximation
- `detailed`: Includes tone and aspiration markers  
- `academic`: Scholarly transcription with diacritics
**Example**:
```bash
python tibetan_processor.py -p simplified
python tibetan_processor.py -p detailed
python tibetan_processor.py --phonetic-style academic
```

### Help and Information

#### `-h, --help`
**Description**: Show help message and exit  
**Example**:
```bash
python tibetan_processor.py -h
python tibetan_processor.py --help
```

## Usage Patterns

### Basic Usage

```bash
# Process default input file with default settings
python tibetan_processor.py

# Process specific file
python tibetan_processor.py -i my_text.txt

# Process with custom output location
python tibetan_processor.py -o /tmp/results.md
```

### Common Combinations

```bash
# Small file, high quality
python tibetan_processor.py -i small_text.txt -p detailed -b 5

# Large file, optimized for speed
python tibetan_processor.py -i large_corpus.txt -b 20 -p simplified

# Academic processing
python tibetan_processor.py -i manuscript.txt -o analysis.md -p academic -b 8

# Memory-constrained processing
python tibetan_processor.py -i huge_file.txt -b 3 -o output.md
```

### Advanced Usage

```bash
# Process multiple files in sequence
for file in *.txt; do
    python tibetan_processor.py -i "$file" -o "${file%.txt}_output.md"
done

# Resume interrupted processing (automatic)
python tibetan_processor.py -i large_file.txt -o output.md
# Will automatically resume from last saved progress

# Debug mode with detailed logging
python tibetan_processor.py -i debug_text.txt -b 1
# Check tibetan_processing.log for detailed information
```

## Exit Codes

The processor returns different exit codes to indicate processing status:

| Exit Code | Meaning | Description |
|-----------|---------|-------------|
| `0` | Success | All lines processed successfully |
| `1` | Error | Fatal error or user interruption |
| `2` | Partial Success | Some lines failed but processing completed |

**Example Usage in Scripts**:
```bash
#!/bin/bash
python tibetan_processor.py -i input.txt -o output.md
exit_code=$?

case $exit_code in
    0) echo "Processing completed successfully" ;;
    1) echo "Processing failed with errors" ;;
    2) echo "Processing completed with some failures" ;;
esac
```

## Environment Variables

### Required

#### `ANTHROPIC_API_KEY`
**Description**: API key for Anthropic Claude service  
**Format**: String starting with `sk-ant-`  
**Setup**:
```bash
# Method 1: .env file (recommended)
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env

# Method 2: Environment variable
export ANTHROPIC_API_KEY="sk-ant-your-key"
```

### Optional

#### `PYTHONUNBUFFERED`
**Description**: Disable Python output buffering for real-time logs  
**Example**:
```bash
export PYTHONUNBUFFERED=1
python tibetan_processor.py -i large_file.txt
```

#### `PYTHONIOENCODING`
**Description**: Set Python I/O encoding (useful on Windows)  
**Example**:
```bash
export PYTHONIOENCODING=utf-8
python tibetan_processor.py -i tibetan_text.txt
```

## Configuration File

The processor uses `config.py` for default settings. Command-line options override config file values.

### Key Configuration Options

```python
# Input/Output defaults
INPUT_FILE = "saras-tibetan.txt"
OUTPUT_FILE = "tibetan_output.md"

# Processing defaults
BATCH_SIZE = 10
PHONETIC_STYLE = "simplified"

# Performance settings
MAX_CONCURRENT_API_CALLS = 3
ENABLE_PROGRESS_SAVING = True
AUTO_SAVE_INTERVAL = 50

# Buddhist translation settings
TRANSLATION_CONTEXT = "vajrayana_buddhist"
PRESERVE_SANSKRIT_TERMS = True
HANDLE_MANTRAS_SPECIALLY = True
```

## File Formats

### Input File Format

**Requirements**:
- UTF-8 encoding
- Wylie transliteration
- Lines separated by `/` character
- Structural markers like `(kha)@#` and `[ ]` supported

**Example Input**:
```
(kha)@#/[ ]/lha mo dbyangs can ma'i sgrub thabs/[ ]/
lha mo dbyangs can ma la phyag 'tshal lo/[ ]/
sdig pa mi dge ba spang*/_tshangs par spyad/[ ]/
```

### Output File Format

**Structure**:
- Markdown format
- Three-column layout per line
- Processing status indicators
- Error reporting
- Summary statistics

**Example Output Structure**:
```markdown
# Tibetan Text Processing Results

## Line 1
**Original:** [original content]
**Wylie:** [wylie transliteration]
**Tibetan:** [unicode tibetan]
**Phonetic:** [phonetic rendering]
**Translation:** [english translation]
**Status:** ✅ success
---
```

## Logging

### Log File Location
- **File**: `tibetan_processing.log`
- **Location**: Current working directory
- **Format**: Timestamped entries with log levels

### Log Levels
- **INFO**: Normal processing information
- **WARNING**: Non-fatal issues (conversion failures, etc.)
- **ERROR**: Serious errors that may affect processing
- **DEBUG**: Detailed debugging information (enable in code)

### Monitoring Progress
```bash
# Watch log file in real-time
tail -f tibetan_processing.log

# Filter for specific information
grep -i "error\|warning" tibetan_processing.log
grep "Progress:" tibetan_processing.log
```

## Performance Considerations

### Memory Usage

**Factors Affecting Memory**:
- Batch size (larger = more memory)
- File size (larger files need more memory)
- Caching settings (caching uses more memory)

**Optimization**:
```bash
# Low memory systems
python tibetan_processor.py -b 3 -i large_file.txt

# High memory systems
python tibetan_processor.py -b 25 -i large_file.txt
```

### Processing Speed

**Factors Affecting Speed**:
- Batch size (larger = faster, up to API limits)
- Network latency to API
- Concurrent API calls setting
- Phonetic style complexity

**Optimization**:
```bash
# Speed-optimized
python tibetan_processor.py -b 20 -p simplified

# Quality-optimized
python tibetan_processor.py -b 5 -p academic
```

### API Rate Limits

**Handling Rate Limits**:
- Automatic retry with exponential backoff
- Configurable concurrent call limits
- Progress saving for resumption

**Rate Limit Configuration** (in `config.py`):
```python
MAX_CONCURRENT_API_CALLS = 3  # Reduce if hitting limits
API_RETRY_DELAY = 1.0         # Increase for more conservative approach
```

## Error Handling

### Common Error Scenarios

#### File Not Found
```bash
$ python tibetan_processor.py -i nonexistent.txt
Error: Input file not found: nonexistent.txt
```

#### Invalid API Key
```bash
$ python tibetan_processor.py
Error: ANTHROPIC_API_KEY environment variable not set
```

#### Processing Errors
```bash
# Partial processing with some failures
$ python tibetan_processor.py -i problematic_input.txt
Warning: Wylie conversion failed for line 15
Processing completed with 2 failures out of 100 lines
```

### Recovery Options

#### Resume Interrupted Processing
```bash
# If processing was interrupted, simply run again
python tibetan_processor.py -i large_file.txt -o output.md
# Automatically resumes from last saved progress
```

#### Force Fresh Start
```bash
# Remove progress file to start over
rm output.md.progress
python tibetan_processor.py -i large_file.txt -o output.md
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# process_tibetan_batch.sh

INPUT_DIR="input_texts"
OUTPUT_DIR="processed_outputs"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.txt; do
    basename=$(basename "$file" .txt)
    echo "Processing $basename..."
    
    python tibetan_processor.py \
        -i "$file" \
        -o "$OUTPUT_DIR/${basename}_processed.md" \
        -b 10 \
        -p simplified
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $basename"
    else
        echo "✗ Failed to process $basename"
    fi
done
```

### Python Integration

```python
#!/usr/bin/env python3
# wrapper_script.py

import subprocess
import sys
import os

def process_tibetan_file(input_file, output_file, batch_size=10):
    """Process a Tibetan file using the processor."""
    cmd = [
        sys.executable, 'tibetan_processor.py',
        '-i', input_file,
        '-o', output_file,
        '-b', str(batch_size)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

# Usage
success, stdout, stderr = process_tibetan_file('input.txt', 'output.md')
if success:
    print("Processing completed successfully")
else:
    print(f"Processing failed: {stderr}")
```

### Makefile Integration

```makefile
# Makefile for Tibetan text processing

PROCESSOR = python tibetan_processor.py
INPUT_DIR = inputs
OUTPUT_DIR = outputs
BATCH_SIZE = 10

# Process all .txt files
all: $(patsubst $(INPUT_DIR)/%.txt,$(OUTPUT_DIR)/%.md,$(wildcard $(INPUT_DIR)/*.txt))

$(OUTPUT_DIR)/%.md: $(INPUT_DIR)/%.txt
	@mkdir -p $(OUTPUT_DIR)
	$(PROCESSOR) -i $< -o $@ -b $(BATCH_SIZE)

clean:
	rm -rf $(OUTPUT_DIR)
	rm -f *.progress *.log

.PHONY: all clean
```

## Troubleshooting Quick Reference

| Issue | Command/Solution |
|-------|------------------|
| Check installation | `python -c "import src.tibetan_line_parser; print('OK')"` |
| Test API connection | `python -c "from anthropic import Anthropic; print('OK')"` |
| Verify input format | `head -5 input.txt` |
| Check Unicode support | `echo "ལྷ་མོ" \| cat` |
| Monitor memory usage | `python memory_test.py` |
| View processing logs | `tail -f tibetan_processing.log` |
| Resume processing | `python tibetan_processor.py -i file.txt -o output.md` |
| Start fresh | `rm output.md.progress && python tibetan_processor.py ...` |

## Best Practices

### File Organization
```
project/
├── inputs/           # Input Wylie files
├── outputs/          # Processed results
├── logs/            # Processing logs
└── config/          # Configuration backups
```

### Naming Conventions
```bash
# Input files
manuscript_001_wylie.txt
daily_practice_wylie.txt

# Output files
manuscript_001_processed.md
daily_practice_processed.md

# Batch processing
python tibetan_processor.py -i "inputs/${name}_wylie.txt" -o "outputs/${name}_processed.md"
```

### Version Control
```bash
# Track configuration changes
git add config.py
git commit -m "Adjust batch size for large corpus processing"

# Ignore generated files
echo "*.log" >> .gitignore
echo "*.progress" >> .gitignore
echo "outputs/" >> .gitignore
```

This command-line reference provides comprehensive information for using the Tibetan Text Processor effectively in various scenarios and environments.