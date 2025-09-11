# Tibetan Text Processor - Usage Examples

This document provides comprehensive examples of using the Tibetan Text Processor for various scenarios and text types.

## Table of Contents

1. [Basic Usage Examples](#basic-usage-examples)
2. [Input Format Examples](#input-format-examples)
3. [Configuration Examples](#configuration-examples)
4. [Output Format Examples](#output-format-examples)
5. [Advanced Usage Scenarios](#advanced-usage-scenarios)
6. [Batch Processing Examples](#batch-processing-examples)
7. [Troubleshooting Examples](#troubleshooting-examples)

## Basic Usage Examples

### Example 1: Simple Saraswati Invocation

**Input file** (`simple_invocation.txt`):
```
lha mo dbyangs can ma la phyag 'tshal lo/[ ]/
```

**Command**:
```bash
python tibetan_processor.py -i simple_invocation.txt -o simple_output.md
```

**Expected Output** (`simple_output.md`):
```markdown
## Line 1

**Original:** lha mo dbyangs can ma la phyag 'tshal lo/[ ]/

**Wylie:** lha mo dbyangs can ma la phyag 'tshal lo

**Tibetan:** ལྷ་མོ་དབྱངས་ཅན་མ་ལ་ཕྱག་འཚལ་ལོ

**Phonetic:** lha mo yang chen ma la chak tsal lo

**Translation:** I prostrate to the goddess Saraswati, the melodious one.

**Status:** ✅ success

---
```

### Example 2: Multiple Lines with Structural Markers

**Input file** (`structured_text.txt`):
```
(kha)@#/[ ]/dbyangs can ma'i sgrub thabs/[ ]/
lha mo dbyangs can ma la phyag 'tshal lo/[ ]/
sdig pa mi dge ba spang*/_tshangs par spyad/[ ]/
```

**Command**:
```bash
python tibetan_processor.py -i structured_text.txt -o structured_output.md --batch-size 5
```

**Key Features Demonstrated**:
- Structural markers `(kha)@#` and `[ ]` are preserved
- Multiple content lines processed in sequence
- Batch processing with custom batch size

## Input Format Examples

### Example 3: Complex Tibetan Text with Various Elements

**Input file** (`complex_text.txt`):
```
(kha)@#/[ ]/bo dong paN chen phyogs las rnam rgyal nas brgyud pa'i/
dbyangs can dkar mo'i sgrub thabs gegs sel rjes gnang/
bstod pa dang bcas pa'i skor rnams bzhugs so//_@#/[ ]/
lha mo dbyangs can ma la phyag 'tshal lo/[ ]/
dbyangs can ma sgrub par 'dod pas sdig to mi dge ba spang*/
_tshangs par spyad/_khyad par rdzun mi smra/
_sha chang sgog btsong la sogs pa'i bza' btung spang*/[ ]/
oM pr-i tI de wI hrIHde bya puSh+pe hU~M/
nas bi le pa ni hU~M/_zhes pa'i bar sngags rnams/[ ]/
```

**Elements Demonstrated**:
- **Structural markers**: `(kha)@#`, `[ ]`, `_@#`
- **Sanskrit mantras**: `oM pr-i tI de wI hrIHde bya puSh+pe hU~M`
- **Wylie conventions**: Apostrophes, underscores, asterisks
- **Line continuations**: Text flowing across multiple segments

### Example 4: Mantra and Sacred Syllables

**Input file** (`mantras.txt`):
```
oM AHhU~M lan gsum brjod pas bdud rtsir byin gyis brlabs te/[ ]/
oM bA su d+ha ri hU~M swA hA/_zhes pa lan nyer gcig gis dbul lo/[ ]/
hrIMH_zhes brjod pa dang 'khor lo'i dbus kyi hrIMHgi nA da las/[ ]/
```

**Special Handling**:
- Sanskrit syllables preserved in translation
- Sacred mantras treated with appropriate reverence
- Tibetan phonetic rendering of Sanskrit elements

## Configuration Examples

### Example 5: Detailed Phonetic Style

**Configuration** (modify `config.py`):
```python
PHONETIC_STYLE = "detailed"  # Instead of "simplified"
```

**Command**:
```bash
python tibetan_processor.py -p detailed -i input.txt -o detailed_output.md
```

**Output Difference**:
```markdown
# Simplified phonetic:
**Phonetic:** lha mo yang chen ma

# Detailed phonetic:
**Phonetic:** lha mo dbyaṅs can ma (with tone markers and aspirations)
```

### Example 6: Academic Phonetic Style

**Command**:
```bash
python tibetan_processor.py --phonetic-style academic -i scholarly_text.txt
```

**Academic Output Features**:
- Scholarly transcription with diacritics
- Precise linguistic notation
- Academic conventions for Tibetan phonetics

### Example 7: Custom Batch Processing

**Configuration for Large Files**:
```python
# In config.py
BATCH_SIZE = 20  # Larger batches for efficiency
ENABLE_PROGRESS_SAVING = True
AUTO_SAVE_INTERVAL = 25
MAX_CONCURRENT_API_CALLS = 5
```

**Command**:
```bash
python tibetan_processor.py -b 20 -i large_corpus.txt -o large_output.md
```

## Output Format Examples

### Example 8: Successful Processing Output

```markdown
## Line 5

**Original:** khrus dang gtsang sbra bya/_zas dang skom dang gnas gtsang ba lta bur blang ngo*/

**Wylie:** khrus dang gtsang sbra bya/_zas dang skom dang gnas gtsang ba lta bur blang ngo*

**Tibetan:** ཁྲུས་དང་གཙང་སྦྲ་བྱ་ཟས་དང་སྐོམ་དང་གནས་གཙང་བ་ལྟ་བུར་བླང་ངོ

**Phonetic:** trü dang tsang tra ja zé dang kom dang né tsang wa ta wur lang ngo

**Translation:** Perform ablutions and maintain cleanliness, and partake of food, drink, and dwelling places that are pure.

**Status:** ✅ success

---
```

### Example 9: Partial Success with Warnings

```markdown
## Line 12

**Original:** complex_wylie_sequence_with_errors/

**Wylie:** complex_wylie_sequence_with_errors

**Tibetan:** [Conversion failed: unknown Wylie sequence 'xyz']

**Phonetic:** complex wylie sequence with errors

**Translation:** [Translation based on partial conversion and context]

**Status:** ⚠️ partial

**Error:** Wylie conversion encountered unknown sequences

---
```

### Example 10: Structural Element Processing

```markdown
## Line 1

**Original:** (kha)@#/[ ]/bo dong paN chen phyogs las rnam rgyal nas brgyud pa'i skor/

**Wylie:** bo dong paN chen phyogs las rnam rgyal nas brgyud pa'i skor

**Tibetan:** བོ་དོང་པཎ་ཆེན་ཕྱོགས་ལས་རྣམ་རྒྱལ་ནས་བརྒྱུད་པའི་སྐོར

**Phonetic:** bo dong pan chen chok lé nam gyal né gyü pé kor

**Translation:** The cycle transmitted through the lineage of Bodong Panchen Chokle Namgyal

**Status:** ✅ success

**Markers:** (kha)@#, [ ]

---
```

## Advanced Usage Scenarios

### Example 11: Processing Different Text Types

#### Ritual Instructions
**Input**:
```
thun 'jog khar rang gi snying gi pad+ma bsdus te/
dper na bum sku bsdus pa bzhin du bsams nas/
mtha' nas phar bcom ldan 'das gshin rje'i gshed dam/[ ]/
```

**Output Features**:
- Ritual terminology preserved
- Technical Buddhist terms explained
- Meditation instructions clearly translated

#### Philosophical Text
**Input**:
```
stong pa nyid kyi ye shes kyis/
'khor ba dang mya ngan las 'das pa/
gnyis su med par rtogs pa ni/[ ]/
```

**Output Features**:
- Philosophical concepts accurately rendered
- Technical terms like "śūnyatā" (stong pa nyid) properly handled
- Context-appropriate translations

### Example 12: Mantra Processing

**Input**:
```
oM ma Ni pad+me hU~M/_oM tA re tu tA re tu re swA hA/
gate gate pA ra gate pA ra saM gate bod+hi swA hA/[ ]/
```

**Special Handling**:
- Sanskrit mantras preserved in original form
- Phonetic rendering provided for pronunciation
- Sacred context maintained in translation
- No literal translation of mantras (preserved as sacred sounds)

### Example 13: Mixed Sanskrit-Tibetan Text

**Input**:
```
sarwa buddha bodhisattva/_sangs rgyas byang chub sems dpa' thams cad/
vajra guru padma siddhi hU~M/_rdo rje bla ma pad+ma dngos grub hU~M/[ ]/
```

**Processing Features**:
- Sanskrit terms identified and preserved
- Tibetan equivalents provided where appropriate
- Mixed-language phonetics handled correctly

## Batch Processing Examples

### Example 14: Processing Multiple Files

**Bash Script** (`process_multiple.sh`):
```bash
#!/bin/bash

# Process all .txt files in input directory
for file in input/*.txt; do
    basename=$(basename "$file" .txt)
    echo "Processing $file..."
    
    python tibetan_processor.py \
        -i "$file" \
        -o "output/${basename}_processed.md" \
        --batch-size 10 \
        --phonetic-style simplified
    
    echo "Completed $basename"
done

echo "All files processed!"
```

### Example 15: Large Corpus Processing

**Configuration for Large Files**:
```python
# config.py settings for large corpus
BATCH_SIZE = 15
ENABLE_PROGRESS_SAVING = True
AUTO_SAVE_INTERVAL = 30
MEMORY_MONITORING_INTERVAL = 5
MAX_CONCURRENT_API_CALLS = 3
```

**Command**:
```bash
# Process large file with progress monitoring
python tibetan_processor.py \
    -i large_tibetan_corpus.txt \
    -o corpus_output.md \
    --batch-size 15

# Monitor progress
tail -f tibetan_processing.log
```

### Example 16: Resume Interrupted Processing

**Scenario**: Processing was interrupted at line 150 of 500

**Command**:
```bash
# Simply run the same command again
python tibetan_processor.py -i large_file.txt -o output.md

# The processor will automatically detect the .progress file and resume
```

**Log Output**:
```
INFO - Resuming from line 151
INFO - Processing batch 16/50 (15 lines)
INFO - Progress: 30.2% - Estimated time remaining: 12.3 minutes
```

## Troubleshooting Examples

### Example 17: Handling Conversion Errors

**Input with Problematic Wylie**:
```
lha mo dbyangs can ma/
invalid_wylie_xyz_sequence/
normal_wylie_again/[ ]/
```

**Output**:
```markdown
## Line 2

**Original:** invalid_wylie_xyz_sequence/

**Wylie:** invalid_wylie_xyz_sequence

**Tibetan:** [Conversion failed: invalid_wylie_xyz_sequence]

**Phonetic:** invalid wylie xyz sequence

**Translation:** [Unable to process due to conversion error]

**Status:** ❌ failed

**Error:** Unknown Wylie sequence 'xyz'

---
```

**Resolution Strategy**:
1. Check input for typos
2. Verify Wylie transliteration standards
3. Use fallback processing for unknown sequences

### Example 18: Memory Management for Large Files

**Problem**: Processing 10,000+ line file causes memory issues

**Solution**:
```bash
# Use smaller batch size and enable monitoring
python tibetan_processor.py \
    -i huge_file.txt \
    -o huge_output.md \
    --batch-size 3

# Monitor memory usage
python memory_test.py
```

**Configuration Adjustments**:
```python
# config.py for memory-constrained systems
BATCH_SIZE = 3
MEMORY_MONITORING_INTERVAL = 1
AUTO_SAVE_INTERVAL = 10
ENABLE_CACHING = False  # Disable to save memory
```

### Example 19: API Rate Limiting Handling

**Problem**: Rate limit exceeded during processing

**Log Output**:
```
WARNING - Rate limit exceeded, waiting 60 seconds...
INFO - Retrying API call (attempt 2/3)
```

**Configuration Solution**:
```python
# config.py adjustments
MAX_CONCURRENT_API_CALLS = 1  # Reduce concurrent calls
API_RETRY_DELAY = 2.0  # Increase delay between calls
BATCH_SIZE = 5  # Smaller batches
```

## Performance Optimization Examples

### Example 20: High-Performance Configuration

**For Fast Processing** (when API limits allow):
```python
# config.py for speed
BATCH_SIZE = 25
MAX_CONCURRENT_API_CALLS = 5
ENABLE_CACHING = True
PHONETIC_STYLE = "simplified"
```

**Command**:
```bash
python tibetan_processor.py -b 25 -p simplified -i input.txt
```

### Example 21: Quality-Focused Configuration

**For Maximum Accuracy**:
```python
# config.py for quality
BATCH_SIZE = 5  # Smaller batches for careful processing
PHONETIC_STYLE = "detailed"
PRESERVE_SANSKRIT_TERMS = True
HANDLE_MANTRAS_SPECIALLY = True
```

**Command**:
```bash
python tibetan_processor.py -b 5 -p detailed -i scholarly_text.txt
```

## Real-World Usage Scenarios

### Example 22: Academic Research

**Scenario**: Processing historical Tibetan manuscripts for linguistic analysis

**Configuration**:
```python
PHONETIC_STYLE = "academic"
PRESERVE_STRUCTURAL_MARKERS = True
TRANSLATION_CONTEXT = "vajrayana_buddhist"
```

**Workflow**:
```bash
# Process manuscript with academic precision
python tibetan_processor.py \
    -i manuscript_wylie.txt \
    -o manuscript_analysis.md \
    --phonetic-style academic \
    --batch-size 8
```

### Example 23: Religious Study

**Scenario**: Processing daily practice texts for practitioners

**Configuration**:
```python
PHONETIC_STYLE = "simplified"  # Easy to read
HANDLE_MANTRAS_SPECIALLY = True
PRESERVE_SANSKRIT_TERMS = True
```

**Workflow**:
```bash
# Process daily practice text
python tibetan_processor.py \
    -i daily_practice.txt \
    -o practice_guide.md \
    --phonetic-style simplified
```

### Example 24: Translation Project

**Scenario**: Creating bilingual editions of Tibetan texts

**Configuration**:
```python
TRANSLATION_CONTEXT = "vajrayana_buddhist"
PHONETIC_STYLE = "detailed"
PRESERVE_STRUCTURAL_MARKERS = True
```

**Workflow**:
```bash
# Process for publication
python tibetan_processor.py \
    -i source_text.txt \
    -o bilingual_edition.md \
    --phonetic-style detailed \
    --batch-size 10
```

## Summary

These examples demonstrate the flexibility and power of the Tibetan Text Processor:

1. **Simple texts** can be processed with default settings
2. **Complex texts** benefit from custom configuration
3. **Large corpora** require memory and performance optimization
4. **Different text types** (ritual, philosophical, mantra) are handled appropriately
5. **Error handling** ensures graceful processing of imperfect input
6. **Batch processing** enables efficient handling of multiple files
7. **Resume capability** allows processing of very large texts over multiple sessions

The key to successful processing is matching the configuration to your specific needs:
- **Speed vs. Quality**: Adjust batch size and phonetic detail
- **Memory vs. Features**: Balance caching and monitoring
- **Accuracy vs. Throughput**: Configure error handling and retry logic

For most users, the default settings provide an excellent balance of speed, accuracy, and reliability.