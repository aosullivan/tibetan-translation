# Tibetan Text Translation Tool 🕉️

An automated tool for translating classical Tibetan texts to English using the Anthropic Claude API.

## How to run

- Clear out any previously generated files with `clean_output.py`
- Paste Tibetan text into `input.txt`
- Run `translation.py`

## Setup

1. Install dependencies:
```bash
pip install anthropic python-dotenv
```

2. Create a `.env` file with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_key_here
```

3. Configure settings in `config.py` if needed:
- Chunk size
- Output format
- Model selection
- File paths

## Usage

Basic usage:
```bash
python translate.py -i input.txt -o output.md
```

Options:
- `-i, --input`: Input file path (default: specified in config)
- `-o, --output`: Output file path (default: specified in config)

## Input Format

- Input should be Unicode Tibetan text
- Sentences should be separated by the Tibetan sentence marker (།)
- The tool handles word wrapping and chunking automatically

## Output Format

Markdown (default):
```markdown
# Tibetan Text Translation

[Translation content]

---

[Next section]
```

Or plain text (configure in `config.py`):
```
[Translation content]

[Next section]
```

## Logging

Logs are written to `translation.log` and include:
- Translation progress
- Chunk boundaries
- API responses
- Error messages

## Error Handling

- Automatically retries on rate limits
- Saves progress on interruption
- Detailed error logging
- Resumable from last successful chunk
