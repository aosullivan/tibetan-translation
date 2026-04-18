# Tibetan translation toolkit

Process classical Tibetan in **Wylie (EWTS)**: convert to Unicode, optional phonetics, and **English translation** via the Anthropic Claude API.

## Setup

Use a virtual environment and install dependencies:

```bash
cd /path/to/tibetan-translation
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On **Python 3.13+**, if `pyewts` fails to install, use a setuptools version that still provides `pkg_resources`:

```bash
pip install 'setuptools<81'
pip install -r requirements.txt
```

Create a **`.env`** file in the project root (or export in your shell):

```bash
echo 'ANTHROPIC_API_KEY=your_key_here' > .env
```

Translation uses **`TranslationConfig.MODEL`** in `src/config.py` (default: **`claude-sonnet-4-6`**). Change it there if you need another Claude model id.

### Quick run

From the repo root, with venv active and `.env` containing `ANTHROPIC_API_KEY`:

```bash
# Plain Wylie → English (chunked; resumable via progress.json)
python wylie_translate_chunks.py -i tibetan-texts/manjusrikirti.txt -o tibetan-texts/manjusrikirti.en.md

# Pecha-style /-delimited Wylie → markdown with Tibetan + phonetic + English
python pecha_processor.py -i input.txt -o tibetan_output.md
```

Put long-form Wylie exports under **`tibetan-texts/`** (or any path; use `-i` / `-o` as needed). For full flags on any entrypoint:

```bash
python wylie_translate_chunks.py --help
python translate.py --help
python pecha_processor.py --help
python xml_to_wylie.py --help
python pdf_to_wylie.py --help
```

---

## How to run

Run commands from the project root with the venv activated.

### 1. Wylie file → English (chunked, resumable)

Best when you have a **plain UTF-8 Wylie/EWTS** file (for example from `xml_to_wylie.py` or another source). The text is split into chunks, translated sequentially with rolling context, and progress is saved so you can resume after interruption.

```bash
python wylie_translate_chunks.py -i tibetan-texts/manjusrikirti.txt -o tibetan-texts/manjusrikirti.en.md
```

Useful options:

| Option | Meaning |
|--------|---------|
| `-i` / `--input` | Input Wylie file (required) |
| `-o` / `--output` | Output markdown (default: `<input-stem>.translation.md`) |
| `--chunk-size N` | Target characters per chunk before snapping at `/` or space (default: 1500) |
| `--progress PATH` | Resume state (default: `progress.json` in cwd) |
| `--log PATH` | Log file (default: `translation.log`) |
| `-v` | Verbose logging |

Equivalent pipeline with shared defaults: **`translate.py`** (also respects `-i`, `-o`; output path is applied to config before run):

```bash
python translate.py -i your.wylie.txt -o out.md
```

Logs: `translation.log`. Output appends **English only** per chunk (see `src/file_handler.py`).

### 2. Structured Wylie lines → Tibetan + phonetic + English (markdown)

For input that uses **`/`-delimited** lines and structural markers (pecha-style), use the full formatter:

```bash
python pecha_processor.py -i input.txt -o tibetan_output.md
```

Defaults are in **`TibetanConfig`** inside `src/config.py` (`INPUT_FILE`, `OUTPUT_FILE`, etc.). Requires `ANTHROPIC_API_KEY`.

A cached / parallel variant lives in **`src/tibetan_processor_optimized.py`** (run with `python src/tibetan_processor_optimized.py --help` if you use it).

### 3. Tibetan Unicode XML → Wylie

ACIP-style XML with `<page>` and `<line/>`:

```bash
python xml_to_wylie.py -i path/to/text.xml -o out.wylie.txt
```

Omit `-o` to write `<stem>.wylie.txt` next to the XML.

### 4. PDF → Wylie (vision / extraction)

```bash
python pdf_to_wylie.py --help
```

Uses the Anthropic API for image or text extraction (see script for flags).

---

## Input formats (summary)

| Script | Input |
|--------|--------|
| `wylie_translate_chunks.py`, `translate.py` | Plain UTF-8 Wylie/EWTS text file |
| `pecha_processor.py` | Wylie with `/` line delimiters and optional structural markers (see `src/tibetan_line_parser.py`) |
| `xml_to_wylie.py` | ACIP-style XML with Tibetan Unicode in `<page>` elements |

---

## Configuration

Central settings: **`src/config.py`**

- **`TranslationConfig`**: `MODEL`, `CHUNK_SIZE`, default `INPUT_FILE` / `OUTPUT_FILE`, `PROGRESS_FILE` for chunk translation.
- **`TibetanConfig`**: extends the above for pecha processing (paths, batch size, caching, etc.).

---

## Tests and cleanup

```bash
python -m pytest tests/
```

Remove common generated outputs/logs (see `src/clean_outputs.py`):

```bash
python src/clean_outputs.py
```

---

## Project layout (high level)

```
├── wylie_translate_chunks.py   # Wylie → English (chunked)
├── translate.py                 # Same chunk pipeline, simpler CLI
├── pecha_processor.py           # Line-based Wylie → Tibetan + phonetic + translation MD
├── xml_to_wylie.py              # ACIP XML → Wylie
├── pdf_to_wylie.py              # PDF → Wylie
├── src/
│   ├── config.py                # TranslationConfig, TibetanConfig
│   ├── translation_client.py    # Chunk translation + summaries
│   ├── translation_manager.py   # Orchestration, resume
│   ├── file_handler.py          # Chunking, progress, output append
│   ├── wylie_converter.py       # Wylie → Unicode (pyewts)
│   ├── tibetan_translation_client.py  # Vajrayana-oriented API prompts (pecha path)
│   └── …
├── requirements.txt
└── tests/
```

---

## Troubleshooting

**`ANTHROPIC_API_KEY` not set** — Add it to `.env` or your environment; `python-dotenv` loads `.env` from the project root when you run the scripts.

**`ModuleNotFoundError`** — Activate the venv and run `pip install -r requirements.txt`.

**Resume chunk translation** — Re-run the same command; progress is stored in `progress.json` (or your `--progress` path). Delete that file to start from scratch.

---

## License

This project is intended for academic and religious study of Tibetan Buddhist texts. Use respectfully and in line with traditional guidelines for sacred material.
