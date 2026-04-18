# Tibetan translation toolkit

A set of Python utilities for working with classical Tibetan texts end-to-end:

1. **Ingest** from PDF, ACIP-style XML, or plain Wylie/EWTS files
2. **Convert** between Wylie and Unicode Tibetan (and generate phonetics)
3. **Translate** to English via the Anthropic Claude API, with chunking, rolling context, and resumable progress

Everything is driven from `ANTHROPIC_API_KEY` and the configuration in [`src/config.py`](src/config.py).

---

## Setup

Use a virtual environment and install dependencies:

```bash
cd /path/to/tibetan-translation
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On **Python 3.13+**, if `pyewts` fails to install, pin a setuptools version that still ships `pkg_resources`:

```bash
pip install 'setuptools<81'
pip install -r requirements.txt
```

Create a `.env` file in the project root (loaded automatically via `python-dotenv`):

```bash
echo 'ANTHROPIC_API_KEY=your_key_here' > .env
```

**Model and chunk-size defaults** live in [`src/config.py`](src/config.py):

- `MODEL = "claude-sonnet-4-6"`
- `CHUNK_SIZE = 2250` (characters per translation chunk, snapping to `/` or whitespace)

---

## Scripts at a glance

| Script | What it does | Input | Output |
|---|---|---|---|
| [`pdf_to_wylie.py`](pdf_to_wylie.py) | Extract Wylie from a PDF (pdfplumber → PyPDF2 → Claude Vision OCR fallback) | `.pdf` | `.wylie.txt` |
| [`xml_to_wylie.py`](xml_to_wylie.py) | Convert ACIP-style Tibetan Unicode XML to Wylie/EWTS | `.xml` | `.wylie.txt` |
| [`wylie_translate_chunks.py`](wylie_translate_chunks.py) | Chunked, resumable Wylie → English translation | `.wylie.txt` / any Wylie `.txt` | `.md` |
| [`translate.py`](translate.py) | Same pipeline as above, minimal CLI (uses config defaults) | Wylie `.txt` | `.md` |
| [`pecha_processor.py`](pecha_processor.py) | Line-by-line pecha processor: Wylie → Tibetan Unicode + phonetic + English, rendered to markdown | `/`-delimited Wylie | Markdown with all three layers |
| [`src/tibetan_processor_optimized.py`](src/tibetan_processor_optimized.py) | Alternate pecha pipeline with caching, batching, and concurrency (see `TibetanConfig`) | Same as `pecha_processor.py` | Markdown |
| [`src/repair_translation_from_log.py`](src/repair_translation_from_log.py) | Rebuild a truncated/interrupted translation by replaying `Raw response:` blocks from `translation.log`; only calls the API for chunks missing from the log | Wylie input + log(s) | Markdown |
| [`src/clean_outputs.py`](src/clean_outputs.py) | Remove `progress.json`, `translation.log`, the default output, and any `*.translated` files | — | — |

Every script supports `--help`.

---

## Typical workflows

Run everything from the repo root with the venv active.

### A. PDF → Wylie → English

```bash
# 1. Extract Wylie from a scanned or digital PDF
python pdf_to_wylie.py -i W1KG9245-v2_img175-224.pdf -o W1KG9245-v2_img175-224.wylie.txt

# 2. Translate to English (chunked, resumable)
python wylie_translate_chunks.py \
    -i W1KG9245-v2_img175-224.wylie.txt \
    -o W1KG9245-v2_img175-224.en.md
```

`pdf_to_wylie.py` tries `pdfplumber` first, falls back to `PyPDF2`, and finally uses **Claude Vision** (`claude-sonnet-4-6`) for image-based pages — which requires `ANTHROPIC_API_KEY`. Images are auto-compressed to stay under the 5 MB API limit.

### B. ACIP XML → Wylie → English

```bash
python xml_to_wylie.py -i "bstan-'gyur_toh2488(rgyud.zi148).xml" -o toh2488.wylie.txt
python wylie_translate_chunks.py -i toh2488.wylie.txt -o toh2488.en.md
```

`xml_to_wylie.py` reads `<pageset><page><line/></page></pageset>`, converts each page's Tibetan Unicode to Wylie via `pyewts`, and emits page-header comments (`### page index=... id=...`). Use `--no-metadata` or `--no-page-headers` to suppress them.

### C. Structured pecha (Wylie + phonetic + English in one doc)

For line-based `/`-delimited Wylie:

```bash
python pecha_processor.py -i phakmo-sadhana.txt -o phakmo-sadhana.md
```

Defaults come from [`TibetanConfig`](src/config.py) (input `saras-tibetan.txt`, output `tibetan_output.md`). The output interleaves **Tibetan Unicode + Roman phonetics + English translation** per line — useful for recitation or study editions.

Prefer the optimized variant when running on long texts; it adds caching, batching, and bounded concurrency:

```bash
python -m src.tibetan_processor_optimized -i phakmo-sadhana.txt -o phakmo-sadhana.md
```

#### What "pecha" means here

A **pecha** (Tib. དཔེ་ཆ་, *dpe cha*) is the traditional Tibetan book format: loose unbound horizontal folios stacked between two wooden covers. There's no sentence case, no paragraph breaks — just continuous text punctuated by the shad (`།`) and occasional folio/section markers. Most canonical Tibetan Buddhist literature (Kangyur, Tengyur, sadhanas, commentaries) still lives in this format; digital editions from archives like **BDRC** and **ACIP** preserve its shape.

`pecha_processor.py` doesn't care about the physical pecha — it cares about the **digital convention for encoding pechas in Wylie**: short `/`-delimited phrases with ACIP/BDRC-style annotation markers. The parser at [`src/tibetan_line_parser.py:51`](src/tibetan_line_parser.py:51) splits on a specific set of delimiter tokens (`////`, `//_!_`, `/[ ]/`, `//`, `/_`, `*/_`, `_`) and recognizes structural markers like `[...]`, `(kha)`, and `!_..._!_` as folio/section annotations. It then coalesces fragments into chunks of ~12 Wylie tokens so each line is a natural syntactic unit.

That shape matters because this pipeline's job is to produce a **three-line stacked display** per segment — Tibetan Unicode, Wylie, English — suitable for recitation, study editions, or interlinear publication. That only works if:

1. **Segments are short and meaningful.** A pecha "line" is typically one clause or verse line. Feed it flat prose with no `/` delimiters and the whole file becomes one giant segment.
2. **Folio markers are preserved.** Canonical texts mark page boundaries inline (`[ ]`, `(kha)`); the parser pulls them out as standalone structural lines.
3. **Each segment translates independently.** The pecha client uses Vajrayana-aware prompts per line, not a rolling chunk pipeline — short self-contained units are the assumption.

**Which script should I use?**

| Your input | Use |
|---|---|
| Running Wylie prose, essay, commentary — anything flowing | `wylie_translate_chunks.py` |
| `/`-delimited Wylie from a digital pecha edition (BDRC, ACIP, or hand-transcribed sadhana) | `pecha_processor.py` |
| You want an **interlinear** study edition with Tibetan script + phonetics + English | `pecha_processor.py` (only it produces the three-layer output) |
| You just want English translation | `wylie_translate_chunks.py` (simpler, resumable, no per-line overhead) |

### D. Repairing an interrupted translation

If a long run was killed but `translation.log` is intact, you can rebuild the output mostly (or entirely) from logs without re-calling the API:

```bash
# Dry run — see which chunk indices are missing from the log
python -m src.repair_translation_from_log \
    -i toh2488.wylie.txt -o toh2488.en.md \
    --log translation.log --dry-run --report repair-report.json

# Apply — fills any gaps via the API, otherwise reuses logged responses
python -m src.repair_translation_from_log \
    -i toh2488.wylie.txt -o toh2488.en.md \
    --log translation.log
```

Pass `--log` multiple times to merge several log files in order (later entries win per chunk index).

---

## Flags worth knowing

### `wylie_translate_chunks.py`

| Flag | Meaning |
|---|---|
| `-i` / `--input` | Input Wylie/EWTS file (required) |
| `-o` / `--output` | Output markdown (default: `<input-stem>.translation.md`) |
| `--chunk-size N` | Target chars per chunk (default: **2250**) |
| `--progress PATH` | Resume state (default: `progress.json` in cwd) |
| `--log PATH` | Log file path (default: `translation.log`) |
| `--outline PATH` | Markdown outline file (sa bcad) for structure-aware headings — see below |
| `-v` | Verbose logging |

#### Outline-aware translation (`--outline`)

For scholastic texts with a topical outline (*sa bcad*), pass a markdown file listing the section hierarchy and the translator will emit matching markdown headings verbatim at the right points, with a cursor that advances across chunks and survives resume (persisted in `progress.json` under `_meta`).

The outline file can use any mix of `#`-headings and `**bold**` entries (indentation and fenced code blocks are fine; `← STATUS` and `(parenthetical)` annotations are stripped). Each entry becomes a flat line the model must emit verbatim when it finds a semantic match in the Tibetan — otherwise in-paragraph enumerations (`dang po ni`, `gnyis pa ni`, …) remain as ordinary prose or numbered lists. Ambiguous cases get flagged inline as `[?editor: possible new section — candidate "X"]`.

Example:

```bash
python wylie_translate_chunks.py \
    -i dispelling-error.wylie.txt \
    -o dispelling-error.en.md \
    --outline Structure.md
```

See [`src/outline.py`](src/outline.py) for the parser. The feature is opt-in — without `--outline` the translator behaves exactly as before.

### `pdf_to_wylie.py`

| Flag | Meaning |
|---|---|
| `-i` / `--input` | Path to `.pdf` (required) |
| `-o` / `--output` | Output path (default: `<stem>.wylie.txt`) |
| `--extract-only` | Skip any post-processing; write raw extracted text |

### `xml_to_wylie.py`

| Flag | Meaning |
|---|---|
| `-i` / `--input` | Path to `.xml` (required) |
| `-o` / `--output` | Output path (default: `<stem>.wylie.txt`) |
| `--no-metadata` | Omit converted `<info>` comment header |
| `--no-page-headers` | Omit `### page index=... id=...` lines between pages |

---

## Configuration

All configurable defaults are in [`src/config.py`](src/config.py):

- **`TranslationConfig`** — used by `translate.py`, `wylie_translate_chunks.py`, and the repair script. Fields: `MODEL`, `CHUNK_SIZE`, `SUMMARY_INTERVAL`, `MAX_RETRIES`, `MAX_TOKENS`, `INPUT_FILE`, `OUTPUT_FILE`, `PROGRESS_FILE`, `OUTPUT_FORMAT`.
- **`TibetanConfig`** — extends the above for the pecha pipeline. Adds Wylie delimiters, Sanskrit preservation, mantra handling, caching, batch size, concurrency limits, auto-save intervals, and API timeout/retry knobs.

> **Note:** `max_tokens` is currently hardcoded (1000) inside `TranslationClient.translate_chunk`. With the larger chunk default it may truncate longer outputs — tracked as a known improvement.

---

## Project layout

```
├── pdf_to_wylie.py                     # PDF → Wylie (text + OCR)
├── xml_to_wylie.py                     # ACIP XML → Wylie
├── wylie_translate_chunks.py           # Wylie → English (chunked, resumable)
├── translate.py                        # Thin wrapper over the same pipeline
├── pecha_processor.py                  # Line-based Wylie → Tibetan + phonetic + English
├── src/
│   ├── config.py                       # TranslationConfig, TibetanConfig
│   ├── translation_client.py           # Chunk translation + summaries (Claude)
│   ├── translation_manager.py          # Orchestration, rolling summary, resume
│   ├── file_handler.py                 # Chunking, progress, output append
│   ├── tibetan_translation_client.py   # Vajrayana-aware prompts for pecha pipeline
│   ├── tibetan_file_handler.py         # Pecha-style IO
│   ├── tibetan_line_parser.py          # Parse `/`-delimited Wylie lines
│   ├── tibetan_output_formatter.py     # Markdown layout for pecha output
│   ├── tibetan_processor_optimized.py  # Cached/concurrent pecha pipeline
│   ├── tibetan_models.py               # Dataclasses shared across pecha code
│   ├── wylie_converter.py              # Wylie ↔ Unicode (pyewts)
│   ├── phonetic_renderer.py            # Romanized phonetics
│   ├── repair_translation_from_log.py  # Rebuild from translation.log
│   └── clean_outputs.py                # Remove logs / progress / outputs
└── tests/                              # pytest suite (+ tests/fixtures/)
```

---

## Tests

```bash
python -m pytest tests/
```

Fixtures live in [`tests/fixtures/`](tests/fixtures/) and are checked in; generated artifacts are not.

---

## Troubleshooting

**`ANTHROPIC_API_KEY` not set** — add it to `.env` or export it in your shell; scripts load `.env` from the project root.

**`ModuleNotFoundError: pyewts` / `pdfplumber`** — make sure the venv is activated and `pip install -r requirements.txt` completed cleanly (see Python 3.13 note above).

**Resume a translation** — re-run the same command; progress is stored in `progress.json` (or your `--progress` path). Delete it to start fresh. For a run that exited badly but wrote a log, use `src/repair_translation_from_log.py` instead of re-translating.

**Image-heavy PDFs fail to extract** — `pdf_to_wylie.py` will fall back to Claude Vision OCR automatically when `pdfplumber` and `PyPDF2` return no text. Ensure `ANTHROPIC_API_KEY` is set and you're on a plan with Vision access.

---

## Notes on repo hygiene

The repository tracks **code and test fixtures only**. PDFs, `.wylie.txt` outputs, translation markdown, progress JSONs, logs, and any per-run input corpora are `.gitignore`d — see [`.gitignore`](.gitignore).

---

## License

This project is intended for academic and religious study of Tibetan Buddhist texts. Use respectfully and in line with traditional guidelines for sacred material.
