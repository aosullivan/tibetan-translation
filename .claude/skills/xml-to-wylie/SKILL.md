---
name: xml-to-wylie
description: Convert ACIP-style Tibetan XML (Unicode text inside <page>/<line/> elements, with <info> metadata) to EWTS/Wylie transliteration. Use when the user asks to "render", "convert", or "transliterate" an XML file to Wylie/EWTS, or runs the script on an ACIP/Tohoku-style .xml file in this repo (e.g. DXXXX.xml).
---

# xml-to-wylie

Converts ACIP-style Tibetan XML files to EWTS/Wylie using the project's `xml_to_wylie.py` script (which wraps `pyewts.toWylie()`, matching the reverse direction of `src/wylie_converter.py`).

## When to invoke

The input is XML with this shape: a root `<text>` containing `<info>` (metadata: location, title, author, tohoku ref) and `<pageset>` with `<page>` elements containing Tibetan Unicode and `<line/>` break tags.

Typical filenames: `D3792.xml`, `bstan-'gyur_toh2488(rgyud.zi148).xml`.

## How to run

```bash
python xml_to_wylie.py -i <input.xml> -o <output.wylie.txt>
```

The output convention in this repo is `<basename>.wylie.txt` next to the input.

If `pyewts` is missing, install it: `pip install pyewts` (or use the project's `.venv312/`).

## Output format

- Header block with `# ---  metadata ---` lines pulled from `<info>` (Tohoku number, organization/collection/volume, title, author, section, status).
- `### page index=N id=X` markers between pages.
- `<line/>` becomes a newline within each page.
- Body is EWTS Wylie (e.g. `bcom ldan 'das`, `phyag 'tshal lo`).

## Notes

- The script is read-only on the input; safe to re-run.
- Don't hand-roll a converter — always shell out to `xml_to_wylie.py` so output matches the rest of the pipeline (`pdf_to_wylie.py`, `wylie_translate_chunks.py`, etc.).
- For PDFs use [pdf_to_wylie.py](../../../pdf_to_wylie.py) instead; for pecha-format scans see [pecha_processor.py](../../../pecha_processor.py).
