---
name: vellum-continuous-endnotes
description: >-
  Renumber a Vellum print PDF's endnotes continuously (1, 2, 3… through
  the whole book), drop the per-chapter Notes subheadings, reflow the
  notes so pages fill, and update every body superscript to match. Use
  whenever the user asks to make Vellum footnotes/endnotes continuous,
  stop notes restarting at 1 each chapter, remove Notes chapter
  headings, reflow notes, or post-process a Vellum Print PDF's notes —
  including Dispelling Error and any later Vellum book with the same
  limitation. Also use after a fresh Vellum Print export when finishing
  the notes is part of the pipeline.
---

# Continuous Vellum endnotes

Vellum restarts endnote numbering at every chapter and groups the Notes
under chapter subheadings. That cannot be changed in the `.docx` or the
Vellum document. Post-process the **finished Print PDF** instead.

Do **not** rebuild or reimport via `build.sh`. The Vellum document is
the text; this skill only rewrites numbers, drops headings, and reflows
already-typeset note lines.

## Pipeline

1. Export **Print** from Vellum.
2. Run any book-specific PDF finish first (for *Dispelling Error*,
   `fix_vellum_export.py` — chapter numbers, Contents, Appendix V, seal).
   Keep that file. It is the source; never overwrite it.
3. Then this skill:

```bash
python3 <skill-or-vellum>/renumber_notes_continuous.py <final.pdf> -o <out.pdf>
python3 <skill-or-vellum>/validate_continuous_notes.py <final.pdf> <out.pdf> \
    --audit <audit.csv>
```

Needs `pymupdf` (`pip install pymupdf --break-system-packages`).

On *Dispelling Error* the scripts already live in `vellum/` next to
`fix_vellum_export.py`. Use those. Elsewhere, use the copies bundled in
this skill's `scripts/` directory (they import `pdf_tokens.py` when
`fix_vellum_export.py` is absent).

Write the output next to the source under a new name
(`…-continuous-notes.pdf`). Saving goes through a scratch file: MEGA-
and Dropbox-synced folders refuse an in-place overwrite.

## What the script does

- Finds the Notes from the page headed `NOTES` to the last page that
  still has text. A trailing blank page is left alone.
- Drops Palatino-Bold 10pt chapter subheadings inside the Notes.
- Renumbers every note `1…N` in reading order, right-aligned on Vellum's
  hanging-number edge so wrapped lines do not move.
- Reflows the remaining body lines across the Notes pages, filling each
  page from that book's top baseline down to its bottom baseline, in
  whichever text column (recto / verso) the line lands in. A note's
  opening line is never left alone at a page foot.
- Deletes Notes pages the reflow empties. Body pagination does not
  change; the Contents entry for Notes stays valid.
- Renumbers every Times 7pt superscript in the body to match, shifting
  only that marker's own line by the extra digit width.
- Rewrites in-note cross-references (`See note 12 in The Second Root
  Downfall`) via `XREF_TARGET` in the script.

Nothing is re-typeset: original TJ arrays, fonts, and kerning survive.
Only digit bytes and chunk positions move.

## Validate, then look

The validator must exit 0. It checks: marker ↔ note correspondence and
sequence; no lost lines (minus the dropped headings); no new overlaps;
Notes pages filled to the foot (except the last); body lines unmoved
vertically; every number actually inks when rasterised.

Then render the Notes pages and look at them. The validator cannot see a
wrong glyph that still inks, or a page that "looks empty" because of a
heading gap — that last one is why reflow exists. Spot-check the audit
CSV (new number · old section · old number · body context · note opening).

## Traps — do not re-learn these

These all bit *Dispelling Error* and will bite the next Vellum book.

- **Do not close up headings only within each page.** That strands the
  saved space at the page foot (a "massive gap, then two blank pages").
  Reflow across page boundaries.
- **The PDF `'` operator** (next-line-and-show) opens lines that start
  with a special italic glyph. A scanner that ignores it reprints that
  line on top of the one above. Use `scan` / `scan2` from these scripts,
  not a homemade walker.
- **Group lines by baseline**, not only by the scanner's newline flag.
  A no-op `0 0 Td` after a line-breaking `TD` clears the flag and merges
  a wrapped line into the one above.
- **Font subsets are not interchangeable.** Two subsets both named
  `Athelas-Italic` encode special glyphs differently. Carry each chunk's
  **font object xref**, and add missing xrefs to a destination page by
  rewriting `Resources/Font` in **one** `xref_set_key`. Per-key
  `Resources/Font/FA0` writes corrupt the dict into an unnamed nest and
  the fonts do not resolve.
- **Each page's Times is subset to the digits that page originally
  used.** A new digit (a `5` on a page that only had `1`) extracts as
  text and prints as nothing. Re-point markers at a complete Times New
  Roman inserted with `insert_font(..., set_simple=1, encoding=1)`.
  Without `set_simple` the Type0/Identity-H font maps `(5)` to NUL.
- **Body text is recognised by size, not family.** An Arial `→` sharing
  an 8.4pt notes line must travel with it; a family test drops it.
- **A same-line marker that grows a digit** shifts the following `Tm`
  right by one tabular Times digit (3.5pt at 7pt). Compensate the
  **first line-break after it** (`TD`/`Td` with `ty ≠ 0`, or `T*`) or
  the whole block creeps right, into the next marker and past the
  margin. Same-line detection must allow the continuation to start up
  to ~1.5pt inside the old digit.
- **Do not add or remove blank pages at the front of the book.** Vellum
  Print already pairs half-title / title / Contents / Foreword / part
  page with blank versos. Compare page 1–12 of the raw export against
  the finished file if the front "looks off"; those blanks are original.

## Adapting to another Vellum book

The scripts assume Vellum's usual Notes styling:

| What | Dispelling Error value | How to re-measure |
|---|---|---|
| Notes body size | 8.4pt | `span["size"]` on a Notes line |
| Notes leading | 10.5pt | baseline difference of two consecutive lines |
| Heading face | Palatino-Bold 10pt | the chapter titles inside Notes |
| Marker face | Times 7pt | superscripts in the body (not Contents) |
| Digit advances | `ATHELAS_DIGIT` | rawdict bboxes, one digit at a time |
| Recto / verso column | 79.8 / 52.8pt | left edge of a notes line on facing pages |
| Top / bottom baselines | 418.75 / 518.5 / 51.25 | first line under `NOTES`; first line of a later Notes page; last line of a full Notes page |

Rebuild `XREF_TARGET` from the book's own wording (`See note N in
<chapter>`). Leave it empty if there are no cross-references; the
script will assert if it meets one it cannot map.

If Notes headings or sizes differ, fix the detectors in
`page_helpers` / `is_heading` / `is_marker` rather than rewriting the
reflow.

## Dispelling Error paths

Project root: the MEGA folder that holds `translation/`, `source/`, and
`vellum/`. Print PDFs:

`vellum/Dispelling Error - A commentary clarifying the fourteen root downfalls/Print/`

Typical names:

- `Dispelling-Error-Print.pdf` — raw Vellum export (do not edit)
- `Dispelling-Error-Print-final-oct.pdf` — after `fix_vellum_export.py`
- `Dispelling-Error-Print-continuous-notes-oct.pdf` — after this skill

The book skill `dispelling-error-vellum` covers Stages 1–3 (Word build,
the four PDF finishes, verification). This skill is Stage 4.
