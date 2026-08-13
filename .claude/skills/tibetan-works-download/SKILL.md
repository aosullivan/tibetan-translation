---
name: tibetan-works-download
description: Generate paste-ready browser-console JavaScript to download canonical texts (Kangyur/Tengyur) as XML from tibetan.works, then convert to EWTS/Wylie. Use when the user wants to fetch, download, or grab source texts from tibetan.works, or asks for "the JS / snippet to download <texts> from the site".
---

# tibetan-works-download

When invoked, **do this interactively** — don't just dump a static file:

## 1. Ask what to download
Find out:
- **Which collection** — `kangyur` (Lhasa "H" numbering) or `tengyur` (Derge "Toh"). Ask if unclear.
- **Which texts** — a corpus or explicit list. Accept Toh numbers / Sanskrit / English names. If the user names a cycle ("the Mañjuśrī corpus"), propose the list (Toh + Sanskrit + Tibetan title) and let them trim. Use AskUserQuestion for scope when it's ambiguous.

## 2. Build the `targets` and hand back the JS
For each requested text, produce a target `{ toh, sanskrit, tib }` where:
- `tib` is a **distinctive Tibetan title substring** (matching is by title, never by number — see below). Make it specific enough to avoid false hits; include `འཇམ་དཔལ` etc. when a bare phrase like "108 names" would collide with other deities' texts.
- `sanskrit` is a short ASCII slug used in the filename.

Take [`xml-fetch.template.js`](xml-fetch.template.js), fill its `COLLECTION` and `targets` block, and return the **complete filled snippet in a code block**. Keep the streaming-peek logic intact.

## 3. Tell the user exactly where/how to run it
Always state:
1. Open a **https://tibetan.works/** page (any) — the fetch is same-origin (`credentials:'include'`); it is CORS-blocked from any other tab.
2. **Reload** the tab first if they've run earlier versions (stray loops keep scanning).
3. Open the console (Cmd-Opt-J Chrome / Cmd-Opt-K Firefox). On Chrome's first paste, type `allow pasting` (Enter) then paste.
4. Files download to `~/Downloads`. The final `console.table` is the index→title→file map; anything under `NOT FOUND` needs its `tib` hook fixed or the range widened (have them read the real title off the reader and adjust).

## 4. Convert and file
When the `.xml` files land in `~/Downloads`, run the repo's converter (real `<pageset>` XML, folio markers intact):
```bash
.venv312/bin/python xml_to_wylie.py -i "~/Downloads/<file>.xml" -o "tibetan-texts/<name>.wylie.txt"
```
Then copy into the catalogue, naming by Toh (matched via each file's `<info><title>`):
`~/MEGA/MEGAsync/texts/Dispelling Error/other-sources/<corpus>-toh###-<name>.wylie.txt`.

## How the fetch works (don't re-derive)
- `reader.php?collection=<c>&index=<N>` returns a JS shell **with the full `<pageset>` XML embedded** in `<div class='text'>`. The snippet streams the first ~6 KB to read `<info><title>` cheaply, and on a title match fetches the whole page and slices `<text>…</text>`.
- **`index` is the Lhasa H / Derge global number, NOT the Toh number** (e.g. index 543 = H0543 = Sitātapatrā, not Toh 543). Always match by Tibetan title.
- The page's own download form gives the canonical filename (`bka'-'gyur_H####(rgyud.xxNNN).xml`); the snippet reuses it.
- Source-script typos (e.g. `'khol lo` for `'khor lo`) pass through unchanged — never normalize; flag for proofreading. See memory [[tibetan-works-download-skill]] and [[vimalaprabha-source]].
