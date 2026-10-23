#!/usr/bin/env python3
"""Renumber the endnotes of *Dispelling Error* continuously (1..60).

The printed book restarts endnote numbering at every chapter and groups
the notes under chapter subheadings.  This post-processes the finished
print PDF (the file produced by fix_vellum_export.py) so that:

  * the chapter subheadings inside the Notes are removed, and the notes
    are reflowed line by line across the Notes pages (filling each
    page as the book itself fills it, in whichever text column --
    recto or verso -- the line lands in), so no page is left with a
    stranded gap at its foot; pages the reflow empties are deleted;
  * every endnote is renumbered 1..60 in order, right-aligned on the
    same fixed right edge Vellum used, so the hanging indent (and every
    wrapped line) stays exactly where it was;
  * every superscript marker in the body is renumbered to match, with
    any text following it on the same line shifted right by exactly the
    extra digit width (Times digits are tabular, 3.5pt at 7pt);
  * cross-references inside the notes ("See note 12 in The Second Root
    Downfall") are renumbered too.

Nothing is re-typeset: original TJ arrays, fonts and kerning survive
byte-for-byte; only digit bytes and chunk positions are touched.
Body pagination is unchanged; the Notes may end a page or two
earlier than in the original, and those emptied pages are removed.

Usage:
    python3 renumber_notes_continuous.py <final.pdf> -o <out.pdf>
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from fix_vellum_export import tokenize, rebuild, Chunk, OPERATOR  # noqa: E402
except ImportError:
    from pdf_tokens import tokenize, rebuild, Chunk, OPERATOR  # noqa: E402


def scan2(tokens):
    """Like fix_vellum_export.scan, but also honours the ' and "
    operators (move to next line, then show the string).

    Vellum emits a line that begins with a special glyph (e.g. the
    *Ā* opening a wrapped line inside a note) as `(...)', and scan()
    treats that operator as inert preamble -- so on rebuild the line
    loses its break and prints on top of the line above.  Every Notes
    page rebuilt through scan() alone is liable to this; handling the
    operator here is the whole difference.
    """
    chunks, pre, operands = [], [], []
    scale = 1.0
    line_x = line_y = cur_x = cur_y = leading = 0.0
    font = None
    newline = True
    in_text = False
    for t in tokens:
        if not OPERATOR.match(t) or t.startswith(b"/"):
            operands.append(t)
            continue
        if t == b"BT":
            in_text, newline = True, True
            pre.append(t)
        elif t == b"ET":
            in_text = False
            pre.append(t)
        elif in_text and t == b"Tm":
            a, _, _, _, e, f = (float(v) for v in operands[-6:])
            scale = a
            line_x = cur_x = e
            line_y = cur_y = f
            newline = True
        elif in_text and t in (b"Td", b"TD"):
            tx, ty = float(operands[-2]), float(operands[-1])
            if t == b"TD":
                leading = -ty
            line_x += tx * scale
            line_y += ty * scale
            cur_x, cur_y = line_x, line_y
            newline = (ty != 0)
        elif in_text and t == b"T*":
            line_y -= leading * scale
            cur_x, cur_y = line_x, line_y
            newline = True
        elif in_text and t == b"TL":
            leading = float(operands[-1])
        elif in_text and t == b"Tf":
            font = (operands[-2], operands[-1])
        elif in_text and t in (b"TJ", b"Tj"):
            chunks.append(Chunk(font, scale, cur_x, cur_y,
                                (operands[-1], t), newline, pre))
            pre = []
            newline = False
        elif in_text and t in (b"'", b'"'):
            line_y -= leading * scale
            cur_x, cur_y = line_x, line_y
            chunks.append(Chunk(font, scale, cur_x, cur_y,
                                (operands[-1], b"Tj"), True, pre))
            pre = []
            newline = False
        else:
            pre += operands + [t]
        operands = []
    return chunks, pre + operands

# Athelas digit advances at 8.4pt, measured char-by-char from the book
# itself (rawdict bboxes on the Notes pages; digits are proportional).
ATHELAS_DIGIT = {"0": 4.49, "1": 2.71, "2": 3.66, "3": 3.47, "4": 4.04,
                 "5": 3.46, "6": 4.03, "7": 3.37, "8": 4.02, "9": 4.03}
TIMES_DIGIT = 3.5          # Times New Roman digits are tabular (0.5em)
BODY_LINE = 10.5           # notes line spacing: 1.25 x 8.4

# Old (section, number) -> new continuous number, for cross-references
# inside the notes.  Sections are identified by the title that follows
# the reference, which is how the book itself words them.
XREF_TARGET = {"brief": {2: 6}, "second": {6: 19, 12: 25}, "intro": {2: 2}}


def ath_width(s):
    return sum(ATHELAS_DIGIT[d] for d in s)


def literals(token):
    """Concatenated literal-string contents of a TJ/Tj token."""
    parts = re.findall(rb"\((?:\\.|[^()\\])*\)", token)
    out = b"".join(p[1:-1] for p in parts)
    return out.replace(rb"\(", b"(").replace(rb"\)", b")")


def notes_pages(doc):
    """0-based pages of the Notes: from the page headed NOTES until the
    notes run out.  Trailing blank pages are not Notes pages and are
    left alone (the book's final blank page survives any reflow)."""
    start = None
    for pno in range(len(doc)):
        text = doc[pno].get_text()
        if start is None and re.search(r"^NOTES$", text, re.M):
            start = pno
    if start is None:
        return []
    pages = list(range(start, len(doc)))
    while pages and not doc[pages[-1]].get_text().strip():
        pages.pop()
    return pages


def font_bases(page):
    """Resource name (b'/F40') -> base font name, for one page."""
    out = {}
    for info in page.get_fonts(full=True):
        out[b"/" + info[4].encode()] = info[3]
    return out


NOTE_START = re.compile(rb"^(\[\()(\d{1,2})\.?\)|^(\()(\d{1,2})\.?\)")


def page_helpers(doc, pno):
    page = doc[pno]
    bases = font_bases(page)
    chunks, tail = scan2(tokenize(b"".join(doc.xref_stream(x)
                                           for x in page.get_contents())))

    def base(c):
        return bases.get(c.font[0], "") if c and c.font else ""

    def is_body(c):
        # notes text is the 8.4pt matter, whatever face it is in:
        # an inline Arial arrow shares a line with the Athelas, and
        # a family test would drop it from the line altogether
        return abs(c.scale - 8.4) < 0.2

    def is_heading(c):
        return (abs(c.scale - 10.0) < 0.2 and "Palatino-Bold" in base(c))

    return page, bases, chunks, tail, base, is_body, is_heading


def collect_notes_lines(doc, npages, state):
    """Every Notes body line in the book, in order, as a list of chunk
    lists, with headings dropped and numbers/cross-references already
    renumbered.  Each line remembers its source page (for the text
    column it was set in) and whether it opens a note."""
    lines = []
    for pno in npages:
        _, bases, chunks, _, _, is_body, is_heading = \
            page_helpers(doc, pno)
        xref_by_name = {b"/" + info[4].encode(): info[0]
                        for info in doc[pno].get_fonts(full=True)}
        for i, c in enumerate(chunks):
            if is_heading(c):
                state["section_old"] = 0
                continue
            if not is_body(c):
                continue
            # carry the exact font object, not this page's resource
            # name and not the family: two subsets both called
            # Athelas-Italic encode their special glyphs differently,
            # so a chunk set in one subset prints (and extracts as)
            # wrong letters in another
            c.font = (b"X%d" % xref_by_name[c.font[0]], c.font[1])

            # cross-reference inside a note?
            text = literals(c.op[0]).decode("latin1", "ignore")
            m = re.search(r"note\s+(\d+)\s+(in|to)\b", text)
            if m:
                old = int(m.group(1))
                following = b" ".join(
                    literals(ch.op[0]) for ch in chunks[i + 1:i + 4]
                ).decode("latin1", "ignore")
                if m.group(2) == "to":
                    target = "intro"
                elif "brief" in following:
                    target = "brief"
                else:
                    target = "second"
                new = XREF_TARGET[target][old]
                if new != old:
                    tok = re.sub(rb"(e |e\)|\()\s*%d(?=[\s\)])" % old,
                                 lambda mm: mm.group(0)[:-len(str(old))]
                                 + str(new).encode(),
                                 c.op[0], count=1)
                    if tok == c.op[0]:    # "(e 2 in )" style
                        tok = re.sub(rb" %d " % old,
                                     b" %d " % new, c.op[0], count=1)
                    delta = ath_width(str(new)) - ath_width(str(old))
                    c.op = (tok, c.op[1])
                    state["xrefs"].append((pno + 1, old, new, target))
                    if delta:
                        # widen the rest of this visual line
                        for ch in chunks[i + 1:]:
                            if ch.newline:
                                break
                            if is_body(ch):
                                ch.x += delta

            # a note's own number?
            starts = False
            if c.newline and NOTE_START.match(c.op[0]):
                m = NOTE_START.match(c.op[0])
                old = int(m.group(2) or m.group(4))
                state["section_old"] += 1
                assert old == state["section_old"], (
                    f"p{pno + 1}: note number {old}, expected "
                    f"{state['section_old']} in its section")
                state["global"] += 1
                new = state["global"]
                if m.group(2):
                    tok = c.op[0][:m.start(2)] + str(new).encode() \
                        + c.op[0][m.end(2):]
                else:
                    tok = c.op[0][:m.start(4)] + str(new).encode() \
                        + c.op[0][m.end(4):]
                c.op = (tok, c.op[1])
                # right-align on Vellum's fixed edge: the text stays,
                # the number grows leftwards into the margin
                c.x -= ath_width(str(new)) - ath_width(str(old))
                state["notes"].append((pno + 1, old, new))
                starts = True

            # group by baseline, not by scan's newline flag alone:
            # a no-op `0 0 Td` after a line-breaking TD clears that
            # flag, which would merge a wrapped line into the line
            # above it and print the two on top of each other
            if (not lines or c.newline
                    or abs(c.y - lines[-1]["y"]) > 0.01):
                lines.append({"chunks": [], "src": pno, "y": c.y,
                              "starts_note": starts})
            lines[-1]["chunks"].append(c)
            lines[-1]["starts_note"] = (lines[-1]["starts_note"]
                                        or starts)
    return lines


TOP_FIRST = 418.75         # first Notes page: first line under the title
TOP_PAGE = 518.5           # later Notes pages: top body baseline
BOTTOM = 51.25             # lowest body baseline the book itself uses


def write_notes_pages(doc, npages, lines):
    """Reflow the collected lines over the Notes pages, filling each
    page to the bottom the book uses, then delete the pages the
    reflow leaves empty.  Returns how many Notes pages were used."""
    start = npages[0]

    def column(pno):
        return 79.8 if (pno - start) % 2 == 0 else 52.8

    def font_plan(pno, take):
        """Map every font xref the page's new lines need to a resource
        name on this page, adding missing fonts to the page's Font
        dictionary in one operation.  (Adding them one key at a time
        with xref_set_key("Resources/Font/FA0", ...) corrupts the
        dictionary into an unnamed nested dict, and the added fonts
        then do not resolve at all.)"""
        existing = {info[0]: b"/" + info[4].encode()
                    for info in doc[pno].get_fonts(full=True)}
        fontmap, additions = {}, []
        for ln in take:
            for c in ln["chunks"]:
                xref = int(c.font[0][1:])
                if xref in fontmap:
                    continue
                if xref in existing:
                    fontmap[xref] = existing[xref]
                else:
                    name = b"/FA%d" % len(additions)
                    fontmap[xref] = name
                    additions.append((name, xref))
        if additions:
            kind, fd = doc.xref_get_key(doc[pno].xref,
                                         "Resources/Font")
            assert kind == "dict" and fd.rstrip().endswith(">>")
            fd = fd.rstrip()[:-2] + "".join(
                f"{name.decode()} {xref} 0 R"
                for name, xref in additions) + ">>"
            doc.xref_set_key(doc[pno].xref, "Resources/Font", fd)
        return fontmap

    templates = {}
    for pno in npages:
        _, bases, chunks, tail, _, is_body, is_heading = \
            page_helpers(doc, pno)
        templates[pno] = {
            "nonbody": [c for c in chunks
                        if not is_body(c) and not is_heading(c)],
            "tail": tail,
            "first_pre": chunks[0].pre if chunks else [],
            "xrefs": doc[pno].get_contents(),
        }

    idx, used = 0, 0
    for di, pno in enumerate(npages):
        if idx >= len(lines):
            break
        top = TOP_FIRST if di == 0 else TOP_PAGE
        baselines = []
        y = top
        while y >= BOTTOM - 0.01 and idx + len(baselines) < len(lines):
            baselines.append(y)
            y -= BODY_LINE
        # never leave a note's opening line alone at a page foot
        while (len(baselines) > 1 and idx + len(baselines) < len(lines)
               and lines[idx + len(baselines) - 1]["starts_note"]):
            baselines.pop()
        take = lines[idx:idx + len(baselines)]
        idx += len(take)
        used += 1

        tmpl = templates[pno]
        fontmap = font_plan(pno, take)
        body = []
        for ln, base_y in zip(take, baselines):
            dx = column(pno) - column(ln["src"])
            for c in ln["chunks"]:
                c.font = (fontmap[int(c.font[0][1:])], c.font[1])
                c.x += dx
                c.y = base_y
                c.pre = []
                body.append(c)
        if di == 0:
            final = tmpl["nonbody"] + body
        else:
            if body:
                body[0].pre = tmpl["first_pre"]
            final = body + tmpl["nonbody"]
        doc.update_stream(tmpl["xrefs"][0], rebuild(final,
                                                    tmpl["tail"]))
        for x in tmpl["xrefs"][1:]:
            doc.update_stream(x, b"")

    for pno in reversed(npages[used:]):
        doc.delete_page(pno)
    return used


def fix_body_page(doc, pno, state):
    """Renumber this page's superscript markers, in stream order, and
    shift a same-line continuation right by the extra digit width."""
    xrefs = doc[pno].get_contents()
    if not xrefs:
        return 0
    # the marker font on this page (Times); Contents numbers are also
    # set at 7pt, but in CrimsonText, and must not be touched
    times = next((name for name, base in font_bases(doc[pno]).items()
                  if "TimesNewRoman" in base), None)
    if times is None:
        return 0
    marker_re = re.compile(
        rb"(" + re.escape(times)
        + rb" 1 Tf\s*\n7 0 0 7 [-\d.]+ [-\d.]+ Tm\s*\n)"
        rb"(\(\d+\)Tj|\[(?:\(\d+\)[-\d.]+ )+\]TJ)"
        rb"(?:(\s*\n/F\d+ 1 Tf\s*\n10\.5 0 0 10\.5 )"
        rb"([-\d.]+)( [-\d.]+ Tm))?")
    data = b"".join(doc.xref_stream(x) for x in xrefs)
    # marker Tm positions, to tell a same-line continuation (the chunk
    # after the marker sits 4pt lower, starting where the old marker
    # ended) from the next paragraph's fresh Tm
    tmpos = [(float(a), float(b)) for a, b in re.findall(
        re.escape(times)
        + rb" 1 Tf\s*\n7 0 0 7 ([-\d.]+) ([-\d.]+) Tm", data)]
    if tmpos:
        # Each page's embedded Times is subset to the digits that page
        # originally used, so a renumbered marker may have no glyph.
        # Give the page a complete Times (shared across the document)
        # and point the markers at it instead.
        doc[pno].insert_font(
            fontname="FTN",
            fontfile="/System/Library/Fonts/Supplemental/"
                     "Times New Roman.ttf",
            set_simple=1, encoding=1)
    try:
        from fix_vellum_export import mask_strings
    except ImportError:
        from pdf_tokens import mask_strings
    masked = mask_strings(data)
    edits = []                 # (start, end, replacement) on `data`
    matches = list(marker_re.finditer(data))
    if len(matches) != len(tmpos):
        raise SystemExit(f"p{pno + 1}: {len(tmpos)} markers, "
                         f"{len(matches)} rewritten")
    for m, (mx, my) in zip(matches, tmpos):
        old_s = b"".join(re.findall(rb"\((\d+)\)", m.group(2)))
        old = int(old_s)
        state["global"] += 1
        new_s = str(state["global"]).encode()
        state["markers"].append((pno + 1, old, state["global"]))
        head = re.sub(rb"^/F\d+ 1 Tf", b"/FTN 1 Tf", m.group(1))
        marker_tok = (b"(%s)Tj" % new_s) if m.group(2).startswith(b"(") \
            else (b"[(%s)500 ]TJ" % new_s)
        if m.group(4) is None:            # nothing follows in this block
            edits.append((m.start(), m.end(), head + marker_tok))
            continue
        x_next = float(m.group(4))
        # Same line?  The continuation sits on the body baseline (the
        # marker's baseline minus the 4pt superscript lift) starting at
        # about where the old marker ended -- Vellum lets it begin up
        # to ~1pt inside the marker's last digit, over the space that
        # usually follows, so allow that much slack.
        mm = re.search(rb"10\.5 0 0 10\.5 [-\d.]+ ([-\d.]+) Tm",
                       m.group(0))
        dy = float(mm.group(1)) if mm else None
        diff = TIMES_DIGIT * (len(new_s) - len(old_s))
        same_line = (diff and dy is not None
                     and abs((my - 4.0) - dy) < 0.1
                     and abs(x_next - (mx + TIMES_DIGIT * len(old_s)))
                     < 1.5)
        if same_line:
            x_next += diff
            # The shifted Tm also sets the line origin for everything
            # that follows in this block, so compensate on the FIRST
            # line-break after it: pull that break's x back by the
            # same amount, restoring every later line to its place.
            for bm in re.finditer(
                    rb"[-\d.]+ [-\d.]+ (?:TD|Td)"
                    rb"|[\d.]+ 0 0 [\d.]+ [-\d.]+ [-\d.]+ Tm"
                    rb"|(?<![A-Za-z])T\*(?![A-Za-z])"
                    rb"|(?<![A-Za-z])ET(?![A-Za-z])",
                    masked[m.end():]):
                tok = bm.group(0)
                if tok in (b"T*",) or tok.startswith(b"T*"):
                    edits.append((m.end() + bm.start(),
                                  m.end() + bm.start(),
                                  b"%.5g 0 Td\n" % (-diff / 10.5)))
                    break
                if tok == b"ET" or b" Tm" in tok:
                    break                 # absolute reset: nothing to do
                tx, ty = tok.split()[:2]
                if float(ty) == 0:
                    continue              # still on the marker's line
                edits.append((m.end() + bm.start(),
                              m.end() + bm.start() + len(tx),
                              b"%g" % (float(tx) - diff / 10.5)))
                break
        edits.append((m.start(), m.end(),
                      head + marker_tok + m.group(3)
                      + (b"%g" % x_next) + m.group(5)))

    if not edits:
        return 0
    out, pos_e = [], 0
    for s, e, r in sorted(edits):
        out.append(data[pos_e:s])
        out.append(r)
        pos_e = e
    out.append(data[pos_e:])
    doc.update_stream(xrefs[0], b"".join(out))
    for x in xrefs[1:]:
        doc.update_stream(x, b"")
    return len(matches)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    npages = notes_pages(doc)
    if not npages:
        sys.exit("Notes pages not found")

    state = {"global": 0, "markers": []}
    for pno in range(npages[0]):
        fix_body_page(doc, pno, state)
    n_markers = state["global"]

    state = {"global": 0, "section_old": 0, "notes": [], "xrefs": []}
    lines = collect_notes_lines(doc, npages, state)
    used = write_notes_pages(doc, npages, lines)

    assert state["global"] == n_markers, (state["global"], n_markers)
    print(f"markers renumbered: {n_markers}")
    print(f"notes renumbered:   {state['global']} "
          f"({len(lines)} lines reflowed over {used} notes pages, "
          f"{len(npages) - used} page(s) removed)")
    print("cross-references:   " + ", ".join(
        f"p{p}: {o}->{n} ({t})" for p, o, n, t in state["xrefs"]))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        scratch = tmp.name
    doc.save(scratch, garbage=4, deflate=True)
    with open(scratch, "rb") as src, open(args.out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.unlink(scratch)
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
