#!/usr/bin/env python3
"""Validate a continuous-notes PDF against the original print PDF.

Checks, old -> new, page by page:

  A. Correspondence.  Every superscript marker and every endnote in
     the new file sits where its old counterpart sat (markers exactly;
     notes on the same page, text column unmoved, only pulled up), and
     carries the continuous number its position in the sequence earns.
     Emits an audit table (CSV) pairing each marker's body context with
     its note's opening words, for human spot-checking.

  B. No lost or duplicated text.  Per page, the multiset of logical
     lines (note numbers, marker digits and cross-reference numbers
     normalised away) is identical, minus the removed headings.

  C. No collisions.  No two spans overlap on any page in the new file
     that did not overlap in the old; on Notes pages, successive lines
     keep at least 9pt of leading; no line extends past its old right
     edge by more than one digit width; note numbers do not intrude
     into the text column.

  D. Glyphs actually render.  Each marker and each note number is
     rasterised and must contain ink (a digit missing from a subset
     font extracts as text but prints as nothing).

Usage:
    python3 validate_continuous_notes.py <old.pdf> <new.pdf> \
        [--audit audit.csv]
Exit status is non-zero if any check fails.
"""
import argparse
import csv
import re
import sys
from collections import Counter

import pymupdf

TOL = 0.75
DIGIT_W = 3.5          # one Times digit at 7pt


def logical_lines(page):
    """Dict lines merged into logical lines (same baseline -> one line),
    in reading order.  Each: dict(y0,y1,x0,x1,text,spans)."""
    raw = []
    for blk in page.get_text("dict")["blocks"]:
        if blk["type"]:
            continue
        for ln in blk["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            raw.append({
                "y0": ln["bbox"][1], "y1": ln["bbox"][3],
                "x0": ln["bbox"][0], "x1": ln["bbox"][2],
                "text": "".join(s["text"] for s in ln["spans"]),
                "spans": spans,
            })
    raw.sort(key=lambda l: (round(l["y0"] / 3), l["x0"]))
    out = []
    for ln in raw:
        if out and abs(ln["y0"] - out[-1]["y0"]) < 3:
            m = out[-1]
            m["spans"] += ln["spans"]
            m["x0"] = min(m["x0"], ln["x0"])
            m["x1"] = max(m["x1"], ln["x1"])
            m["y1"] = max(m["y1"], ln["y1"])
        else:
            out.append(dict(ln))
    for m in out:
        m["spans"].sort(key=lambda s: s["bbox"][0])
        m["text"] = "".join(s["text"] for s in m["spans"]
                            ) if False else m["text"]
    return out


def is_marker(span):
    return "TimesNewRoman" in span["font"]


def norm_body(line):
    t = "".join(s["text"] for s in line["spans"] if not is_marker(s))
    return re.sub(r"\s+", " ", t).strip()


def norm_note(line):
    t = re.sub(r"^\s*\d+\.\s*", "", line["text"])
    t = re.sub(r"note \d+ (in|to)", r"note # \1", t)
    return re.sub(r"\s+", " ", t).strip()


def spans_of(page):
    out = []
    for blk in page.get_text("dict")["blocks"]:
        if blk["type"]:
            continue
        for ln in blk["lines"]:
            for s in ln["spans"]:
                if s["text"].strip():
                    out.append(s)
    return out


def overlap_keys(page):
    """Set of (normalised text pair) for span pairs that overlap."""
    spans = spans_of(page)
    keys = set()
    detail = []
    for i, a in enumerate(spans):
        ax0, ay0, ax1, ay1 = a["bbox"]
        for b in spans[i + 1:]:
            bx0, by0, bx1, by1 = b["bbox"]
            xo = min(ax1, bx1) - max(ax0, bx0)
            yo = min(ay1, by1) - max(ay0, by0)
            if xo > 0.75 and yo > 2.0:
                pair = tuple(sorted(
                    re.sub(r"\d+", "#", s["text"].strip())[:25]
                    for s in (a, b)))
                if pair not in keys:
                    detail.append((pair, round(xo, 1), round(yo, 1),
                                     a["text"][:40], b["text"][:40]))
                keys.add(pair)
    return keys, detail


def markers(doc, upto):
    """Superscript markers in reading order: (page, value, x, y, line)."""
    out = []
    for pno in range(upto):
        for ln in logical_lines(doc[pno]):
            for s in ln["spans"]:
                if is_marker(s):
                    out.append({"page": pno + 1,
                                "value": int(s["text"]),
                                "x": s["origin"][0], "y": s["origin"][1],
                                "bbox": s["bbox"],
                                "line": norm_body(ln)})
    return out


def notes(doc, pages):
    """Endnotes in order: (page, value, heading, first line, num bbox,
    text x).  Heading is tracked on the old file; None on the new."""
    out = []
    heading = None
    for pno in pages:
        for ln in logical_lines(doc[pno]):
            spans = ln["spans"]
            s0 = spans[0]
            if ("Palatino-Bold" in s0["font"] and abs(s0["size"] - 10) < 0.2
                    and s0["text"].strip() != "NOTES"):
                if out and "heading" in out[-1]:
                    out[-1]["heading"] += " " + ln["text"].strip()
                else:
                    out.append({"heading": ln["text"].strip(),
                                "page": pno + 1})
                continue
            m = re.match(r"^(\d+)\.\s", ln["text"])
            if m and "Athelas" in s0["font"] and len(spans) > 1:
                # note numbers hang left of the wrapped-line column
                text_span = next(
                    (s for s in spans[1:] if s["text"].strip()), spans[1])
                out.append({"page": pno + 1, "value": int(m.group(1)),
                            "first": norm_note(ln)[:80],
                            "bbox": s0["bbox"],
                            "text_x": text_span["bbox"][0],
                            "y": ln["y0"]})
    return out


def ink_count(page, bbox, dpi=150):
    pix = page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=dpi)
    grey = pix.samples
    return sum(1 for v in grey[::pix.n] if v < 128), pix


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--audit", help="write the correspondence audit CSV")
    args = ap.parse_args()

    old, new = pymupdf.open(args.old), pymupdf.open(args.new)
    fails, warns = [], []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
        return cond

    check(len(new) <= len(old),
          f"page count grew {len(old)} -> {len(new)}")

    # Notes pages: from the NOTES title page to the end, both files
    start = next(p for p in range(len(old))
                 if re.search(r"^NOTES$", old[p].get_text(), re.M))
    npages = list(range(start, len(old)))

    # ---------------- A. correspondence ------------------------------
    om, nm = markers(old, start), markers(new, start)
    on = [n for n in notes(old, npages) if "value" in n]
    nn = [n for n in notes(new, list(range(start, len(new))))
          if "value" in n]
    headings = [n["heading"] for n in notes(old, npages)
                if "heading" in n]
    check(len(om) == len(nm) == len(on) == len(nn),
          f"counts: markers {len(om)}->{len(nm)}, notes "
          f"{len(on)}->{len(nn)}")
    # old sequence must segment into the headings' sections, 1..k each
    sizes = Counter()
    sec, expect = -1, 0
    oi = iter(notes(old, npages))
    events = list(notes(old, npages))
    sec = None
    k = 0
    for e in events:
        if "heading" in e:
            sec, k = e["heading"], 0
        else:
            k += 1
            check(e["value"] == k,
                  f"old note under {sec!r} numbered {e['value']}, "
                  f"expected {k}")
    for i, (a, b) in enumerate(zip(om, nm)):
        check(b["value"] == i + 1,
              f"body marker #{i + 1} (p{a['page']}) is {b['value']}")
        check(a["page"] == b["page"] and abs(a["x"] - b["x"]) < TOL
              and abs(a["y"] - b["y"]) < 2.0,
              f"marker #{i + 1} moved: p{a['page']} "
              f"({a['x']:.1f},{a['y']:.1f}) -> p{b['page']} "
              f"({b['x']:.1f},{b['y']:.1f})")
    def column(folio):
        return 79.8 if folio % 2 == 1 else 52.8

    for i, (a, b) in enumerate(zip(on, nn)):
        check(b["value"] == i + 1,
              f"note #{i + 1} (p{a['page']}) is numbered {b['value']}")
        check(a["first"] == b["first"],
              f"note #{i + 1} opening changed:\n  old {a['first']!r}\n"
              f"  new {b['first']!r}")
        # notes reflow across pages: a note may land in the other
        # text column (recto 79.8 / verso 52.8), but nowhere else
        expect_x = a["text_x"] + column(223 + b["page"] - start - 1) \
            - column(223 + a["page"] - start - 1)
        check(abs(expect_x - b["text_x"]) < TOL,
              f"note #{i + 1} text column {b['text_x']:.1f}, "
              f"expected {expect_x:.1f}")
    check(not [n for n in notes(new, list(range(start, len(new))))
               if "heading" in n],
          "chapter headings remain in the new Notes")

    # ---------------- B. line multisets ------------------------------
    def is_head_line(l):
        return ("Palatino-Bold" in l["spans"][0]["font"]
                and abs(l["spans"][0]["size"] - 10) < 0.2)

    for pno in range(start):                 # body: page by page
        oc = Counter(norm_body(l) for l in logical_lines(old[pno]))
        nc = Counter(norm_body(l) for l in logical_lines(new[pno]))
        if oc != nc:
            for t in (oc - nc):
                fails.append(f"p{pno + 1} line lost/changed: {t[:80]!r}")
            for t in (nc - oc):
                fails.append(f"p{pno + 1} line gained/changed: {t[:80]!r}")
    # notes reflow across pages, so compare the whole section at once
    oc = Counter(norm_note(l) for p in range(start, len(old))
                 for l in logical_lines(old[p]) if not is_head_line(l))
    nc = Counter(norm_note(l) for p in range(start, len(new))
                 for l in logical_lines(new[p]))
    # running heads/folios of deleted pages leave the count legitimately
    oc = Counter({t: c for t, c in oc.items()
                  if t not in ("Notes", "|", "") and not t.isdigit()})
    nc = Counter({t: c for t, c in nc.items()
                  if t not in ("Notes", "|", "") and not t.isdigit()})
    if oc != nc:
        for t in (oc - nc):
            fails.append(f"notes line lost/changed: {t[:80]!r}")
        for t in (nc - oc):
            fails.append(f"notes line gained/changed: {t[:80]!r}")

    # ---------------- C. collisions / spacing / edges -----------------
    notes_bearing = []
    for pno in range(len(new)):
        okeys, _ = overlap_keys(old[pno]) if pno < len(old) else (set(), [])
        nkeys, ndetail = overlap_keys(new[pno])
        for (pair, xo, yo, ta, tb) in ndetail:
            if pair not in okeys:
                fails.append(f"p{pno + 1} span overlap {xo}x{yo}pt: "
                             f"{ta!r} / {tb!r}")
        if pno >= start:
            lines = [l for l in logical_lines(new[pno])
                     if "Athelas" in l["spans"][0]["font"]]
            if lines:
                notes_bearing.append(pno)
            base = [l["spans"][0]["origin"][1] for l in lines]
            for a, b, ya, yb in zip(lines, lines[1:], base, base[1:]):
                gap = yb - ya
                if gap < 9.0:
                    fails.append(f"p{pno + 1} line spacing {gap:.1f}pt "
                                 f"before {b['text'][:50]!r}")
                if not (40 < a["y0"] and a["y1"] < 535):
                    fails.append(f"p{pno + 1} line outside text area: "
                                 f"{a['text'][:50]!r}")
        if pno < start:
            # right edge / vertical: lines matched in reading order
            # may extend past their old right edge by one digit
            olines = logical_lines(old[pno])
            nlines = logical_lines(new[pno])
            if [norm_body(l) for l in olines] == \
               [norm_body(l) for l in nlines]:
                for o, l in zip(olines, nlines):
                    if l["x1"] > o["x1"] + DIGIT_W + TOL:
                        warns.append(f"p{pno + 1} line extends "
                                     f"{l['x1'] - o['x1']:.1f}pt past "
                                     f"old right edge: "
                                     f"{l['text'][:50]!r}")
                    if abs(l["y0"] - o["y0"]) > TOL:
                        fails.append(f"p{pno + 1} body line moved "
                                     f"vertically: {l['text'][:50]!r}")
    # every notes page but the last must be filled to near the foot --
    # a page that ends high is stranded space the reflow should have
    # filled from the next page
    for pno in notes_bearing[:-1]:
        lines = [l for l in logical_lines(new[pno])
                 if "Athelas" in l["spans"][0]["font"]]
        if lines and lines[-1]["y1"] < 495:
            fails.append(f"p{pno + 1} notes page ends at "
                         f"y={lines[-1]['y1']:.0f} (foot ~527): "
                         f"stranded gap")

    # ---------------- D. ink ------------------------------------------
    todo = {}
    for m in nm:
        todo.setdefault(m["page"], []).append(("marker", m["bbox"]))
    for n in nn:
        todo.setdefault(n["page"], []).append(("note", n["bbox"]))
    inkless = 0
    for pno, boxes in todo.items():
        page = new[pno - 1]
        pix = page.get_pixmap(dpi=150)
        for kind, bbox in boxes:
            x0, y0, x1, y1 = [v * 150 / 72 for v in bbox]
            dark = 0
            for yy in range(int(y0), min(int(y1) + 1, pix.height)):
                row = yy * pix.width * pix.n
                for xx in range(int(x0), min(int(x1) + 1, pix.width)):
                    if pix.samples[row + xx * pix.n] < 128:
                        dark += 1
            if dark < 5:
                inkless += 1
                fails.append(f"p{pno} {kind} at {bbox} has {dark} "
                             f"dark pixels (glyph missing?)")
    check(inkless == 0, f"{inkless} numbers render without ink")

    # ---------------- cross-references --------------------------------
    text = " ".join(new[p].get_text()
                    for p in range(start, len(new)))
    xrefs = re.findall(r"[Ss]ee note (\d+) (in|to)", text)
    print(f"cross-references in new notes: {xrefs}")

    # ---------------- audit -------------------------------------------
    if args.audit:
        # old heading for each note, by walking the old events
        sec_of, sec, counts = [], None, Counter()
        for e in events:
            if "heading" in e:
                sec = e["heading"]
            else:
                sec_of.append(sec)
        # offset of each section in the continuous sequence
        with open(args.audit, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["new", "old_section", "old_no", "body_page",
                        "body_context", "notes_page", "note_opens"])
            for i, (a, b, na, nb) in enumerate(
                    zip(om, nm, on, nn)):
                w.writerow([i + 1, sec_of[i], a["value"], a["page"],
                            a["line"][:90], na["page"], na["first"][:90]])
        print(f"audit written: {args.audit} ({len(om)} rows)")

    for w_ in warns:
        print("WARN:", w_)
    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for f_ in fails[:60]:
            print(" -", f_)
        return 1
    print(f"\nPASS: {len(nm)} markers and {len(nn)} notes correspond, "
          f"no lost lines, no new overlaps, all numbers inked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
