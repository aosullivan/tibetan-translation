#!/usr/bin/env python3
"""pecha_ocr_stacked.py — OCR wide pecha folio scans to EWTS Wylie.

Why this exists alongside pdf_to_wylie.py --tile:

  1. Works from the original page images, not a PDF render round-trip.
  2. Strips the printed frame before line detection.
  3. Detects lines with a peak-relative threshold. Tibetan lines never separate
     to zero ink — descenders and vowel signs keep the valleys well above it —
     so an absolute threshold merges the whole folio into one "line".
  4. Stacks a line's segments vertically into ONE image. The API downscales any
     image wider than 1568px, which shrinks a 3200px folio line into mush; a
     stacked image keeps full glyph resolution AND lets the model see the line
     in order.
  5. Uses thinking={"type":"adaptive"} + output_config={"effort":"high"}.
     Note for config.py: claude-opus-5 rejects `temperature` outright, and with
     thinking left at its default a small max_tokens is consumed entirely by
     thinking, so the call returns no text block at all.

CAVEAT: even so, output on cursive dbu-can woodblock is NOT scholarly quality.
Treat it as a draft to proofread against the scan, never as a source text.

Usage: python pecha_ocr_stacked.py page1.jpg page2.jpg ...
"""
import os, sys, base64, io, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
MODEL = os.environ.get('OCR_MODEL', 'claude-opus-5')

PROMPT = """This image shows ONE line of a Tibetan dbu-can woodblock pecha, cut into {n} horizontal pieces and stacked top to bottom. Piece 1 is the leftmost part of the line, piece {n} the rightmost. Read them in that order as a single continuous line of text.

Output ONLY the EWTS Wylie transliteration of the whole line, on one line.

Rules:
- EWTS: `/` for shad, `//` for double shad, `+` for stacking, `_` for an explicit tsheg before a shad.
- Transcribe what is printed. Do not normalise archaic spellings or silently emend.
- Write [?] for a genuinely illegible syllable rather than guessing.
- Wylie only: no Tibetan Unicode, no commentary, no markdown fences."""


def load_binary(path):
    im = Image.open(path).convert('L')
    a = np.array(im)
    return im, (a < 128)


def strip_frame(mask, im):
    """Drop the printed rectangle: rows/cols that are dark across most of their extent."""
    h, w = mask.shape
    rows = mask.sum(axis=1)
    cols = mask.sum(axis=0)
    row_frame = rows > 0.60 * w
    col_frame = cols > 0.60 * h
    top, bot = 0, h
    for i in range(h // 2):
        if row_frame[i]:
            top = i + 1
    for i in range(h - 1, h // 2, -1):
        if row_frame[i]:
            bot = i
    left, right = 0, w
    for i in range(w // 3):
        if col_frame[i]:
            left = i + 1
    for i in range(w - 1, 2 * w // 3, -1):
        if col_frame[i]:
            right = i
    pad = 4
    return max(0, top - pad), min(h, bot + pad), max(0, left - pad), min(w, right + pad)


def find_lines(mask, y0, y1, x0, x1):
    band = mask[y0:y1, x0:x1]
    rows = band.sum(axis=1)
    # Tibetan lines never fully separate — descenders and vowel signs keep the
    # inter-line valleys well above zero — so threshold relative to peak ink.
    thresh = 0.22 * rows.max()
    on = rows > thresh
    lines, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            lines.append((start, i))
            start = None
    if start is not None:
        lines.append((start, len(on)))
    merged = []
    for s, e in lines:
        if merged and s - merged[-1][1] < 5:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # pad each band outward to recapture vowel marks clipped by the threshold
    out = []
    for s, e in merged:
        if e - s < 25:
            continue
        out.append((y0 + max(0, s - 14), y0 + min(band.shape[0], e + 14)))
    return out


def split_points(mask, ys, ye, x0, x1, nseg):
    """Cut a line into nseg pieces, preferring blank columns near each target."""
    cols = mask[ys:ye, x0:x1].sum(axis=0)
    width = x1 - x0
    targets = [int(width * k / nseg) for k in range(1, nseg)]
    cuts = []
    for t in targets:
        win = 90
        lo, hi = max(0, t - win), min(width, t + win)
        seg = cols[lo:hi]
        blanks = np.where(seg == 0)[0]
        cuts.append(x0 + (lo + int(blanks[np.argmin(abs(blanks - (t - lo)))]) if len(blanks) else t))
    return [x0] + cuts + [x1]




def stack_line(im, ys, ye, bounds, pad=10, gap=18, target_w=1450):
    crops = []
    for i in range(len(bounds) - 1):
        c = im.crop((bounds[i], max(0, ys - pad), bounds[i + 1], min(im.height, ye + pad)))
        if c.width > 10:
            crops.append(c)
    if not crops:
        return None, 0
    w = max(c.width for c in crops)
    scale = min(target_w / w, 3.0)
    if scale > 1:
        crops = [c.resize((int(c.width * scale), int(c.height * scale)), Image.LANCZOS) for c in crops]
        w = int(w * scale)
    h = sum(c.height for c in crops) + gap * (len(crops) - 1)
    canvas = Image.new('L', (w, h), 255)
    y = 0
    for c in crops:
        canvas.paste(c.convert('L'), (0, y))
        y += c.height + gap
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode(), len(crops)


def ocr(b64, n, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=8000,
                thinking={"type": "adaptive"}, output_config={"effort": "high"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": PROMPT.format(n=n)}]}])
            return ' '.join(b.text for b in r.content if getattr(b, 'type', '') == 'text').strip().replace('\n', ' ')
        except Exception as e:
            if attempt == retries - 1:
                return f"[OCR-FAIL: {type(e).__name__}: {str(e)[:80]}]"
            time.sleep(2 * (attempt + 1))


def do_page(path, nseg=3):
    im, mask = load_binary(path)
    y0, y1, x0, x1 = strip_frame(mask, im)
    lines = find_lines(mask, y0, y1, x0, x1)
    jobs = []
    for li, (ys, ye) in enumerate(lines):
        bounds = split_points(mask, ys, ye, x0, x1, nseg)
        b64, n = stack_line(im, ys, ye, bounds)
        jobs.append((li, b64, n))
    res = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(ocr, b, n): li for li, b, n in jobs if b}
        for f, li in futs.items():
            res[li] = f.result()
    return [res.get(i, '') for i in range(len(lines))]


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(f"### {os.path.basename(p)}")
        for i, l in enumerate(do_page(p), 1):
            print(f"{i}| {l}")
        print()
