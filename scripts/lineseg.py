import numpy as np
from PIL import Image


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


