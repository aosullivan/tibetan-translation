"""Content-stream tokenizer shared by Vellum PDF post-processing.

Copied from Dispelling Error's `vellum/fix_vellum_export.py` so the
continuous-endnotes scripts can run without that book-specific file.
Keep the two copies in step if the tokenizer changes.
"""
import re


def tokenize(data):
    """Split a content stream into PDF tokens, keeping strings and arrays
    whole so their bytes survive untouched."""
    out, i, n = [], 0, len(data)
    while i < n:
        c = data[i:i + 1]
        if c in b" \t\r\n":
            i += 1
        elif c == b"%":
            j = data.find(b"\n", i)
            i = n if j < 0 else j + 1
        elif c == b"(":
            j, depth = i + 1, 1
            while j < n and depth:
                if data[j:j + 1] == b"\\":
                    j += 2
                    continue
                if data[j:j + 1] == b"(":
                    depth += 1
                elif data[j:j + 1] == b")":
                    depth -= 1
                j += 1
            out.append(data[i:j])
            i = j
        elif c == b"[":
            j, depth = i + 1, 1
            while j < n and depth:
                if data[j:j + 1] == b"\\":
                    j += 2
                    continue
                if data[j:j + 1] == b"(":
                    k, d2 = j + 1, 1
                    while k < n and d2:
                        if data[k:k + 1] == b"\\":
                            k += 2
                            continue
                        if data[k:k + 1] == b"(":
                            d2 += 1
                        elif data[k:k + 1] == b")":
                            d2 -= 1
                        k += 1
                    j = k
                    continue
                if data[j:j + 1] == b"[":
                    depth += 1
                elif data[j:j + 1] == b"]":
                    depth -= 1
                j += 1
            out.append(data[i:j])
            i = j
        elif c == b"<":
            j = data.find(b">", i)
            out.append(data[i:j + 1])
            i = j + 1
        else:
            j = i + 1 if c == b"/" else i
            while j < n and data[j:j + 1] not in b" \t\r\n()[]<>/%":
                j += 1
            if j == i:
                j = i + 1
            out.append(data[i:j])
            i = j
    return out


class Chunk:
    __slots__ = ("font", "scale", "x", "y", "op", "newline", "pre")

    def __init__(self, font, scale, x, y, op, newline, pre):
        self.font = font
        self.scale = scale
        self.x = x
        self.y = y
        self.op = op
        self.newline = newline
        self.pre = pre


def mask_strings(data, depth=0):
    """Same-length copy with literal-string contents blanked, so a search
    for an operator cannot match text that merely looks like one."""
    out = bytearray(data)
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        if depth:
            if c == 0x5C:
                out[i] = 0x20
                if i + 1 < n:
                    out[i + 1] = 0x20
                i += 2
                continue
            if c == 0x28:
                depth += 1
            elif c == 0x29:
                depth -= 1
            if depth:
                out[i] = 0x20
        elif c == 0x28:
            depth = 1
        i += 1
    return bytes(out)


OPERATOR = re.compile(rb"^[A-Za-z'\"][A-Za-z0-9'\"*]*$")


def scan(tokens):
    """Walk the stream, resolving every text-drawing op to an absolute
    position. Honours the PDF `'` and `"` operators (next-line-and-show);
    without that, a line Vellum opens with a special glyph via `'` is
    rebuilt on top of the line above it."""
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


def rebuild(chunks, tail):
    out = []
    for c in chunks:
        out += c.pre
        out += [c.font[0], c.font[1], b"Tf"] if c.font else []
        out += [b"%g 0 0 %g %.4f %.4f" % (c.scale, c.scale, c.x, c.y),
                b"Tm", c.op[0], c.op[1]]
    return b"\n".join(out + list(tail))
