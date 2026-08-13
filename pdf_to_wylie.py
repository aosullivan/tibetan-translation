#!/usr/bin/env python3
"""
Script to extract Wylie transliteration text from PDF files.
Supports both text-based PDFs (direct extraction) and image-based PDFs (OCR via Claude Vision).
"""

import os
import sys
import argparse
import logging
import re
import math
import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv
import pdfplumber
import PyPDF2

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import TranslationConfig as cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_to_wylie.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFTibetanExtractor:
    """Extract text from PDF files (supports both text-based and image-based PDFs)."""
    
    # Prompt for OCR'ing a whole page in one shot.
    _PAGE_PROMPT = """Transcribe the Tibetan text in this image into EWTS Wylie transliteration.

The image contains Tibetan dbu-can script (typically a pecha or printed page). Read every legible syllable and output it as standard EWTS Wylie (Roman alphabet).

Conventions:
- Use `_` for tsheg (the inter-syllable dot) as needed; `/` for shad; `//` for double shad.
- Preserve line breaks as they appear in the source.
- For Sanskrit syllables and stacked consonants, use standard EWTS conventions (e.g. `+` for stacking, capitals for retroflex/Sanskrit letters where appropriate).
- If a syllable is illegible or uncertain, write `[?]` in place of the uncertain syllable rather than guessing.

Output only the Wylie transliteration. No commentary, no translation, no preamble."""

    # Prompt for OCR'ing one horizontal segment of a single pecha line.
    # Deliberately strict: on small ambiguous crops the model otherwise narrates
    # ("let me re-transcribe…") and emits Unicode alongside the Wylie.
    _TILE_PROMPT = """You are an OCR engine. The image is a small horizontal crop from ONE line of a Tibetan dbu-can pecha (woodblock print), possibly cut off at the left/right edge.

Output ONLY the EWTS Wylie transliteration (Latin letters) of the Tibetan, on a single line, left to right.

ABSOLUTE RULES:
- Wylie only. NEVER output Tibetan Unicode script.
- NEVER write English words, commentary, or phrases like "I see", "let me", "the image", "retranscribe".
- Do NOT offer alternatives or re-readings. One pass only.
- If the crop is blank, just a border rule, or fully illegible, output exactly: [?]
- Use `_` for tsheg, `/` for shad, `//` for double shad, `+` for stacked/Sanskrit consonants, capitals for retroflex/Sanskrit letters.

Transcribe now, Wylie only:"""

    def __init__(self, api_key: Optional[str] = None, dpi: int = 300,
                 tile: bool = False, tile_target_width: int = 4500,
                 tile_seg_width: int = 1550):
        self.api_key = api_key
        self.dpi = dpi
        self.tile = tile
        # Effective full-folio width (px) to render before tiling, and the max
        # width of each segment sent to the API (kept under the ~1568px long-edge
        # cap so the model sees the crop at full detail).
        self.tile_target_width = tile_target_width
        self.tile_seg_width = tile_seg_width
        if api_key:
            self.claude_client = Anthropic(api_key=api_key)
        else:
            self.claude_client = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        Extract text from PDF using multiple methods for better coverage.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (combined_text, extraction_methods_used)
        """
        methods_used = []
        extracted_texts = []
        
        # Method 1: pdfplumber (better for complex layouts)
        pdfplumber_error = None
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                if text_parts:
                    pdfplumber_text = "\n\n".join(text_parts)
                    extracted_texts.append(("pdfplumber", pdfplumber_text))
                    methods_used.append("pdfplumber")
                    logger.info(f"Extracted {len(pdfplumber_text)} characters using pdfplumber")
                else:
                    logger.warning("pdfplumber extracted no text (PDF may be image-based)")
        except Exception as e:
            pdfplumber_error = str(e)
            logger.warning(f"pdfplumber extraction failed: {e}", exc_info=True)
        
        # Method 2: PyPDF2 (fallback)
        pypdf2_error = None
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                if text_parts:
                    pypdf2_text = "\n\n".join(text_parts)
                    extracted_texts.append(("PyPDF2", pypdf2_text))
                    methods_used.append("PyPDF2")
                    logger.info(f"Extracted {len(pypdf2_text)} characters using PyPDF2")
                else:
                    logger.warning("PyPDF2 extracted no text (PDF may be image-based)")
        except Exception as e:
            pypdf2_error = str(e)
            logger.warning(f"PyPDF2 extraction failed: {e}", exc_info=True)
        
        # Method 3: OCR using Claude Vision (for image-based PDFs)
        if not extracted_texts and self.claude_client:
            logger.info("Attempting OCR extraction using Claude Vision API")
            try:
                ocr_text = self._extract_text_with_claude_vision(pdf_path)
                if ocr_text:
                    extracted_texts.append(("Claude Vision OCR", ocr_text))
                    methods_used.append("Claude Vision OCR")
                    logger.info(f"Extracted {len(ocr_text)} characters using Claude Vision OCR")
            except Exception as e:
                logger.warning(f"Claude Vision OCR extraction failed: {e}")
        
        if not extracted_texts:
            error_msg = "Failed to extract text using any method"
            if pdfplumber_error:
                error_msg += f"\n  pdfplumber error: {pdfplumber_error}"
            if pypdf2_error:
                error_msg += f"\n  PyPDF2 error: {pypdf2_error}"
            if not pdfplumber_error and not pypdf2_error:
                error_msg += " (PDF may be image-based or contain no extractable text)"
            if not self.claude_client:
                error_msg += "\n  Note: Claude Vision OCR not available (ANTHROPIC_API_KEY not set)"
            raise ValueError(error_msg)
        
        # Combine texts, preferring pdfplumber if available
        combined_text = extracted_texts[0][1]  # Use first successful extraction
        
        return combined_text, methods_used
    
    
    def _extract_text_with_claude_vision(self, pdf_path: str) -> str:
        """
        Extract text from image-based PDF using Claude Vision API.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from all pages
        """
        if not self.claude_client:
            raise ValueError("Claude client not initialized (API key required)")
        
        text_parts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {total_pages} pages with Claude Vision OCR")
                
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"Processing page {page_num + 1}/{total_pages} with Claude Vision")
                    
                    # Render page as image for OCR
                    try:
                        render_dpi = self._tile_render_dpi(page) if self.tile else self.dpi
                        page_image = page.to_image(resolution=render_dpi)
                        pil_image = page_image.original
                    except Exception as e:
                        logger.warning(f"Failed to render page {page_num + 1} as image: {e}")
                        continue

                    # Transcribe: line-aware tiling for wide folios, else whole-page
                    try:
                        if self.tile:
                            page_text = self._ocr_page_tiled(pil_image, page_num + 1)
                        else:
                            page_text = self._vision_ocr_image(pil_image, self._PAGE_PROMPT)
                    except Exception as e:
                        logger.warning(f"Claude Vision API error on page {page_num + 1}: {e}")
                        continue

                    page_text = (page_text or "").strip()
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                        logger.info(f"Extracted {len(page_text)} characters from page {page_num + 1}")
        
        except Exception as e:
            logger.error(f"Claude Vision OCR process failed: {e}")
            raise
        
        if not text_parts:
            return ""

        return "\n\n".join(text_parts)

    def _vision_ocr_image(self, pil_image, prompt: str, max_tokens: int = 2000) -> str:
        """Send a single PIL image to Claude Vision and return the transcribed text."""
        pil_image = self._compress_image_for_claude(pil_image)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=90, optimize=True)
        if len(img_buffer.getvalue()) / (1024 * 1024) > 4.5:
            pil_image = self._compress_image_for_claude(pil_image, max_dimension=2000)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=80, optimize=True)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        response = self.claude_client.messages.create(
            model=cfg.MODEL,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_base64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def _tile_render_dpi(self, page) -> int:
        """DPI that renders this page to roughly tile_target_width px (capped at self.dpi)."""
        width_pt = float(page.width) or 1.0
        wanted = int(math.ceil(self.tile_target_width * 1.1 * 72.0 / width_pt))
        return max(20, min(self.dpi, wanted))

    def _detect_line_bands(self, pil_image):
        """Find vertical (y0, y1) bands for text lines via peaks in a horizontal ink profile.

        Pecha lines are tightly spaced and joined by vowel signs / ya-ra subscripts, so
        the inter-line valleys never reach white. A flat threshold merges everything;
        instead we locate each line's dark head-line peak, then split at the ink valley
        between consecutive peaks. Solid frame rules (near-full-width ink) are ignored.
        """
        import numpy as np
        gray = np.asarray(pil_image.convert("L"))
        H, W = gray.shape
        ink = (gray < 160).sum(axis=1).astype(float)
        ink[ink > 0.85 * W] = 0.0  # drop frame rules

        # Smooth with a small moving average to stabilize peak finding.
        k = max(3, int(0.006 * H) | 1)
        kern = np.ones(k) / k
        s = np.convolve(ink, kern, mode="same")
        if s.max() <= 0:
            return [(0, H)]

        peak_thresh = 0.28 * s.max()
        min_dist = max(8, int(0.035 * H))   # minimum spacing between line centers
        peaks = []
        for y in range(1, H - 1):
            if s[y] >= peak_thresh and s[y] >= s[y - 1] and s[y] > s[y + 1]:
                if peaks and y - peaks[-1] < min_dist:
                    if s[y] > s[peaks[-1]]:
                        peaks[-1] = y
                else:
                    peaks.append(y)
        if not peaks:
            return [(0, H)]

        text_rows = np.where(s > 0.06 * s.max())[0]
        top, bot = int(text_rows[0]), int(text_rows[-1])
        splits = [top]
        for a, b in zip(peaks, peaks[1:]):
            splits.append(a + int(np.argmin(s[a:b])))
        splits.append(bot)

        pad = int(0.004 * H)
        raw = []
        for i in range(len(splits) - 1):
            y0, y1 = splits[i], splits[i + 1]
            if y1 - y0 >= 0.01 * H:
                raw.append((y0, y1))
        # Drop frame slivers / near-empty bands: keep bands whose total ink is a
        # meaningful fraction of the busiest band.
        if raw:
            band_ink = [float(ink[y0:y1].sum()) for y0, y1 in raw]
            cutoff = 0.18 * max(band_ink)
            raw = [b for b, bi in zip(raw, band_ink) if bi >= cutoff]
        return [(max(0, y0 - pad), min(H, y1 + pad)) for y0, y1 in raw]

    def _segment_columns(self, line_img):
        """Split a line image into (x0, x1) segments <= tile_seg_width, cutting at ink minima.

        Cut points are nudged to the whitest nearby column (an inter-syllable gap) so
        glyphs aren't sliced through.
        """
        import numpy as np
        w, h = line_img.size
        if w <= self.tile_seg_width:
            return [(0, w)]
        gray = np.asarray(line_img.convert("L"))
        col_ink = (gray < 160).sum(axis=0).astype(float)
        n_seg = int(math.ceil(w / self.tile_seg_width))
        step = w / n_seg
        cuts = [0]
        for k in range(1, n_seg):
            target = int(k * step)
            win = max(10, int(step * 0.18))
            lo, hi = max(cuts[-1] + 1, target - win), min(w - 1, target + win)
            cut = target if hi <= lo else lo + int(np.argmin(col_ink[lo:hi]))
            cuts.append(cut)
        cuts.append(w)
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def _ocr_page_tiled(self, pil_page, page_num: int) -> str:
        """OCR one wide folio by detecting lines and OCR'ing each line's width-segments."""
        from PIL import Image
        W0, H0 = pil_page.size
        if W0 > self.tile_target_width:
            scale = self.tile_target_width / W0
            pil_page = pil_page.resize(
                (self.tile_target_width, max(1, int(H0 * scale))), Image.Resampling.LANCZOS)

        bands = self._detect_line_bands(pil_page) or [(0, pil_page.size[1])]
        logger.info(f"Page {page_num}: {len(bands)} line band(s) detected")

        lines_out = []
        for li, (y0, y1) in enumerate(bands):
            line_img = pil_page.crop((0, y0, pil_page.size[0], y1))
            segs = self._segment_columns(line_img)
            seg_texts = []
            for si, (x0, x1) in enumerate(segs):
                tile = line_img.crop((x0, 0, x1, line_img.size[1]))
                txt = self._vision_ocr_image(tile, self._TILE_PROMPT, max_tokens=700)
                txt = txt.replace("\n", " ").strip()
                seg_texts.append(txt)
                logger.info(
                    f"  page {page_num} line {li + 1}/{len(bands)} "
                    f"seg {si + 1}/{len(segs)} -> {len(txt)} chars")
            lines_out.append(" ".join(t for t in seg_texts if t))
        return "\n".join(l for l in lines_out if l)

    def _compress_image_for_claude(self, pil_image, max_dimension: int = 3000):
        """
        Compress/resize image to meet Claude's size requirements.
        
        Args:
            pil_image: PIL Image object
            max_dimension: Maximum width or height in pixels
            
        Returns:
            Compressed/resized PIL Image
        """
        from PIL import Image
        
        width, height = pil_image.size
        
        # Resize if image is too large
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        return pil_image


# Lines of model chatter that leak into Vision OCR output despite the prompt.
# These phrases never occur in valid EWTS/Wylie, so matching them is safe.
_COMMENTARY_RE = re.compile(
    r"(?i)("
    r"i'?ll\b|i can\b|i've\b|i have\b|i should\b|i'?m\b|"
    r"here'?s\b|here is\b|"
    r"transcrib|translitera|resolution|legible|candid|"
    r"line by line|with confidence|as best|best as possible|"
    r"no commentary|no preamble|uncertain|densely written|"
    r"scan would|would be needed|partial attempt|conventions\b|"
    r"^\s*note\s*:"
    r")"
)


def _has_tibetan(s: str) -> bool:
    """True if the string contains any Tibetan Unicode codepoint (U+0F00–U+0FFF)."""
    return any(0x0F00 <= ord(c) <= 0x0FFF for c in s)


# Narration fragments that occasionally survive inside a tiled line (the model
# "thinking out loud" on a hard crop). Removed in-place, leaving surrounding Wylie.
_INLINE_NARRATION = [
    re.compile(r"(?i)\s*(?:wait[,!.]?\s*)?(?:let me|let's|i['’]?ll|i can|i['’]?ve|i will|i'm going to|"
               r"below is|here is|i'd)\b[^.:]*[.:]"),
    re.compile(r"(?i)\s*(?:i see|i cannot|i could not|the image|this image|this appears|this segment|"
               r"transliterating what i see|my best (?:reading|attempt))\b[^.]*\."),
]


def _scrub_inline_narration(line: str) -> str:
    """Strip model self-narration fragments from inside a line, keeping the Wylie."""
    for rx in _INLINE_NARRATION:
        line = rx.sub(" ", line)
    return re.sub(r"\s{2,}", " ", line).strip()


def clean_ocr_text(raw_text: str) -> Tuple[str, dict]:
    """Strip model preambles/fences/commentary and convert stray Unicode pages to Wylie.

    Vision OCR output is contaminated with English preambles ("I'll transcribe…"),
    markdown code fences, trailing meta-commentary, and occasional pages emitted as
    Tibetan Unicode instead of EWTS. This normalizes all of that so the result is
    clean Wylie ready for the translation pipeline.

    Returns (cleaned_text, stats).
    """
    try:
        import pyewts
        converter = pyewts.pyewts()
    except Exception as e:
        logger.warning(f"pyewts unavailable ({e}); Unicode pages will be flagged, not converted")
        converter = None

    stats = {"preamble_lines": 0, "fence_lines": 0, "unicode_lines_converted": 0}
    out = []
    for line in raw_text.split("\n"):
        stripped = line.strip()

        # Preserve page markers verbatim
        if stripped.startswith("--- Page "):
            out.append(line)
            continue

        # Remove any self-narration fragments that survived inside a tiled line
        if stripped and not _has_tibetan(stripped):
            scrubbed = _scrub_inline_narration(line)
            if scrubbed != line.strip():
                line = scrubbed
                stripped = line.strip()
        # Drop markdown code fences
        if stripped.startswith("```"):
            stats["fence_lines"] += 1
            continue
        # Drop bare horizontal-rule separators the model inserts
        if stripped == "---":
            continue
        # Drop English commentary/preamble (never contains Tibetan or Wylie content)
        if stripped and not _has_tibetan(stripped) and _COMMENTARY_RE.search(stripped):
            stats["preamble_lines"] += 1
            continue
        # Drop contentless lines: only [?] markers, folio numbers, and punctuation
        # (these come from empty bands a wide-folio line detector over-segments).
        if stripped and re.sub(r"\[\?\]|[\s/\\|_+\-:;.,']|\d", "", stripped) == "":
            continue
        # Convert stray Tibetan Unicode to Wylie. Only the Tibetan runs are
        # converted — mixed lines keep their existing Wylie untouched (running
        # whole already-Wylie lines through pyewts mangles them with brackets).
        if _has_tibetan(line):
            if converter is not None:
                try:
                    new_line = re.sub(
                        r'[ༀ-࿿]+',
                        lambda m: converter.toWylie(m.group(0)),
                        line,
                    )
                    if new_line != line:
                        stats["unicode_lines_converted"] += 1
                    line = new_line
                except Exception as e:
                    logger.warning(f"pyewts conversion failed on a line: {e}")
            else:
                logger.warning("Untransliterated Unicode line left in output (pyewts missing)")
        out.append(line)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    return text, stats


def main():
    """Main function to extract Wylie text from PDF."""
    parser = argparse.ArgumentParser(description='Extract Wylie transliteration text from PDF (supports text-based and image-based PDFs)')
    parser.add_argument('--input', '-i', required=True, help='Input PDF file path')
    parser.add_argument('--output', '-o', help='Output Wylie text file path')
    parser.add_argument('--extract-only', action='store_true', help='Only extract text (same as default behavior)')
    parser.add_argument('--dpi', type=int, default=300, help='Render resolution for Claude Vision OCR (default: 300)')
    parser.add_argument('--no-clean', action='store_true', help='Skip post-OCR cleanup (keep raw model output)')
    parser.add_argument('--tile', action='store_true', help='Line-aware tiling for wide pecha folios (detect lines, OCR width-segments, stitch). Best for ultra-wide scans.')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1
    
    if not input_path.suffix.lower() == '.pdf':
        logger.error("Input file must be a PDF")
        return 1
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.wylie.txt')
    
    try:
        # Get API key for OCR if needed
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Extract text from PDF
        logger.info(f"Extracting text from {input_path}")
        extractor = PDFTibetanExtractor(api_key=api_key, dpi=args.dpi, tile=args.tile)
        raw_text, methods = extractor.extract_text_from_pdf(str(input_path))
        logger.info(f"Extraction methods used: {', '.join(methods)}")
        
        if not raw_text:
            logger.error("No text extracted from PDF")
            return 1

        # Clean up model chatter / stray Unicode that leaks into OCR output
        if not args.no_clean:
            raw_text, clean_stats = clean_ocr_text(raw_text)
            logger.info(
                f"Cleanup: dropped {clean_stats['preamble_lines']} preamble line(s), "
                f"{clean_stats['fence_lines']} fence line(s), "
                f"converted {clean_stats['unicode_lines_converted']} Unicode line(s) to Wylie"
            )

        if args.extract_only:
            # Save extracted text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            logger.info(f"Extracted text saved to {output_path}")
            logger.info("Text extraction complete")
            return 0
        
        # If not extract-only, save to Wylie output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
        logger.info(f"Wylie text saved to {output_path}")
        logger.info(f"Processing complete: {len(raw_text)} characters extracted")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
