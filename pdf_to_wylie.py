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
import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv
import pdfplumber
import PyPDF2

# Add src directory to path for imports (if needed in future)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
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
                        # Start with lower resolution to avoid size limits
                        page_image = page.to_image(resolution=200)
                        pil_image = page_image.original
                    except Exception as e:
                        logger.warning(f"Failed to render page {page_num + 1} as image: {e}")
                        continue
                    
                    # Compress image to meet Claude's 5MB limit
                    pil_image = self._compress_image_for_claude(pil_image)
                    
                    # Convert PIL Image to base64
                    img_buffer = io.BytesIO()
                    # Use JPEG with compression for smaller file size
                    pil_image.save(img_buffer, format='JPEG', quality=85, optimize=True)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    
                    # Check size and reduce if needed
                    img_size_mb = len(img_buffer.getvalue()) / (1024 * 1024)
                    if img_size_mb > 4.5:  # Leave some margin under 5MB
                        logger.warning(f"Image for page {page_num + 1} is {img_size_mb:.2f}MB, compressing further...")
                        pil_image = self._compress_image_for_claude(pil_image, max_dimension=2000)
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format='JPEG', quality=75, optimize=True)
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        img_size_mb = len(img_buffer.getvalue()) / (1024 * 1024)
                        logger.info(f"Compressed to {img_size_mb:.2f}MB")
                    
                    # Use Claude Vision to extract text
                    try:
                        response = self.claude_client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=4000,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": img_base64
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": """Extract all Wylie transliteration text from this image. 

The image contains Wylie transliteration (Tibetan text in Roman alphabet).

Preserve:
- Line breaks and paragraph structure
- All punctuation marks and spacing
- Exact Wylie spelling

Output only the extracted Wylie text, no explanations or translations."""
                                    }
                                ]
                            }]
                        )
                        
                        page_text = response.content[0].text.strip()
                        if page_text:
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                            logger.info(f"Extracted {len(page_text)} characters from page {page_num + 1}")
                    
                    except Exception as e:
                        logger.warning(f"Claude Vision API error on page {page_num + 1}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Claude Vision OCR process failed: {e}")
            raise
        
        if not text_parts:
            return ""
        
        return "\n\n".join(text_parts)
    
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


def main():
    """Main function to extract Wylie text from PDF."""
    parser = argparse.ArgumentParser(description='Extract Wylie transliteration text from PDF (supports text-based and image-based PDFs)')
    parser.add_argument('--input', '-i', required=True, help='Input PDF file path')
    parser.add_argument('--output', '-o', help='Output Wylie text file path')
    parser.add_argument('--extract-only', action='store_true', help='Only extract text (same as default behavior)')
    
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
        extractor = PDFTibetanExtractor(api_key=api_key)
        raw_text, methods = extractor.extract_text_from_pdf(str(input_path))
        logger.info(f"Extraction methods used: {', '.join(methods)}")
        
        # Since PDF contains Wylie already, we don't need to filter or convert
        # Just save the extracted text directly
        if not raw_text:
            logger.error("No text extracted from PDF")
            return 1
        
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
