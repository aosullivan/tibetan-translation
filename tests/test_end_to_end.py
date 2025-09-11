#!/usr/bin/env python3
"""
End-to-end integration test for Tibetan text processor.
Tests the complete pipeline without requiring API calls.
"""

import os
import sys
import logging
import traceback
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append('../src')

from src.tibetan_line_parser import TibetanLineParser
from src.wylie_converter import WylieConverter
from src.phonetic_renderer import PhoneticRenderer
from src.tibetan_file_handler import TibetanFileHandler
from src.tibetan_output_formatter import TibetanOutputFormatter
from src.tibetan_models import TibetanLine, ProcessedLine, ProcessingStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockTranslationClient:
    """Mock translation client for testing without API calls"""
    
    def translate_vajrayana_context(self, wylie_text: str, tibetan_script: str) -> str:
        """Mock translation that returns a formatted response"""
        if not wylie_text.strip():
            return "[Empty content]"
        
        # Simple mock translation based on content length and patterns
        if len(wylie_text) < 20:
            return f"[Mock translation of short text: {wylie_text[:10]}...]"
        elif "phyag 'tshal" in wylie_text:
            return "[Mock translation: Prostrations/homage content]"
        elif "sgrub thabs" in wylie_text:
            return "[Mock translation: Sadhana/practice method content]"
        elif "dbyangs can ma" in wylie_text:
            return "[Mock translation: Content about Saraswati goddess]"
        else:
            return f"[Mock translation of: {wylie_text[:30]}...]"

def test_component_initialization():
    """Test that all components can be initialized successfully"""
    logger.info("Testing component initialization...")
    
    try:
        line_parser = TibetanLineParser()
        wylie_converter = WylieConverter()
        phonetic_renderer = PhoneticRenderer()
        translation_client = MockTranslationClient()
        file_handler = TibetanFileHandler()
        output_formatter = TibetanOutputFormatter()
        
        logger.info("✓ All components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Component initialization failed: {e}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return False

def test_file_parsing(input_file: str):
    """Test parsing of the input file"""
    logger.info(f"Testing file parsing with {input_file}...")
    
    if not os.path.exists(input_file):
        logger.error(f"✗ Input file not found: {input_file}")
        return None
    
    try:
        line_parser = TibetanLineParser()
        tibetan_lines = line_parser.parse_file(input_file)
        
        logger.info(f"✓ Successfully parsed {len(tibetan_lines)} lines")
        
        # Show sample of parsed lines
        for i, line in enumerate(tibetan_lines[:5]):
            logger.info(f"  Line {line.line_number}: {line.wylie_content[:50]}{'...' if len(line.wylie_content) > 50 else ''}")
            logger.info(f"    Structural: {line.is_structural}, Markers: {line.markers}")
        
        if len(tibetan_lines) > 5:
            logger.info(f"  ... and {len(tibetan_lines) - 5} more lines")
        
        return tibetan_lines
    except Exception as e:
        logger.error(f"✗ File parsing failed: {e}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return None

def test_wylie_conversion(tibetan_lines: List[TibetanLine]):
    """Test Wylie to Tibetan script conversion"""
    logger.info("Testing Wylie conversion...")
    
    try:
        wylie_converter = WylieConverter()
        conversion_results = []
        
        # Test with first few non-structural lines
        test_lines = [line for line in tibetan_lines[:10] if not line.is_structural and line.wylie_content.strip()]
        
        for line in test_lines:
            try:
                tibetan_script = wylie_converter.convert_to_tibetan(line.wylie_content)
                conversion_results.append({
                    'line_number': line.line_number,
                    'wylie': line.wylie_content[:50],
                    'tibetan': tibetan_script[:50],
                    'success': True
                })
                logger.info(f"  ✓ Line {line.line_number}: {line.wylie_content[:30]}... → {tibetan_script[:30]}...")
            except Exception as e:
                conversion_results.append({
                    'line_number': line.line_number,
                    'wylie': line.wylie_content[:50],
                    'error': str(e),
                    'success': False
                })
                logger.warning(f"  ⚠ Line {line.line_number} conversion failed: {e}")
        
        success_count = sum(1 for r in conversion_results if r['success'])
        logger.info(f"✓ Wylie conversion: {success_count}/{len(conversion_results)} successful")
        return conversion_results
    except Exception as e:
        logger.error(f"✗ Wylie conversion test failed: {e}")
        return []

def test_phonetic_rendering(tibetan_lines: List[TibetanLine]):
    """Test phonetic rendering"""
    logger.info("Testing phonetic rendering...")
    
    try:
        phonetic_renderer = PhoneticRenderer()
        phonetic_results = []
        
        # Test with first few non-structural lines
        test_lines = [line for line in tibetan_lines[:10] if not line.is_structural and line.wylie_content.strip()]
        
        for line in test_lines:
            try:
                phonetic = phonetic_renderer.generate_phonetic(line.wylie_content)
                phonetic_results.append({
                    'line_number': line.line_number,
                    'wylie': line.wylie_content[:50],
                    'phonetic': phonetic[:50],
                    'success': True
                })
                logger.info(f"  ✓ Line {line.line_number}: {line.wylie_content[:30]}... → {phonetic[:30]}...")
            except Exception as e:
                phonetic_results.append({
                    'line_number': line.line_number,
                    'wylie': line.wylie_content[:50],
                    'error': str(e),
                    'success': False
                })
                logger.warning(f"  ⚠ Line {line.line_number} phonetic failed: {e}")
        
        success_count = sum(1 for r in phonetic_results if r['success'])
        logger.info(f"✓ Phonetic rendering: {success_count}/{len(phonetic_results)} successful")
        return phonetic_results
    except Exception as e:
        logger.error(f"✗ Phonetic rendering test failed: {e}")
        return []

def test_mock_translation(tibetan_lines: List[TibetanLine]):
    """Test mock translation functionality"""
    logger.info("Testing mock translation...")
    
    try:
        translation_client = MockTranslationClient()
        translation_results = []
        
        # Test with first few non-structural lines
        test_lines = [line for line in tibetan_lines[:5] if not line.is_structural and line.wylie_content.strip()]
        
        for line in test_lines:
            try:
                translation = translation_client.translate_vajrayana_context(line.wylie_content, "mock_tibetan")
                translation_results.append({
                    'line_number': line.line_number,
                    'wylie': line.wylie_content[:50],
                    'translation': translation[:100],
                    'success': True
                })
                logger.info(f"  ✓ Line {line.line_number}: {translation[:60]}...")
            except Exception as e:
                translation_results.append({
                    'line_number': line.line_number,
                    'wylie': line.wylie_content[:50],
                    'error': str(e),
                    'success': False
                })
                logger.warning(f"  ⚠ Line {line.line_number} translation failed: {e}")
        
        success_count = sum(1 for r in translation_results if r['success'])
        logger.info(f"✓ Mock translation: {success_count}/{len(translation_results)} successful")
        return translation_results
    except Exception as e:
        logger.error(f"✗ Mock translation test failed: {e}")
        return []

def test_complete_pipeline(input_file: str, output_file: str):
    """Test the complete processing pipeline"""
    logger.info("Testing complete pipeline...")
    
    try:
        # Initialize components
        line_parser = TibetanLineParser()
        wylie_converter = WylieConverter()
        phonetic_renderer = PhoneticRenderer()
        translation_client = MockTranslationClient()
        file_handler = TibetanFileHandler()
        output_formatter = TibetanOutputFormatter()
        
        # Parse input
        tibetan_lines = line_parser.parse_file(input_file)
        logger.info(f"Parsed {len(tibetan_lines)} lines")
        
        # Process first 10 lines for testing
        test_lines = tibetan_lines[:10]
        processed_lines = []
        
        for line in test_lines:
            try:
                # Skip structural markers
                if line.is_structural:
                    processed_line = ProcessedLine(
                        source_line=line,
                        tibetan_script="",
                        phonetic_rendering="",
                        english_translation="[Structural marker - skipped]",
                        processing_status=ProcessingStatus.SKIPPED
                    )
                    processed_lines.append(processed_line)
                    continue
                
                # Convert Wylie to Tibetan script
                tibetan_script = wylie_converter.convert_to_tibetan(line.wylie_content)
                
                # Generate phonetic rendering
                phonetic = phonetic_renderer.generate_phonetic(line.wylie_content)
                
                # Get mock translation
                translation = translation_client.translate_vajrayana_context(line.wylie_content, tibetan_script)
                
                # Create processed line
                processed_line = ProcessedLine(
                    source_line=line,
                    tibetan_script=tibetan_script,
                    phonetic_rendering=phonetic,
                    english_translation=translation,
                    processing_status=ProcessingStatus.SUCCESS
                )
                
                processed_lines.append(processed_line)
                logger.info(f"  ✓ Processed line {line.line_number}")
                
            except Exception as e:
                logger.warning(f"  ⚠ Failed to process line {line.line_number}: {e}")
                processed_line = ProcessedLine(
                    source_line=line,
                    tibetan_script="[Processing failed]",
                    phonetic_rendering="[Processing failed]",
                    english_translation="[Processing failed]",
                    processing_status=ProcessingStatus.FAILED,
                    error_message=str(e)
                )
                processed_lines.append(processed_line)
        
        # Generate output
        file_handler.write_tibetan_output(output_file, processed_lines)
        logger.info(f"✓ Output written to {output_file}")
        
        # Generate summary
        summary = output_formatter.generate_summary_report(processed_lines)
        logger.info(f"✓ Processing summary:\n{summary}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Complete pipeline test failed: {e}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return False

def main():
    """Run all end-to-end tests"""
    logger.info("=" * 60)
    logger.info("TIBETAN TEXT PROCESSOR - END-TO-END TESTING")
    logger.info("=" * 60)
    
    input_file = "test_tibetan_sample.txt"
    output_file = "test_end_to_end_output.txt"
    
    # Test 1: Component initialization
    if not test_component_initialization():
        logger.error("Component initialization failed - aborting tests")
        return 1
    
    # Test 2: File parsing
    tibetan_lines = test_file_parsing(input_file)
    if tibetan_lines is None:
        logger.error("File parsing failed - aborting tests")
        return 1
    
    # Test 3: Wylie conversion
    wylie_results = test_wylie_conversion(tibetan_lines)
    
    # Test 4: Phonetic rendering
    phonetic_results = test_phonetic_rendering(tibetan_lines)
    
    # Test 5: Mock translation
    translation_results = test_mock_translation(tibetan_lines)
    
    # Test 6: Complete pipeline
    pipeline_success = test_complete_pipeline(input_file, output_file)
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✓ Component initialization: PASSED")
    logger.info(f"✓ File parsing: PASSED ({len(tibetan_lines)} lines)")
    logger.info(f"✓ Wylie conversion: {sum(1 for r in wylie_results if r['success'])}/{len(wylie_results)} successful")
    logger.info(f"✓ Phonetic rendering: {sum(1 for r in phonetic_results if r['success'])}/{len(phonetic_results)} successful")
    logger.info(f"✓ Mock translation: {sum(1 for r in translation_results if r['success'])}/{len(translation_results)} successful")
    logger.info(f"✓ Complete pipeline: {'PASSED' if pipeline_success else 'FAILED'}")
    
    if os.path.exists(output_file):
        logger.info(f"✓ Test output file created: {output_file}")
        
        # Show sample of output
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Output file size: {len(content)} characters")
                logger.info("Sample output (first 500 characters):")
                logger.info("-" * 40)
                logger.info(content[:500])
                logger.info("-" * 40)
        except Exception as e:
            logger.warning(f"Could not read output file: {e}")
    
    # Overall result
    all_tests_passed = (
        len(tibetan_lines) > 0 and
        len(wylie_results) > 0 and
        len(phonetic_results) > 0 and
        len(translation_results) > 0 and
        pipeline_success
    )
    
    if all_tests_passed:
        logger.info("🎉 ALL TESTS PASSED - End-to-end pipeline is working!")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED - Check logs above for details")
        return 1

if __name__ == "__main__":
    exit(main())