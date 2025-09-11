#!/usr/bin/env python3
"""
Accuracy validation test for Tibetan text processor components.
Tests the quality and accuracy of Wylie conversion, phonetic rendering, and output formatting.
"""

import sys
import logging
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.append('../src')

from src.tibetan_line_parser import TibetanLineParser
from src.wylie_converter import WylieConverter
from src.phonetic_renderer import PhoneticRenderer
from src.tibetan_output_formatter import TibetanOutputFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wylie_conversion_accuracy():
    """Test Wylie to Tibetan script conversion accuracy with known examples"""
    logger.info("Testing Wylie conversion accuracy...")
    
    converter = WylieConverter()
    
    # Test cases with known correct conversions
    test_cases = [
        ("om mani padme hum", "ཨོམ་མ་ནི་པདྨེ་ཧཱུམ"),
        ("lha mo dbyangs can ma", "ལྷ་མོ་དབྱངས་ཅན་མ"),
        ("phyag 'tshal lo", "ཕྱག་འཚལ་ལོ"),
        ("sgrub thabs", "སྒྲུབ་ཐབས"),
        ("bcom ldan 'das", "བཅོམ་ལྡན་འདས"),
        ("sangs rgyas", "སངས་རྒྱས"),
        ("byang chub sems dpa'", "བྱང་ཆུབ་སེམས་དཔའ"),
    ]
    
    results = []
    for wylie, expected_tibetan in test_cases:
        try:
            actual_tibetan = converter.convert_to_tibetan(wylie)
            is_correct = actual_tibetan == expected_tibetan
            results.append({
                'wylie': wylie,
                'expected': expected_tibetan,
                'actual': actual_tibetan,
                'correct': is_correct
            })
            
            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} {wylie} → {actual_tibetan}")
            if not is_correct:
                logger.info(f"    Expected: {expected_tibetan}")
        except Exception as e:
            logger.error(f"  ✗ {wylie} → ERROR: {e}")
            results.append({
                'wylie': wylie,
                'expected': expected_tibetan,
                'actual': f"ERROR: {e}",
                'correct': False
            })
    
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = (correct_count / len(results)) * 100
    logger.info(f"Wylie conversion accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    
    return results

def test_phonetic_rendering_quality():
    """Test phonetic rendering quality and consistency"""
    logger.info("Testing phonetic rendering quality...")
    
    renderer = PhoneticRenderer()
    
    # Test cases for phonetic quality
    test_cases = [
        ("om mani padme hum", ["ohm", "ma-ni", "pad-me", "hoom"]),
        ("lha mo dbyangs can ma", ["lha", "mo", "dbyangs", "chan", "ma"]),
        ("phyag 'tshal lo", ["phyag", "tshal", "lo"]),
        ("sgrub thabs", ["sgrub", "thabs"]),
        ("bcom ldan 'das", ["chom", "den", "de"]),
    ]
    
    results = []
    for wylie, expected_elements in test_cases:
        try:
            phonetic = renderer.generate_phonetic(wylie)
            
            # Check if expected elements are present
            elements_found = sum(1 for elem in expected_elements if elem.lower() in phonetic.lower())
            quality_score = (elements_found / len(expected_elements)) * 100
            
            results.append({
                'wylie': wylie,
                'phonetic': phonetic,
                'expected_elements': expected_elements,
                'elements_found': elements_found,
                'quality_score': quality_score
            })
            
            logger.info(f"  {wylie} → {phonetic} (quality: {quality_score:.0f}%)")
            
        except Exception as e:
            logger.error(f"  ✗ {wylie} → ERROR: {e}")
            results.append({
                'wylie': wylie,
                'phonetic': f"ERROR: {e}",
                'expected_elements': expected_elements,
                'elements_found': 0,
                'quality_score': 0
            })
    
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    logger.info(f"Average phonetic quality: {avg_quality:.1f}%")
    
    return results

def test_buddhist_terminology_handling():
    """Test handling of Buddhist terminology"""
    logger.info("Testing Buddhist terminology handling...")
    
    converter = WylieConverter()
    renderer = PhoneticRenderer()
    
    # Buddhist terms that should be handled specially
    buddhist_terms = [
        "bcom ldan 'das",  # Buddha
        "sangs rgyas",     # Buddha
        "byang chub sems dpa'",  # Bodhisattva
        "dbyangs can ma",  # Saraswati
        "sgrub thabs",     # Sadhana
        "rdo rje",         # Vajra
        "pad+ma",          # Lotus
        "thugs rje",       # Compassion
        "shes rab",        # Wisdom
    ]
    
    results = []
    for term in buddhist_terms:
        try:
            tibetan = converter.convert_to_tibetan(term)
            phonetic = renderer.generate_phonetic(term)
            
            # Check if conversion produces reasonable output
            has_tibetan = len(tibetan) > 0 and any(ord(c) >= 0x0F00 and ord(c) <= 0x0FFF for c in tibetan)
            has_phonetic = len(phonetic) > 0 and phonetic != term
            
            results.append({
                'term': term,
                'tibetan': tibetan,
                'phonetic': phonetic,
                'has_tibetan': has_tibetan,
                'has_phonetic': has_phonetic,
                'success': has_tibetan and has_phonetic
            })
            
            status = "✓" if has_tibetan and has_phonetic else "⚠"
            logger.info(f"  {status} {term}")
            logger.info(f"    Tibetan: {tibetan}")
            logger.info(f"    Phonetic: {phonetic}")
            
        except Exception as e:
            logger.error(f"  ✗ {term} → ERROR: {e}")
            results.append({
                'term': term,
                'tibetan': f"ERROR: {e}",
                'phonetic': f"ERROR: {e}",
                'has_tibetan': False,
                'has_phonetic': False,
                'success': False
            })
    
    success_count = sum(1 for r in results if r['success'])
    success_rate = (success_count / len(results)) * 100
    logger.info(f"Buddhist terminology handling: {success_count}/{len(results)} ({success_rate:.1f}%)")
    
    return results

def test_output_formatting_quality():
    """Test output formatting quality and structure"""
    logger.info("Testing output formatting quality...")
    
    formatter = TibetanOutputFormatter()
    
    # Create sample processed lines for testing
    from src.tibetan_models import TibetanLine, ProcessedLine, ProcessingStatus
    
    sample_lines = [
        ProcessedLine(
            source_line=TibetanLine(1, "lha mo dbyangs can ma", "lha mo dbyangs can ma", False, []),
            tibetan_script="ལྷ་མོ་དབྱངས་ཅན་མ",
            phonetic_rendering="lha mo dbyangs chan ma",
            english_translation="Goddess Saraswati",
            processing_status=ProcessingStatus.SUCCESS
        ),
        ProcessedLine(
            source_line=TibetanLine(2, "(kha)@#", "", True, ["(kha)@#"]),
            tibetan_script="",
            phonetic_rendering="",
            english_translation="[Structural marker - skipped]",
            processing_status=ProcessingStatus.SKIPPED
        ),
    ]
    
    try:
        # Test summary report generation
        summary = formatter.generate_summary_report(sample_lines)
        
        # Check summary quality
        has_statistics = "Total lines processed" in summary
        has_status_breakdown = "Successfully completed" in summary
        has_content_analysis = "Content Analysis" in summary
        has_encoding_info = "utf-8" in summary
        
        quality_checks = [
            ("Statistics present", has_statistics),
            ("Status breakdown", has_status_breakdown),
            ("Content analysis", has_content_analysis),
            ("Encoding info", has_encoding_info),
        ]
        
        logger.info("Output formatting quality checks:")
        for check_name, passed in quality_checks:
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check_name}")
        
        passed_checks = sum(1 for _, passed in quality_checks if passed)
        quality_score = (passed_checks / len(quality_checks)) * 100
        
        logger.info(f"Output formatting quality: {quality_score:.0f}%")
        
        return {
            'summary': summary,
            'quality_checks': quality_checks,
            'quality_score': quality_score,
            'success': quality_score >= 75
        }
        
    except Exception as e:
        logger.error(f"Output formatting test failed: {e}")
        return {
            'summary': f"ERROR: {e}",
            'quality_checks': [],
            'quality_score': 0,
            'success': False
        }

def main():
    """Run all accuracy validation tests"""
    logger.info("=" * 60)
    logger.info("TIBETAN TEXT PROCESSOR - ACCURACY VALIDATION")
    logger.info("=" * 60)
    
    # Run all validation tests
    wylie_results = test_wylie_conversion_accuracy()
    phonetic_results = test_phonetic_rendering_quality()
    buddhist_results = test_buddhist_terminology_handling()
    formatting_results = test_output_formatting_quality()
    
    # Calculate overall scores
    wylie_accuracy = (sum(1 for r in wylie_results if r['correct']) / len(wylie_results)) * 100
    phonetic_quality = sum(r['quality_score'] for r in phonetic_results) / len(phonetic_results)
    buddhist_success = (sum(1 for r in buddhist_results if r['success']) / len(buddhist_results)) * 100
    formatting_quality = formatting_results['quality_score']
    
    # Overall assessment
    overall_score = (wylie_accuracy + phonetic_quality + buddhist_success + formatting_quality) / 4
    
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Wylie Conversion Accuracy: {wylie_accuracy:.1f}%")
    logger.info(f"Phonetic Rendering Quality: {phonetic_quality:.1f}%")
    logger.info(f"Buddhist Terminology Handling: {buddhist_success:.1f}%")
    logger.info(f"Output Formatting Quality: {formatting_quality:.1f}%")
    logger.info("-" * 40)
    logger.info(f"Overall Quality Score: {overall_score:.1f}%")
    
    # Determine pass/fail
    if overall_score >= 75:
        logger.info("🎉 VALIDATION PASSED - System meets quality standards!")
        return 0
    elif overall_score >= 50:
        logger.info("⚠️  VALIDATION PARTIAL - System needs improvement")
        return 1
    else:
        logger.error("❌ VALIDATION FAILED - System requires significant fixes")
        return 2

if __name__ == "__main__":
    exit(main())