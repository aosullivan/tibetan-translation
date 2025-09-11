#!/usr/bin/env python3
"""
Linguistic Accuracy Validation for Tibetan Text Processor

This module provides tools for validating the accuracy of:
1. Wylie to Tibetan script conversion
2. Phonetic rendering accuracy
3. Buddhist translation quality

Usage:
    python validation/linguistic_validation.py --test-all
    python validation/linguistic_validation.py --wylie-only
    python validation/linguistic_validation.py --phonetic-only
    python validation/linguistic_validation.py --translation-only
"""

import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wylie_converter import WylieConverter
from src.phonetic_renderer import PhoneticRenderer
from src.tibetan_translation_client import TibetanTranslationClient
from anthropic import Anthropic
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation/validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    input_text: str
    expected_output: str
    actual_output: str
    is_correct: bool
    confidence_score: float
    notes: str = ""

@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_confidence: float
    accuracy_percentage: float
    category_results: Dict[str, Dict[str, int]]

class WylieValidationTests:
    """Validation tests for Wylie to Tibetan conversion."""
    
    def __init__(self):
        self.converter = WylieConverter()
        self.test_cases = self._load_wylie_test_cases()
    
    def _load_wylie_test_cases(self) -> List[Dict]:
        """Load known Wylie-Tibetan conversion pairs for testing."""
        return [
            # Basic syllables
            {"wylie": "ka", "tibetan": "ཀ", "category": "basic_consonants"},
            {"wylie": "kha", "tibetan": "ཁ", "category": "basic_consonants"},
            {"wylie": "ga", "tibetan": "ག", "category": "basic_consonants"},
            {"wylie": "nga", "tibetan": "ང", "category": "basic_consonants"},
            
            # Vowels
            {"wylie": "ki", "tibetan": "ཀི", "category": "vowels"},
            {"wylie": "ku", "tibetan": "ཀུ", "category": "vowels"},
            {"wylie": "ke", "tibetan": "ཀེ", "category": "vowels"},
            {"wylie": "ko", "tibetan": "ཀོ", "category": "vowels"},
            
            # Common words
            {"wylie": "lha", "tibetan": "ལྷ", "category": "common_words"},
            {"wylie": "mo", "tibetan": "མོ", "category": "common_words"},
            {"wylie": "dbyangs", "tibetan": "དབྱངས", "category": "common_words"},
            {"wylie": "can", "tibetan": "ཅན", "category": "common_words"},
            {"wylie": "ma", "tibetan": "མ", "category": "common_words"},
            
            # Complex clusters
            {"wylie": "phyag", "tibetan": "ཕྱག", "category": "consonant_clusters"},
            {"wylie": "'tshal", "tibetan": "འཚལ", "category": "consonant_clusters"},
            {"wylie": "brgya", "tibetan": "བརྒྱ", "category": "consonant_clusters"},
            {"wylie": "sgrub", "tibetan": "སྒྲུབ", "category": "consonant_clusters"},
            
            # Phrases
            {"wylie": "lha mo", "tibetan": "ལྷ་མོ", "category": "phrases"},
            {"wylie": "dbyangs can ma", "tibetan": "དབྱངས་ཅན་མ", "category": "phrases"},
            {"wylie": "phyag 'tshal lo", "tibetan": "ཕྱག་འཚལ་ལོ", "category": "phrases"},
            
            # Sanskrit terms
            {"wylie": "oM", "tibetan": "ཨོཾ", "category": "sanskrit"},
            {"wylie": "hU~M", "tibetan": "ཧཱུྃ", "category": "sanskrit"},
            {"wylie": "swA hA", "tibetan": "སྭཱ་ཧཱ", "category": "sanskrit"},
            
            # Numbers
            {"wylie": "gcig", "tibetan": "གཅིག", "category": "numbers"},
            {"wylie": "gnyis", "tibetan": "གཉིས", "category": "numbers"},
            {"wylie": "gsum", "tibetan": "གསུམ", "category": "numbers"},
            
            # Punctuation and special characters
            {"wylie": "shad", "tibetan": "།", "category": "punctuation"},
            {"wylie": "||", "tibetan": "༎", "category": "punctuation"},
        ]
    
    def run_validation(self) -> List[ValidationResult]:
        """Run all Wylie conversion validation tests."""
        results = []
        
        logger.info(f"Running {len(self.test_cases)} Wylie conversion tests...")
        
        for i, test_case in enumerate(self.test_cases, 1):
            wylie = test_case["wylie"]
            expected = test_case["tibetan"]
            category = test_case["category"]
            
            try:
                actual = self.converter.convert_to_tibetan(wylie)
                is_correct = actual == expected
                confidence = 1.0 if is_correct else 0.0
                
                # Check for partial matches or reasonable alternatives
                if not is_correct:
                    confidence = self._calculate_similarity_score(expected, actual)
                
                result = ValidationResult(
                    test_name=f"wylie_{category}_{i}",
                    input_text=wylie,
                    expected_output=expected,
                    actual_output=actual,
                    is_correct=is_correct,
                    confidence_score=confidence,
                    notes=f"Category: {category}"
                )
                
                results.append(result)
                
                if is_correct:
                    logger.debug(f"✓ {wylie} → {actual}")
                else:
                    logger.warning(f"✗ {wylie} → {actual} (expected: {expected})")
                    
            except Exception as e:
                logger.error(f"Error testing {wylie}: {str(e)}")
                result = ValidationResult(
                    test_name=f"wylie_{category}_{i}",
                    input_text=wylie,
                    expected_output=expected,
                    actual_output=f"ERROR: {str(e)}",
                    is_correct=False,
                    confidence_score=0.0,
                    notes=f"Category: {category}, Error: {str(e)}"
                )
                results.append(result)
        
        return results
    
    def _calculate_similarity_score(self, expected: str, actual: str) -> float:
        """Calculate similarity score between expected and actual output."""
        if not expected or not actual:
            return 0.0
        
        # Simple character-level similarity
        expected_chars = set(expected)
        actual_chars = set(actual)
        
        if not expected_chars:
            return 0.0
        
        intersection = expected_chars.intersection(actual_chars)
        similarity = len(intersection) / len(expected_chars)
        
        return similarity

class PhoneticValidationTests:
    """Validation tests for phonetic rendering accuracy."""
    
    def __init__(self, style: str = "simplified"):
        self.renderer = PhoneticRenderer()
        self.style = style  # Store for reference but PhoneticRenderer doesn't use it
        self.test_cases = self._load_phonetic_test_cases()
    
    def _load_phonetic_test_cases(self) -> List[Dict]:
        """Load known Wylie-phonetic pairs for testing."""
        return [
            # Basic consonants
            {"wylie": "ka", "phonetic": "ka", "category": "basic_consonants"},
            {"wylie": "kha", "phonetic": "kha", "category": "basic_consonants"},
            {"wylie": "ga", "phonetic": "ga", "category": "basic_consonants"},
            {"wylie": "nga", "phonetic": "nga", "category": "basic_consonants"},
            
            # Aspirated consonants
            {"wylie": "pha", "phonetic": "pha", "category": "aspirated"},
            {"wylie": "tha", "phonetic": "tha", "category": "aspirated"},
            {"wylie": "tsha", "phonetic": "tsha", "category": "aspirated"},
            
            # Common words
            {"wylie": "lha", "phonetic": "lha", "category": "common_words"},
            {"wylie": "mo", "phonetic": "mo", "category": "common_words"},
            {"wylie": "dbyangs", "phonetic": "yang", "category": "common_words"},
            {"wylie": "can", "phonetic": "chen", "category": "common_words"},
            
            # Complex clusters
            {"wylie": "phyag", "phonetic": "chak", "category": "consonant_clusters"},
            {"wylie": "'tshal", "phonetic": "tsal", "category": "consonant_clusters"},
            {"wylie": "brgya", "phonetic": "gya", "category": "consonant_clusters"},
            {"wylie": "sgrub", "phonetic": "drub", "category": "consonant_clusters"},
            
            # Phrases
            {"wylie": "lha mo", "phonetic": "lha mo", "category": "phrases"},
            {"wylie": "dbyangs can ma", "phonetic": "yang chen ma", "category": "phrases"},
            {"wylie": "phyag 'tshal lo", "phonetic": "chak tsal lo", "category": "phrases"},
            
            # Sanskrit mantras
            {"wylie": "oM", "phonetic": "om", "category": "sanskrit"},
            {"wylie": "hU~M", "phonetic": "hung", "category": "sanskrit"},
            {"wylie": "swA hA", "phonetic": "soha", "category": "sanskrit"},
        ]
    
    def run_validation(self) -> List[ValidationResult]:
        """Run all phonetic rendering validation tests."""
        results = []
        
        logger.info(f"Running {len(self.test_cases)} phonetic rendering tests...")
        
        for i, test_case in enumerate(self.test_cases, 1):
            wylie = test_case["wylie"]
            expected = test_case["phonetic"]
            category = test_case["category"]
            
            try:
                actual = self.renderer.generate_phonetic(wylie)
                
                # Normalize for comparison (remove extra spaces, lowercase)
                expected_norm = expected.lower().strip()
                actual_norm = actual.lower().strip()
                
                is_correct = actual_norm == expected_norm
                confidence = 1.0 if is_correct else self._calculate_phonetic_similarity(expected_norm, actual_norm)
                
                result = ValidationResult(
                    test_name=f"phonetic_{category}_{i}",
                    input_text=wylie,
                    expected_output=expected,
                    actual_output=actual,
                    is_correct=is_correct,
                    confidence_score=confidence,
                    notes=f"Category: {category}"
                )
                
                results.append(result)
                
                if is_correct:
                    logger.debug(f"✓ {wylie} → {actual}")
                else:
                    logger.warning(f"✗ {wylie} → {actual} (expected: {expected})")
                    
            except Exception as e:
                logger.error(f"Error testing phonetic for {wylie}: {str(e)}")
                result = ValidationResult(
                    test_name=f"phonetic_{category}_{i}",
                    input_text=wylie,
                    expected_output=expected,
                    actual_output=f"ERROR: {str(e)}",
                    is_correct=False,
                    confidence_score=0.0,
                    notes=f"Category: {category}, Error: {str(e)}"
                )
                results.append(result)
        
        return results
    
    def _calculate_phonetic_similarity(self, expected: str, actual: str) -> float:
        """Calculate phonetic similarity score."""
        if not expected or not actual:
            return 0.0
        
        # Split into syllables and compare
        expected_parts = expected.split()
        actual_parts = actual.split()
        
        if len(expected_parts) != len(actual_parts):
            # Penalize different syllable counts
            return 0.3
        
        matches = 0
        for exp_part, act_part in zip(expected_parts, actual_parts):
            if exp_part == act_part:
                matches += 1
            elif self._syllable_similarity(exp_part, act_part) > 0.7:
                matches += 0.7
        
        return matches / len(expected_parts) if expected_parts else 0.0
    
    def _syllable_similarity(self, syl1: str, syl2: str) -> float:
        """Calculate similarity between two syllables."""
        if not syl1 or not syl2:
            return 0.0
        
        # Simple character overlap
        chars1 = set(syl1)
        chars2 = set(syl2)
        
        if not chars1:
            return 0.0
        
        overlap = len(chars1.intersection(chars2))
        return overlap / len(chars1)

class TranslationValidationTests:
    """Validation tests for Buddhist translation quality."""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        client = Anthropic(api_key=api_key)
        self.translation_client = TibetanTranslationClient(client)
        self.test_cases = self._load_translation_test_cases()
    
    def _load_translation_test_cases(self) -> List[Dict]:
        """Load known translation test cases."""
        return [
            # Basic religious terms
            {
                "wylie": "lha mo dbyangs can ma",
                "tibetan": "ལྷ་མོ་དབྱངས་ཅན་མ",
                "expected_concepts": ["goddess", "saraswati", "melodious"],
                "category": "deities"
            },
            {
                "wylie": "phyag 'tshal lo",
                "tibetan": "ཕྱག་འཚལ་ལོ",
                "expected_concepts": ["prostrate", "bow", "homage"],
                "category": "ritual_actions"
            },
            {
                "wylie": "sangs rgyas",
                "tibetan": "སངས་རྒྱས",
                "expected_concepts": ["buddha", "awakened", "enlightened"],
                "category": "spiritual_beings"
            },
            
            # Mantras (should be preserved, not translated)
            {
                "wylie": "oM ma Ni pad+me hU~M",
                "tibetan": "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ",
                "expected_concepts": ["om mani padme hung", "mantra"],
                "category": "mantras"
            },
            
            # Philosophical terms
            {
                "wylie": "stong pa nyid",
                "tibetan": "སྟོང་པ་ཉིད",
                "expected_concepts": ["emptiness", "voidness", "sunyata"],
                "category": "philosophy"
            },
            {
                "wylie": "byang chub sems dpa'",
                "tibetan": "བྱང་ཆུབ་སེམས་དཔའ",
                "expected_concepts": ["bodhisattva", "enlightenment", "being"],
                "category": "spiritual_beings"
            },
            
            # Ritual instructions
            {
                "wylie": "sgrub thabs",
                "tibetan": "སྒྲུབ་ཐབས",
                "expected_concepts": ["sadhana", "practice", "method"],
                "category": "practice_terms"
            },
            {
                "wylie": "bsnyen pa",
                "tibetan": "བསྙེན་པ",
                "expected_concepts": ["approach", "preliminary", "practice"],
                "category": "practice_terms"
            },
        ]
    
    def run_validation(self) -> List[ValidationResult]:
        """Run translation quality validation tests."""
        results = []
        
        logger.info(f"Running {len(self.test_cases)} translation quality tests...")
        
        for i, test_case in enumerate(self.test_cases, 1):
            wylie = test_case["wylie"]
            tibetan = test_case["tibetan"]
            expected_concepts = test_case["expected_concepts"]
            category = test_case["category"]
            
            try:
                # Get translation
                translation = self.translation_client.translate_vajrayana_context(wylie, tibetan)
                
                # Evaluate translation quality
                is_correct, confidence, notes = self._evaluate_translation_quality(
                    translation, expected_concepts, category
                )
                
                result = ValidationResult(
                    test_name=f"translation_{category}_{i}",
                    input_text=f"{wylie} ({tibetan})",
                    expected_output=f"Should contain: {', '.join(expected_concepts)}",
                    actual_output=translation,
                    is_correct=is_correct,
                    confidence_score=confidence,
                    notes=notes
                )
                
                results.append(result)
                
                if is_correct:
                    logger.debug(f"✓ {wylie} → {translation[:50]}...")
                else:
                    logger.warning(f"✗ {wylie} → {translation[:50]}... (missing concepts)")
                    
            except Exception as e:
                logger.error(f"Error testing translation for {wylie}: {str(e)}")
                result = ValidationResult(
                    test_name=f"translation_{category}_{i}",
                    input_text=f"{wylie} ({tibetan})",
                    expected_output=f"Should contain: {', '.join(expected_concepts)}",
                    actual_output=f"ERROR: {str(e)}",
                    is_correct=False,
                    confidence_score=0.0,
                    notes=f"Category: {category}, Error: {str(e)}"
                )
                results.append(result)
        
        return results
    
    def _evaluate_translation_quality(self, translation: str, expected_concepts: List[str], 
                                    category: str) -> Tuple[bool, float, str]:
        """Evaluate the quality of a translation."""
        translation_lower = translation.lower()
        
        # Check for expected concepts
        found_concepts = []
        for concept in expected_concepts:
            if concept.lower() in translation_lower:
                found_concepts.append(concept)
        
        # Calculate confidence based on concept coverage
        concept_coverage = len(found_concepts) / len(expected_concepts)
        
        # Additional quality checks
        quality_score = concept_coverage
        notes = []
        
        # Check for appropriate Buddhist context
        buddhist_terms = ["dharma", "buddha", "enlighten", "practice", "meditation", 
                         "wisdom", "compassion", "mantra", "deity", "blessing"]
        
        has_buddhist_context = any(term in translation_lower for term in buddhist_terms)
        if has_buddhist_context:
            quality_score += 0.1
            notes.append("Contains Buddhist context")
        
        # Check for mantras (should not be literally translated)
        if category == "mantras":
            if any(word in translation_lower for word in ["om", "mani", "padme", "hung"]):
                quality_score += 0.2
                notes.append("Mantra preserved appropriately")
            elif "mantra" in translation_lower:
                quality_score += 0.1
                notes.append("Identified as mantra")
        
        # Check translation length (not too short, not too verbose)
        word_count = len(translation.split())
        if 3 <= word_count <= 20:
            quality_score += 0.05
        elif word_count > 50:
            quality_score -= 0.1
            notes.append("Translation may be too verbose")
        elif word_count < 2:
            quality_score -= 0.2
            notes.append("Translation too brief")
        
        # Final assessment
        is_correct = quality_score >= 0.7
        confidence = min(quality_score, 1.0)
        
        notes_str = f"Found concepts: {found_concepts}. " + "; ".join(notes)
        
        return is_correct, confidence, notes_str

class ValidationRunner:
    """Main validation runner that coordinates all tests."""
    
    def __init__(self):
        self.wylie_tests = WylieValidationTests()
        self.phonetic_tests = PhoneticValidationTests()
        self.translation_tests = None  # Initialize only if API key available
        
        # Try to initialize translation tests
        try:
            self.translation_tests = TranslationValidationTests()
        except ValueError as e:
            logger.warning(f"Translation tests disabled: {e}")
    
    def run_all_validations(self) -> ValidationSummary:
        """Run all validation tests and return summary."""
        all_results = []
        
        # Run Wylie conversion tests
        logger.info("=== Running Wylie Conversion Validation ===")
        wylie_results = self.wylie_tests.run_validation()
        all_results.extend(wylie_results)
        
        # Run phonetic rendering tests
        logger.info("=== Running Phonetic Rendering Validation ===")
        phonetic_results = self.phonetic_tests.run_validation()
        all_results.extend(phonetic_results)
        
        # Run translation tests if available
        if self.translation_tests:
            logger.info("=== Running Translation Quality Validation ===")
            translation_results = self.translation_tests.run_validation()
            all_results.extend(translation_results)
        else:
            logger.warning("Skipping translation tests (API key not available)")
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        # Save detailed results
        self._save_results(all_results, summary)
        
        return summary
    
    def run_wylie_only(self) -> ValidationSummary:
        """Run only Wylie conversion validation."""
        logger.info("=== Running Wylie Conversion Validation Only ===")
        results = self.wylie_tests.run_validation()
        summary = self._generate_summary(results)
        self._save_results(results, summary, "wylie_validation")
        return summary
    
    def run_phonetic_only(self) -> ValidationSummary:
        """Run only phonetic rendering validation."""
        logger.info("=== Running Phonetic Rendering Validation Only ===")
        results = self.phonetic_tests.run_validation()
        summary = self._generate_summary(results)
        self._save_results(results, summary, "phonetic_validation")
        return summary
    
    def run_translation_only(self) -> ValidationSummary:
        """Run only translation quality validation."""
        if not self.translation_tests:
            raise ValueError("Translation tests not available (API key required)")
        
        logger.info("=== Running Translation Quality Validation Only ===")
        results = self.translation_tests.run_validation()
        summary = self._generate_summary(results)
        self._save_results(results, summary, "translation_validation")
        return summary
    
    def _generate_summary(self, results: List[ValidationResult]) -> ValidationSummary:
        """Generate summary statistics from validation results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.is_correct)
        failed_tests = total_tests - passed_tests
        
        average_confidence = sum(r.confidence_score for r in results) / total_tests if total_tests > 0 else 0.0
        accuracy_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
        
        # Category breakdown
        category_results = {}
        for result in results:
            category = result.notes.split("Category: ")[1].split(",")[0] if "Category: " in result.notes else "unknown"
            
            if category not in category_results:
                category_results[category] = {"total": 0, "passed": 0, "failed": 0}
            
            category_results[category]["total"] += 1
            if result.is_correct:
                category_results[category]["passed"] += 1
            else:
                category_results[category]["failed"] += 1
        
        return ValidationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_confidence=average_confidence,
            accuracy_percentage=accuracy_percentage,
            category_results=category_results
        )
    
    def _save_results(self, results: List[ValidationResult], summary: ValidationSummary, 
                     filename_prefix: str = "validation") -> None:
        """Save validation results to files."""
        # Create validation directory if it doesn't exist
        Path("validation").mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        results_data = {
            "summary": {
                "total_tests": summary.total_tests,
                "passed_tests": summary.passed_tests,
                "failed_tests": summary.failed_tests,
                "average_confidence": summary.average_confidence,
                "accuracy_percentage": summary.accuracy_percentage,
                "category_results": summary.category_results
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "input_text": r.input_text,
                    "expected_output": r.expected_output,
                    "actual_output": r.actual_output,
                    "is_correct": r.is_correct,
                    "confidence_score": r.confidence_score,
                    "notes": r.notes
                }
                for r in results
            ]
        }
        
        json_file = f"validation/{filename_prefix}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {json_file}")
        
        # Save human-readable report
        report_file = f"validation/{filename_prefix}_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(results, summary))
        
        logger.info(f"Human-readable report saved to {report_file}")
    
    def _generate_markdown_report(self, results: List[ValidationResult], 
                                summary: ValidationSummary) -> str:
        """Generate a markdown report of validation results."""
        report_lines = [
            "# Tibetan Text Processor - Linguistic Validation Report",
            "",
            f"**Generated**: {Path(__file__).stat().st_mtime}",
            f"**Total Tests**: {summary.total_tests}",
            f"**Accuracy**: {summary.accuracy_percentage:.1f}%",
            f"**Average Confidence**: {summary.average_confidence:.3f}",
            "",
            "## Summary",
            "",
            f"- ✅ **Passed**: {summary.passed_tests} tests",
            f"- ❌ **Failed**: {summary.failed_tests} tests",
            f"- 📊 **Overall Accuracy**: {summary.accuracy_percentage:.1f}%",
            "",
            "## Results by Category",
            ""
        ]
        
        # Category breakdown
        for category, stats in summary.category_results.items():
            accuracy = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report_lines.extend([
                f"### {category.replace('_', ' ').title()}",
                f"- Total: {stats['total']} tests",
                f"- Passed: {stats['passed']} ({accuracy:.1f}%)",
                f"- Failed: {stats['failed']}",
                ""
            ])
        
        # Failed tests details
        failed_results = [r for r in results if not r.is_correct]
        if failed_results:
            report_lines.extend([
                "## Failed Tests Details",
                ""
            ])
            
            for result in failed_results:
                report_lines.extend([
                    f"### {result.test_name}",
                    f"**Input**: `{result.input_text}`",
                    f"**Expected**: `{result.expected_output}`",
                    f"**Actual**: `{result.actual_output}`",
                    f"**Confidence**: {result.confidence_score:.3f}",
                    f"**Notes**: {result.notes}",
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if summary.accuracy_percentage < 80:
            report_lines.append("- ⚠️ **Low accuracy detected**. Review conversion rules and test cases.")
        
        if summary.average_confidence < 0.7:
            report_lines.append("- ⚠️ **Low confidence scores**. Consider improving algorithms or test cases.")
        
        # Category-specific recommendations
        for category, stats in summary.category_results.items():
            accuracy = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            if accuracy < 70:
                report_lines.append(f"- ⚠️ **{category}** needs attention (accuracy: {accuracy:.1f}%)")
        
        if summary.accuracy_percentage >= 90:
            report_lines.append("- ✅ **Excellent accuracy**. System performing well.")
        
        return "\n".join(report_lines)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Validate Tibetan Text Processor linguistic accuracy")
    parser.add_argument("--test-all", action="store_true", help="Run all validation tests")
    parser.add_argument("--wylie-only", action="store_true", help="Run only Wylie conversion tests")
    parser.add_argument("--phonetic-only", action="store_true", help="Run only phonetic rendering tests")
    parser.add_argument("--translation-only", action="store_true", help="Run only translation quality tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = ValidationRunner()
    
    try:
        if args.wylie_only:
            summary = runner.run_wylie_only()
        elif args.phonetic_only:
            summary = runner.run_phonetic_only()
        elif args.translation_only:
            summary = runner.run_translation_only()
        else:
            summary = runner.run_all_validations()
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary.total_tests}")
        print(f"Passed: {summary.passed_tests}")
        print(f"Failed: {summary.failed_tests}")
        print(f"Accuracy: {summary.accuracy_percentage:.1f}%")
        print(f"Average Confidence: {summary.average_confidence:.3f}")
        print("="*60)
        
        # Exit with appropriate code
        if summary.accuracy_percentage >= 90:
            print("✅ Excellent accuracy!")
            return 0
        elif summary.accuracy_percentage >= 70:
            print("⚠️ Acceptable accuracy, but room for improvement.")
            return 1
        else:
            print("❌ Low accuracy detected. Review needed.")
            return 2
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return 3

if __name__ == "__main__":
    exit(main())