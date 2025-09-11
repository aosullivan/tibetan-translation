# Tibetan Text Processor - Validation Framework

This directory contains the linguistic accuracy validation framework for the Tibetan Text Processor.

## Overview

The validation framework provides comprehensive testing and expert review capabilities for:

1. **Wylie to Tibetan Script Conversion** - Accuracy of Unicode Tibetan output
2. **Phonetic Rendering** - Quality of pronunciation guides
3. **Buddhist Translation** - Contextual accuracy and cultural appropriateness

## Files in this Directory

### Core Validation Tools

- **`linguistic_validation.py`** - Main validation script with automated tests
- **`validation_config.json`** - Configuration settings for validation parameters
- **`expert_review_guidelines.md`** - Comprehensive guidelines for expert reviewers

### Generated Reports

- **`wylie_validation_report.md`** - Wylie conversion accuracy results
- **`phonetic_validation_report.md`** - Phonetic rendering accuracy results
- **`wylie_validation_results.json`** - Detailed JSON results for Wylie tests
- **`phonetic_validation_results.json`** - Detailed JSON results for phonetic tests
- **`validation.log`** - Detailed logging of validation process

### Documentation

- **`VALIDATION_SUMMARY.md`** - Comprehensive summary of all validation results
- **`README.md`** - This file

## Quick Start

### Running Validation Tests

```bash
# Run all available tests
python validation/linguistic_validation.py --test-all

# Run specific component tests
python validation/linguistic_validation.py --wylie-only
python validation/linguistic_validation.py --phonetic-only
python validation/linguistic_validation.py --translation-only

# Enable verbose logging
python validation/linguistic_validation.py --test-all --verbose
```

### Setting Up for Translation Tests

Translation quality tests require an Anthropic API key:

```bash
# Set up API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run translation tests
python validation/linguistic_validation.py --translation-only
```

## Current Validation Status

### Latest Results Summary

| Component | Accuracy | Status | Priority |
|-----------|----------|--------|----------|
| **Wylie Conversion** | 89.3% | ⚠️ Good | Medium |
| **Phonetic Rendering** | 47.6% | ❌ Needs Work | High |
| **Translation Quality** | Pending | ⏸️ Requires Review | High |

### Key Findings

**Wylie Conversion (89.3%)**:
- ✅ Excellent performance on basic vocabulary and common phrases
- ⚠️ Minor issues with Sanskrit terms and punctuation marks
- 🔧 Specific fixes identified and documented

**Phonetic Rendering (47.6%)**:
- ❌ Significant issues with consonant cluster simplification
- ❌ Inconsistent vowel rendering
- 🔧 Comprehensive improvements needed

**Translation Quality**:
- ⏸️ Awaiting expert review with API access
- 📋 Test framework ready for evaluation
- 👥 Expert reviewer guidelines prepared

## Validation Framework Features

### Automated Testing

- **Comprehensive Test Coverage**: 28 Wylie tests, 21 phonetic tests
- **Category-Based Organization**: Tests grouped by linguistic features
- **Confidence Scoring**: Similarity analysis for partial matches
- **Detailed Reporting**: JSON and Markdown output formats

### Expert Review Support

- **Structured Guidelines**: Clear evaluation criteria and methodology
- **Qualification Requirements**: Specified reviewer credentials
- **Scoring Framework**: Weighted evaluation system
- **Cultural Sensitivity**: Buddhist context and reverence considerations

### Quality Metrics

- **Accuracy Thresholds**: Defined standards for acceptable performance
- **Confidence Scoring**: Quantitative assessment of result quality
- **Category Analysis**: Performance breakdown by linguistic features
- **Trend Tracking**: Historical accuracy monitoring capability

## Using the Validation Framework

### For Developers

1. **Run Tests Before Commits**:
   ```bash
   python validation/linguistic_validation.py --test-all
   ```

2. **Check Specific Components**:
   ```bash
   # After modifying Wylie converter
   python validation/linguistic_validation.py --wylie-only
   
   # After updating phonetic renderer
   python validation/linguistic_validation.py --phonetic-only
   ```

3. **Review Detailed Results**:
   ```bash
   # Check JSON results for programmatic analysis
   cat validation/wylie_validation_results.json
   
   # Read human-friendly reports
   cat validation/wylie_validation_report.md
   ```

### For Linguistic Experts

1. **Review Guidelines**: Read `expert_review_guidelines.md` thoroughly
2. **Run Validation**: Execute tests to see current performance
3. **Evaluate Samples**: Use provided test cases and add your own
4. **Provide Feedback**: Use structured scoring and recommendation format

### For Project Managers

1. **Monitor Quality**: Check `VALIDATION_SUMMARY.md` for current status
2. **Track Progress**: Compare validation reports over time
3. **Plan Improvements**: Use identified issues to prioritize development
4. **Coordinate Reviews**: Schedule expert validation cycles

## Validation Schedule

### Automated Testing
- **Daily**: Core Wylie conversion tests (CI/CD integration)
- **Weekly**: Full phonetic rendering validation
- **Monthly**: Translation quality spot checks (with API access)

### Expert Review
- **Quarterly**: Comprehensive linguistic accuracy review
- **Annually**: Full cultural and contextual appropriateness assessment
- **As Needed**: Validation of major updates or new features

## Extending the Validation Framework

### Adding New Test Cases

```python
# In linguistic_validation.py, add to appropriate test class:
new_test_cases = [
    {
        "wylie": "your_wylie_text",
        "tibetan": "expected_tibetan_output",
        "category": "test_category"
    }
]
```

### Custom Validation Categories

```python
# Create new validation class:
class CustomValidationTests:
    def __init__(self):
        self.test_cases = self._load_custom_test_cases()
    
    def run_validation(self) -> List[ValidationResult]:
        # Implement custom validation logic
        pass
```

### Integration with CI/CD

```yaml
# Example GitHub Actions workflow
name: Linguistic Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Validation
        run: python validation/linguistic_validation.py --wylie-only
```

## Expert Reviewer Onboarding

### Qualifications Required

**Primary Reviewers**:
- PhD in Tibetan Studies, Buddhist Studies, or Linguistics
- Native or near-native Tibetan language proficiency
- Experience with Wylie transliteration systems
- Specialization in Tibetan Buddhist texts

**Secondary Reviewers**:
- Advanced Tibetan language students or teachers
- Buddhist practitioners with extensive Tibetan study
- Cultural preservation specialists

### Review Process

1. **Preparation**: Study validation guidelines and sample outputs
2. **Testing**: Run validation tools to understand current performance
3. **Evaluation**: Systematic review using provided scoring framework
4. **Documentation**: Structured feedback using report templates
5. **Follow-up**: Participate in improvement validation cycles

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure you're in the project root directory
cd /path/to/tibetan-text-processor
python validation/linguistic_validation.py --wylie-only
```

**API Key Issues**:
```bash
# Check API key setup
echo $ANTHROPIC_API_KEY
# or
cat .env | grep ANTHROPIC_API_KEY
```

**Permission Errors**:
```bash
# Ensure validation directory is writable
chmod 755 validation/
```

### Getting Help

1. **Check Logs**: Review `validation/validation.log` for detailed error information
2. **Run Verbose**: Use `--verbose` flag for additional debugging output
3. **Test Components**: Run individual component tests to isolate issues
4. **Review Documentation**: Check `expert_review_guidelines.md` for detailed guidance

## Contributing to Validation

### Reporting Issues

When reporting validation issues:
1. Include specific test cases that fail
2. Provide expected vs. actual outputs
3. Specify which component (Wylie, phonetic, translation)
4. Include relevant cultural or linguistic context

### Suggesting Improvements

1. **Test Cases**: Contribute additional test cases for edge cases
2. **Evaluation Criteria**: Suggest improvements to scoring methods
3. **Expert Guidelines**: Enhance reviewer documentation
4. **Automation**: Propose additional automated validation features

## Conclusion

This validation framework ensures the Tibetan Text Processor maintains high standards of linguistic accuracy and cultural appropriateness. Regular use of these tools, combined with expert review, provides confidence in the system's output quality for processing sacred Tibetan Buddhist texts.

The framework is designed to be:
- **Comprehensive**: Covering all major system components
- **Extensible**: Easy to add new tests and validation criteria
- **Expert-Friendly**: Supporting human review with clear guidelines
- **Automated**: Enabling continuous quality monitoring
- **Culturally Sensitive**: Respecting the sacred nature of Tibetan Buddhist texts

For questions or assistance with the validation framework, please refer to the detailed documentation in this directory or contact the project maintainers.