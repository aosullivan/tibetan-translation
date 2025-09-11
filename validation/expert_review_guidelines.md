# Expert Review Guidelines for Tibetan Text Processor

This document provides guidelines for linguistic experts to review and validate the accuracy of the Tibetan Text Processor's output.

## Overview

The Tibetan Text Processor performs three main functions that require expert validation:

1. **Wylie to Tibetan Script Conversion**: Converting romanized Wylie transliteration to Unicode Tibetan script
2. **Phonetic Rendering**: Generating readable phonetic approximations of Tibetan pronunciation
3. **Buddhist Translation**: Providing contextually accurate English translations with Vajrayana Buddhist expertise

## Review Process

### Phase 1: Automated Validation

Before expert review, run the automated validation suite:

```bash
# Run all automated tests
python validation/linguistic_validation.py --test-all

# Review results
cat validation/validation_report.md
```

### Phase 2: Expert Review Categories

#### A. Wylie Conversion Accuracy

**Scope**: Review conversion of Wylie transliteration to Unicode Tibetan script

**Key Areas to Evaluate**:

1. **Basic Consonants and Vowels**
   - Verify standard Wylie-to-Tibetan mappings
   - Check vowel diacritics (i, u, e, o, etc.)
   - Validate consonant accuracy

2. **Consonant Clusters**
   - Complex clusters (e.g., `sgrub` → སྒྲུབ)
   - Prefix consonants (e.g., `dbyangs` → དབྱངས)
   - Suffix consonants (e.g., `phyag` → ཕྱག)

3. **Special Characters**
   - Apostrophes and their positioning
   - Sanskrit transliterations
   - Punctuation marks

**Sample Test Cases**:
```
Input: lha mo dbyangs can ma la phyag 'tshal lo
Expected: ལྷ་མོ་དབྱངས་ཅན་མ་ལ་ཕྱག་འཚལ་ལོ
```

**Evaluation Criteria**:
- ✅ **Excellent (90-100%)**: All characters correctly converted
- ⚠️ **Good (70-89%)**: Minor errors in complex clusters
- ❌ **Needs Work (<70%)**: Significant conversion errors

#### B. Phonetic Rendering Accuracy

**Scope**: Review phonetic approximations for pronunciation accuracy

**Key Areas to Evaluate**:

1. **Consonant Pronunciation**
   - Aspirated vs. unaspirated consonants
   - Consonant cluster simplification
   - Silent letters handling

2. **Vowel Sounds**
   - Accurate vowel representation
   - Tone considerations (if applicable)
   - Syllable boundaries

3. **Regional Variations**
   - Central Tibetan pronunciation (primary)
   - Acknowledgment of dialectal differences

**Sample Test Cases**:
```
Wylie: dbyangs can ma
Phonetic: yang chen ma
Evaluation: Check if 'dbyangs' → 'yang' is accurate
```

**Evaluation Criteria**:
- ✅ **Excellent**: Phonetics match standard Central Tibetan pronunciation
- ⚠️ **Good**: Minor variations but generally accurate
- ❌ **Needs Work**: Significant mispronunciations or unclear phonetics

#### C. Buddhist Translation Quality

**Scope**: Review English translations for accuracy and appropriate Buddhist context

**Key Areas to Evaluate**:

1. **Technical Terminology**
   - Accurate translation of Buddhist technical terms
   - Appropriate use of Sanskrit terms vs. English equivalents
   - Consistency in terminology usage

2. **Contextual Accuracy**
   - Vajrayana Buddhist context maintained
   - Appropriate reverence for sacred content
   - Cultural sensitivity in translation choices

3. **Mantra and Sacred Text Handling**
   - Mantras preserved rather than literally translated
   - Sacred syllables handled appropriately
   - Ritual instructions clearly conveyed

**Sample Test Cases**:
```
Wylie: lha mo dbyangs can ma la phyag 'tshal lo
Translation: "I prostrate to the goddess Saraswati"
Evaluation: Check accuracy, reverence, and Buddhist context
```

**Evaluation Criteria**:
- ✅ **Excellent**: Accurate, contextually appropriate, maintains Buddhist reverence
- ⚠️ **Good**: Generally accurate but may lack nuance or context
- ❌ **Needs Work**: Inaccurate, inappropriate, or lacks Buddhist understanding

## Expert Review Methodology

### Step 1: Preparation

1. **Review Sample Texts**: Examine the input Wylie texts to understand content type
2. **Check Output Format**: Familiarize yourself with the three-column output format
3. **Understand Context**: Note that texts are primarily Vajrayana Buddhist practices

### Step 2: Systematic Review

#### Wylie Conversion Review

1. **Select Representative Samples**:
   - 20-30 lines covering different complexity levels
   - Include basic words, complex clusters, and phrases
   - Cover different text types (mantras, instructions, prayers)

2. **Character-by-Character Verification**:
   - Compare each Wylie syllable with Tibetan output
   - Check consonant clusters and vowel diacritics
   - Verify punctuation and spacing

3. **Document Issues**:
   ```
   Line X: Input "sgrub thabs" → Output "སྒྲུབ་ཐབས" ✓ Correct
   Line Y: Input "phyag 'tshal" → Output "ཕྱག་ཚལ" ❌ Missing འ
   ```

#### Phonetic Review

1. **Pronunciation Testing**:
   - Read phonetic output aloud
   - Compare with known Tibetan pronunciation
   - Check for clarity and accuracy

2. **Syllable Structure**:
   - Verify syllable boundaries
   - Check consonant cluster simplification
   - Validate vowel sounds

3. **Consistency Check**:
   - Same Wylie should produce same phonetics
   - Similar patterns should be handled consistently

#### Translation Review

1. **Terminology Verification**:
   - Check Buddhist technical terms
   - Verify deity names and attributes
   - Confirm ritual terminology

2. **Contextual Appropriateness**:
   - Ensure Vajrayana Buddhist context
   - Check for cultural sensitivity
   - Verify appropriate reverence level

3. **Completeness and Clarity**:
   - All meaning conveyed
   - Clear and understandable English
   - Appropriate for intended audience

### Step 3: Scoring and Feedback

#### Scoring System

For each category, provide scores on a 1-10 scale:

**Wylie Conversion (Weight: 30%)**
- 9-10: Excellent accuracy, minimal errors
- 7-8: Good accuracy, minor issues
- 5-6: Acceptable with notable errors
- 3-4: Poor accuracy, significant issues
- 1-2: Unacceptable, major problems

**Phonetic Rendering (Weight: 25%)**
- 9-10: Highly accurate pronunciation guide
- 7-8: Generally accurate, minor issues
- 5-6: Usable but some inaccuracies
- 3-4: Misleading pronunciation
- 1-2: Completely inaccurate

**Translation Quality (Weight: 45%)**
- 9-10: Excellent Buddhist scholarship, accurate and appropriate
- 7-8: Good translation with minor issues
- 5-6: Acceptable but lacks nuance
- 3-4: Inaccurate or inappropriate
- 1-2: Completely wrong or offensive

#### Feedback Format

```markdown
## Expert Review Report

**Reviewer**: [Name and Credentials]
**Date**: [Review Date]
**Sample Size**: [Number of lines reviewed]

### Overall Assessment
- **Wylie Conversion**: X/10
- **Phonetic Rendering**: X/10
- **Translation Quality**: X/10
- **Overall Score**: X/10

### Detailed Findings

#### Strengths
- [List positive aspects]

#### Areas for Improvement
- [List specific issues with examples]

#### Critical Issues
- [List any serious problems requiring immediate attention]

### Recommendations
1. [Specific improvement suggestions]
2. [Additional validation needed]
3. [Expert consultation recommendations]
```

## Validation Test Cases

### High-Priority Test Cases

#### Basic Vocabulary
```
lha → ལྷ → lha → "deity/god"
mo → མོ → mo → "female/woman"
dbyangs → དབྱངས → yang → "melody/voice"
can → ཅན → chen → "having/possessing"
```

#### Complex Clusters
```
sgrub thabs → སྒྲུབ་ཐབས → drub tab → "sadhana/practice method"
phyag 'tshal → ཕྱག་འཚལ → chak tsal → "prostrate/bow"
byang chub → བྱང་ཆུབ → chang chub → "enlightenment"
```

#### Sacred Terms
```
oM ma Ni pad+me hU~M → ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ → om mani padme hung → "Om Mani Padme Hung (mantra)"
sangs rgyas → སངས་རྒྱས → sang gye → "Buddha"
```

#### Ritual Instructions
```
mchod pa → མཆོད་པ → chö pa → "offering/worship"
bsnyen pa → བསྙེན་པ → nyen pa → "approach practice"
```

### Edge Cases to Test

1. **Ambiguous Wylie**: Cases where multiple interpretations possible
2. **Sanskrit Integration**: Mixed Tibetan-Sanskrit phrases
3. **Regional Variations**: Terms with different pronunciations
4. **Modern vs. Classical**: Contemporary vs. traditional usage

## Expert Qualifications

### Recommended Reviewer Qualifications

**Primary Reviewers** (at least one required):
- PhD in Tibetan Studies, Buddhist Studies, or Linguistics
- Native or near-native Tibetan language proficiency
- Specialization in Tibetan Buddhist texts
- Experience with Wylie transliteration systems

**Secondary Reviewers** (additional validation):
- Advanced practitioners with extensive Tibetan study
- Tibetan language teachers or translators
- Buddhist scholars with Tibetan text experience

**Cultural Consultants**:
- Native Tibetan speakers
- Traditional Buddhist teachers (lamas/geshes)
- Cultural preservation specialists

## Review Timeline and Process

### Phase 1: Initial Validation (Week 1-2)
- Automated testing completion
- Primary expert review of core functionality
- Initial feedback and critical issue identification

### Phase 2: Detailed Review (Week 3-4)
- Comprehensive expert evaluation
- Secondary reviewer validation
- Cultural consultant input

### Phase 3: Refinement (Week 5-6)
- Address identified issues
- Re-validation of corrections
- Final expert approval

### Phase 4: Documentation (Week 7)
- Compile final validation report
- Document approved accuracy levels
- Create ongoing validation procedures

## Ongoing Validation

### Regular Review Schedule
- **Monthly**: Spot checks on new content
- **Quarterly**: Comprehensive accuracy review
- **Annually**: Full expert validation cycle

### Community Feedback Integration
- User-reported accuracy issues
- Scholar feedback incorporation
- Continuous improvement process

### Version Control
- Track accuracy improvements over time
- Document validation history
- Maintain expert review records

## Conclusion

This expert review process ensures that the Tibetan Text Processor maintains high standards of linguistic and cultural accuracy. The combination of automated testing and expert human validation provides comprehensive quality assurance for this important tool in Tibetan Buddhist text processing.

Regular expert review is essential for maintaining accuracy and cultural appropriateness, especially given the sacred nature of many Tibetan Buddhist texts and the importance of precise transmission of Buddhist teachings.