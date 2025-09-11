# Tibetan Text Processor - Linguistic Validation Report

**Generated**: 1757557695.491391
**Total Tests**: 21
**Accuracy**: 47.6%
**Average Confidence**: 0.756

## Summary

- ✅ **Passed**: 10 tests
- ❌ **Failed**: 11 tests
- 📊 **Overall Accuracy**: 47.6%

## Results by Category

### Basic Consonants
- Total: 4 tests
- Passed: 4 (100.0%)
- Failed: 0

### Aspirated
- Total: 3 tests
- Passed: 3 (100.0%)
- Failed: 0

### Common Words
- Total: 4 tests
- Passed: 2 (50.0%)
- Failed: 2

### Consonant Clusters
- Total: 4 tests
- Passed: 0 (0.0%)
- Failed: 4

### Phrases
- Total: 3 tests
- Passed: 1 (33.3%)
- Failed: 2

### Sanskrit
- Total: 3 tests
- Passed: 0 (0.0%)
- Failed: 3

## Failed Tests Details

### phonetic_common_words_10
**Input**: `dbyangs`
**Expected**: `yang`
**Actual**: `dbyangs`
**Confidence**: 0.700
**Notes**: Category: common_words

### phonetic_common_words_11
**Input**: `can`
**Expected**: `chen`
**Actual**: `chan`
**Confidence**: 0.700
**Notes**: Category: common_words

### phonetic_consonant_clusters_12
**Input**: `phyag`
**Expected**: `chak`
**Actual**: `phya`
**Confidence**: 0.000
**Notes**: Category: consonant_clusters

### phonetic_consonant_clusters_13
**Input**: `'tshal`
**Expected**: `tsal`
**Actual**: `tshal'`
**Confidence**: 0.700
**Notes**: Category: consonant_clusters

### phonetic_consonant_clusters_14
**Input**: `brgya`
**Expected**: `gya`
**Actual**: `brgya`
**Confidence**: 0.700
**Notes**: Category: consonant_clusters

### phonetic_consonant_clusters_15
**Input**: `sgrub`
**Expected**: `drub`
**Actual**: `sgrub`
**Confidence**: 0.700
**Notes**: Category: consonant_clusters

### phonetic_phrases_17
**Input**: `dbyangs can ma`
**Expected**: `yang chen ma`
**Actual**: `dbyangs chan ma`
**Confidence**: 0.800
**Notes**: Category: phrases

### phonetic_phrases_18
**Input**: `phyag 'tshal lo`
**Expected**: `chak tsal lo`
**Actual**: `phyag tshal' lo`
**Confidence**: 0.567
**Notes**: Category: phrases

### phonetic_sanskrit_19
**Input**: `oM`
**Expected**: `om`
**Actual**: `OHM`
**Confidence**: 0.700
**Notes**: Category: sanskrit

### phonetic_sanskrit_20
**Input**: `hU~M`
**Expected**: `hung`
**Actual**: `HOOM`
**Confidence**: 0.000
**Notes**: Category: sanskrit

### phonetic_sanskrit_21
**Input**: `swA hA`
**Expected**: `soha`
**Actual**: `swaa haa`
**Confidence**: 0.300
**Notes**: Category: sanskrit

## Recommendations

- ⚠️ **Low accuracy detected**. Review conversion rules and test cases.
- ⚠️ **common_words** needs attention (accuracy: 50.0%)
- ⚠️ **consonant_clusters** needs attention (accuracy: 0.0%)
- ⚠️ **phrases** needs attention (accuracy: 33.3%)
- ⚠️ **sanskrit** needs attention (accuracy: 0.0%)