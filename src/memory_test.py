#!/usr/bin/env python3
"""
Memory usage and performance testing without API calls.
Tests local processing components only.
"""

import os
import time
import psutil
import logging
from typing import List
from .tibetan_line_parser import TibetanLineParser
from .wylie_converter import WylieConverter
from .phonetic_renderer import PhoneticRenderer
from .tibetan_models import TibetanLine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_local_processing_performance(input_file: str, batch_sizes: List[int] = None):
    """Test performance of local processing components."""
    if batch_sizes is None:
        batch_sizes = [1, 5, 10, 20, 50]
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Initialize components
    line_parser = TibetanLineParser()
    wylie_converter = WylieConverter()
    phonetic_renderer = PhoneticRenderer()
    
    # Parse input file once
    logger.info("Parsing input file...")
    start_memory = measure_memory_usage()
    tibetan_lines = line_parser.parse_file(input_file)
    parse_memory = measure_memory_usage()
    
    logger.info(f"Parsed {len(tibetan_lines)} lines")
    logger.info(f"Memory usage: {start_memory:.1f} MB -> {parse_memory:.1f} MB (+{parse_memory - start_memory:.1f} MB)")
    
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\nTesting batch size: {batch_size}")
        
        start_time = time.time()
        memory_samples = []
        processed_count = 0
        
        # Process in batches
        for i in range(0, len(tibetan_lines), batch_size):
            batch = tibetan_lines[i:i + batch_size]
            
            # Sample memory before batch
            memory_before = measure_memory_usage()
            
            # Process batch (local operations only)
            for line in batch:
                if not line.is_structural:
                    try:
                        # Wylie conversion
                        tibetan_script = wylie_converter.convert_to_tibetan(line.wylie_content)
                        
                        # Phonetic rendering
                        phonetic = phonetic_renderer.generate_phonetic(line.wylie_content)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.debug(f"Processing failed for line {line.line_number}: {e}")
            
            # Sample memory after batch
            memory_after = measure_memory_usage()
            memory_samples.append({
                'batch': i // batch_size + 1,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before
            })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate statistics
        peak_memory = max(sample['memory_after'] for sample in memory_samples)
        avg_memory_delta = sum(sample['memory_delta'] for sample in memory_samples) / len(memory_samples)
        lines_per_second = processed_count / processing_time
        
        results[batch_size] = {
            'processing_time': processing_time,
            'processed_count': processed_count,
            'lines_per_second': lines_per_second,
            'peak_memory_mb': peak_memory,
            'avg_memory_delta_mb': avg_memory_delta,
            'memory_samples': memory_samples
        }
        
        logger.info(f"Batch size {batch_size} results:")
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        logger.info(f"  Lines processed: {processed_count}")
        logger.info(f"  Lines per second: {lines_per_second:.2f}")
        logger.info(f"  Peak memory: {peak_memory:.1f} MB")
        logger.info(f"  Avg memory delta per batch: {avg_memory_delta:.3f} MB")
    
    # Find optimal batch size
    best_batch_size = max(results.keys(), key=lambda k: results[k]['lines_per_second'])
    logger.info(f"\nOptimal batch size: {best_batch_size} ({results[best_batch_size]['lines_per_second']:.2f} lines/sec)")
    
    return results

def test_caching_effectiveness():
    """Test the effectiveness of caching for repeated operations."""
    logger.info("\nTesting caching effectiveness...")
    
    # Sample Wylie texts (some duplicates to test caching)
    test_texts = [
        "bcom ldan 'das",
        "sangs rgyas",
        "byang chub sems dpa'",
        "bcom ldan 'das",  # duplicate
        "chos kyi dbyings",
        "sangs rgyas",     # duplicate
        "ye shes",
        "byang chub sems dpa'",  # duplicate
    ]
    
    wylie_converter = WylieConverter()
    phonetic_renderer = PhoneticRenderer()
    
    # Test without caching
    start_time = time.time()
    for text in test_texts:
        wylie_converter.convert_to_tibetan(text)
        phonetic_renderer.generate_phonetic(text)
    no_cache_time = time.time() - start_time
    
    # Test with simple caching
    wylie_cache = {}
    phonetic_cache = {}
    
    start_time = time.time()
    for text in test_texts:
        if text not in wylie_cache:
            wylie_cache[text] = wylie_converter.convert_to_tibetan(text)
        if text not in phonetic_cache:
            phonetic_cache[text] = phonetic_renderer.generate_phonetic(text)
    cache_time = time.time() - start_time
    
    cache_hits = len(test_texts) - len(set(test_texts))
    speedup = no_cache_time / cache_time if cache_time > 0 else float('inf')
    
    logger.info(f"Without caching: {no_cache_time:.4f} seconds")
    logger.info(f"With caching: {cache_time:.4f} seconds")
    logger.info(f"Cache hits: {cache_hits}/{len(test_texts)}")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return {
        'no_cache_time': no_cache_time,
        'cache_time': cache_time,
        'cache_hits': cache_hits,
        'speedup': speedup
    }

def main():
    input_file = "test_tibetan_sample.txt"
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    logger.info("Starting local performance testing...")
    
    # Test processing performance
    processing_results = test_local_processing_performance(input_file)
    
    # Test caching effectiveness
    caching_results = test_caching_effectiveness()
    
    # Generate simple report
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE TEST SUMMARY")
    logger.info("="*60)
    
    if processing_results:
        logger.info("\nProcessing Performance:")
        for batch_size, results in processing_results.items():
            logger.info(f"  Batch {batch_size:2d}: {results['lines_per_second']:6.2f} lines/sec, "
                       f"{results['peak_memory_mb']:6.1f} MB peak")
    
    logger.info(f"\nCaching Effectiveness:")
    logger.info(f"  Speedup: {caching_results['speedup']:.2f}x")
    logger.info(f"  Cache hits: {caching_results['cache_hits']}/{len(['test'] * 8)}")
    
    logger.info("\nLocal performance testing completed!")
    return 0

if __name__ == "__main__":
    exit(main())