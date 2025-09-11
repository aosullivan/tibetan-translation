#!/usr/bin/env python3
"""
Performance testing and optimization for Tibetan text processor.
Tests processing speed, memory usage, and implements optimizations.
"""

import os
import time
import psutil
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# Import our modules
from .tibetan_line_parser import TibetanLineParser
from .wylie_converter import WylieConverter
from .phonetic_renderer import PhoneticRenderer
from .tibetan_translation_client import TibetanTranslationClient
from .tibetan_file_handler import TibetanFileHandler
from .tibetan_output_formatter import TibetanOutputFormatter
from .tibetan_models import TibetanLine, ProcessedLine, ProcessingStatus
from .config import TibetanConfig as cfg
from anthropic import Anthropic
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for processing operations."""
    total_lines: int = 0
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_start_mb: float = 0.0
    api_calls_made: int = 0
    api_call_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    lines_per_second: float = 0.0
    api_calls_per_minute: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class OptimizedTibetanProcessor:
    """Optimized version of Tibetan processor with caching and batching."""
    
    def __init__(self, client: Anthropic, batch_size: int = 10, enable_cache: bool = True):
        self.client = client
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        
        # Initialize components
        self.line_parser = TibetanLineParser()
        self.wylie_converter = WylieConverter()
        self.phonetic_renderer = PhoneticRenderer()
        self.translation_client = TibetanTranslationClient(client)
        self.file_handler = TibetanFileHandler()
        self.output_formatter = TibetanOutputFormatter()
        
        # Caching
        self.translation_cache = {}
        self.wylie_cache = {}
        self.phonetic_cache = {}
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.api_call_times = []
        self.memory_samples = []
        
        # Thread safety
        self.cache_lock = threading.Lock()
        
    def get_cached_translation(self, wylie_text: str, tibetan_script: str = "") -> str:
        """Get cached translation or make API call."""
        if not self.enable_cache:
            return self._make_translation_call(wylie_text, tibetan_script)
            
        cache_key = f"{wylie_text}|{tibetan_script}"
        
        with self.cache_lock:
            if cache_key in self.translation_cache:
                self.metrics.cache_hits += 1
                return self.translation_cache[cache_key]
        
        # Cache miss - make API call
        self.metrics.cache_misses += 1
        translation = self._make_translation_call(wylie_text, tibetan_script)
        
        with self.cache_lock:
            self.translation_cache[cache_key] = translation
            
        return translation
    
    def _make_translation_call(self, wylie_text: str, tibetan_script: str = "") -> str:
        """Make actual API call and track timing."""
        start_time = time.time()
        try:
            translation = self.translation_client.translate_vajrayana_context(wylie_text, tibetan_script)
            self.metrics.api_calls_made += 1
            call_time = time.time() - start_time
            self.api_call_times.append(call_time)
            self.metrics.api_call_time += call_time
            return translation
        except Exception as e:
            self.metrics.errors.append(f"API call failed: {str(e)}")
            return f"[Translation failed: {str(e)}]"
    
    def get_cached_wylie_conversion(self, wylie_text: str) -> str:
        """Get cached Wylie conversion."""
        if not self.enable_cache:
            return self.wylie_converter.convert_to_tibetan(wylie_text)
            
        if wylie_text in self.wylie_cache:
            return self.wylie_cache[wylie_text]
            
        result = self.wylie_converter.convert_to_tibetan(wylie_text)
        self.wylie_cache[wylie_text] = result
        return result
    
    def get_cached_phonetic(self, wylie_text: str) -> str:
        """Get cached phonetic rendering."""
        if not self.enable_cache:
            return self.phonetic_renderer.generate_phonetic(wylie_text)
            
        if wylie_text in self.phonetic_cache:
            return self.phonetic_cache[wylie_text]
            
        result = self.phonetic_renderer.generate_phonetic(wylie_text)
        self.phonetic_cache[wylie_text] = result
        return result
    
    def process_batch_parallel(self, lines: List[TibetanLine]) -> List[ProcessedLine]:
        """Process a batch of lines with parallel API calls."""
        processed_lines = []
        
        # First, do all non-API processing
        for line in lines:
            if line.is_structural and cfg.SKIP_EMPTY_LINES:
                processed_line = ProcessedLine(
                    source_line=line,
                    tibetan_script="",
                    phonetic_rendering="",
                    english_translation="[Structural marker - skipped]",
                    processing_status=ProcessingStatus.SKIPPED
                )
                processed_lines.append(processed_line)
                continue
                
            # Convert Wylie and generate phonetics (fast operations)
            try:
                tibetan_script = self.get_cached_wylie_conversion(line.wylie_content)
                phonetic = self.get_cached_phonetic(line.wylie_content)
            except Exception as e:
                self.metrics.errors.append(f"Conversion failed for line {line.line_number}: {str(e)}")
                tibetan_script = f"[Conversion failed: {line.wylie_content}]"
                phonetic = f"[Phonetic failed: {line.wylie_content}]"
            
            processed_lines.append({
                'line': line,
                'tibetan_script': tibetan_script,
                'phonetic': phonetic,
                'index': len(processed_lines)
            })
        
        # Now do translations in parallel (slow operations)
        translation_futures = {}
        with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent API calls
            for item in processed_lines:
                if isinstance(item, dict):  # Not a skipped line
                    future = executor.submit(
                        self.get_cached_translation,
                        item['line'].wylie_content,
                        item['tibetan_script']
                    )
                    translation_futures[future] = item
            
            # Collect results
            for future in as_completed(translation_futures):
                item = translation_futures[future]
                try:
                    translation = future.result()
                    
                    # Determine processing status
                    status = ProcessingStatus.SUCCESS
                    error_msg = None
                    if any(fail_marker in text for text in [item['tibetan_script'], item['phonetic'], translation] 
                           for fail_marker in ["[Conversion failed:", "[Phonetic failed:", "[Translation failed:"]):
                        status = ProcessingStatus.PARTIAL_SUCCESS
                        error_msg = "Some processing steps failed"
                    
                    processed_line = ProcessedLine(
                        source_line=item['line'],
                        tibetan_script=item['tibetan_script'],
                        phonetic_rendering=item['phonetic'],
                        english_translation=translation,
                        processing_status=status,
                        error_message=error_msg
                    )
                    
                    # Replace the dict with the ProcessedLine at the correct index
                    processed_lines[item['index']] = processed_line
                    
                except Exception as e:
                    self.metrics.errors.append(f"Translation failed for line {item['line'].line_number}: {str(e)}")
                    processed_line = ProcessedLine(
                        source_line=item['line'],
                        tibetan_script=item['tibetan_script'],
                        phonetic_rendering=item['phonetic'],
                        english_translation=f"[Translation failed: {str(e)}]",
                        processing_status=ProcessingStatus.FAILED,
                        error_message=str(e)
                    )
                    processed_lines[item['index']] = processed_line
        
        return [line for line in processed_lines if isinstance(line, ProcessedLine)]
    
    def sample_memory_usage(self):
        """Sample current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
        if memory_mb > self.metrics.memory_peak_mb:
            self.metrics.memory_peak_mb = memory_mb
    
    def process_file_optimized(self, input_file: str, output_file: str) -> PerformanceMetrics:
        """Process file with optimizations and performance tracking."""
        start_time = time.time()
        
        # Initial memory measurement
        process = psutil.Process()
        self.metrics.memory_start_mb = process.memory_info().rss / 1024 / 1024
        
        # Parse input file
        logger.info("Parsing input file...")
        tibetan_lines = self.line_parser.parse_file(input_file)
        self.metrics.total_lines = len(tibetan_lines)
        
        if not tibetan_lines:
            logger.warning("No lines found to process")
            return self.metrics
        
        logger.info(f"Processing {len(tibetan_lines)} lines with batch size {self.batch_size}")
        
        processed_lines = []
        
        # Process in batches
        for i in range(0, len(tibetan_lines), self.batch_size):
            batch = tibetan_lines[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(tibetan_lines) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Sample memory usage
            self.sample_memory_usage()
            
            # Process batch with parallel API calls
            batch_results = self.process_batch_parallel(batch)
            processed_lines.extend(batch_results)
            
            # Progress reporting
            progress = len(processed_lines) / len(tibetan_lines) * 100
            elapsed = time.time() - start_time
            if len(processed_lines) > 0:
                rate = len(processed_lines) / elapsed
                eta = (len(tibetan_lines) - len(processed_lines)) / rate if rate > 0 else 0
                logger.info(f"Progress: {progress:.1f}% - Rate: {rate:.2f} lines/sec - ETA: {eta/60:.1f} min")
        
        # Final memory sample
        self.sample_memory_usage()
        
        # Calculate final metrics
        self.metrics.processing_time = time.time() - start_time
        self.metrics.lines_per_second = len(processed_lines) / self.metrics.processing_time
        if self.metrics.api_call_time > 0:
            self.metrics.api_calls_per_minute = (self.metrics.api_calls_made / self.metrics.api_call_time) * 60
        
        # Write output
        logger.info("Writing output file...")
        self.file_handler.write_tibetan_output(output_file, processed_lines)
        
        return self.metrics
    
    def save_cache_to_disk(self, cache_file: str):
        """Save caches to disk for reuse."""
        cache_data = {
            'translation_cache': self.translation_cache,
            'wylie_cache': self.wylie_cache,
            'phonetic_cache': self.phonetic_cache
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cache saved to {cache_file}")
    
    def load_cache_from_disk(self, cache_file: str):
        """Load caches from disk."""
        if not os.path.exists(cache_file):
            logger.info("No cache file found, starting with empty cache")
            return
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.translation_cache = cache_data.get('translation_cache', {})
            self.wylie_cache = cache_data.get('wylie_cache', {})
            self.phonetic_cache = cache_data.get('phonetic_cache', {})
            
            logger.info(f"Cache loaded: {len(self.translation_cache)} translations, "
                       f"{len(self.wylie_cache)} wylie conversions, "
                       f"{len(self.phonetic_cache)} phonetic renderings")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")

def run_performance_test(input_file: str, batch_sizes: List[int] = None, 
                        enable_cache: bool = True) -> Dict[str, PerformanceMetrics]:
    """Run performance tests with different configurations."""
    if batch_sizes is None:
        batch_sizes = [1, 5, 10, 20]
    
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = Anthropic(api_key=api_key)
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing batch size: {batch_size}")
        logger.info(f"{'='*60}")
        
        processor = OptimizedTibetanProcessor(
            client=client,
            batch_size=batch_size,
            enable_cache=enable_cache
        )
        
        # Load cache if available
        cache_file = f"performance_cache_batch_{batch_size}.json"
        processor.load_cache_from_disk(cache_file)
        
        output_file = f"performance_test_output_batch_{batch_size}.md"
        
        try:
            metrics = processor.process_file_optimized(input_file, output_file)
            results[f"batch_{batch_size}"] = metrics
            
            # Save cache for reuse
            processor.save_cache_to_disk(cache_file)
            
            # Print results
            print_performance_results(metrics, batch_size)
            
        except Exception as e:
            logger.error(f"Performance test failed for batch size {batch_size}: {e}")
            results[f"batch_{batch_size}"] = None
    
    return results

def print_performance_results(metrics: PerformanceMetrics, batch_size: int):
    """Print formatted performance results."""
    print(f"\nPerformance Results (Batch Size: {batch_size})")
    print("-" * 50)
    print(f"Total lines processed: {metrics.total_lines}")
    print(f"Processing time: {metrics.processing_time:.2f} seconds")
    print(f"Lines per second: {metrics.lines_per_second:.2f}")
    print(f"Memory usage: {metrics.memory_start_mb:.1f} MB -> {metrics.memory_peak_mb:.1f} MB")
    print(f"Memory increase: {metrics.memory_peak_mb - metrics.memory_start_mb:.1f} MB")
    print(f"API calls made: {metrics.api_calls_made}")
    print(f"API call time: {metrics.api_call_time:.2f} seconds")
    print(f"API calls per minute: {metrics.api_calls_per_minute:.1f}")
    print(f"Cache hits: {metrics.cache_hits}")
    print(f"Cache misses: {metrics.cache_misses}")
    if metrics.cache_hits + metrics.cache_misses > 0:
        cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) * 100
        print(f"Cache hit rate: {cache_hit_rate:.1f}%")
    print(f"Errors: {len(metrics.errors)}")
    if metrics.errors:
        print("First few errors:")
        for error in metrics.errors[:3]:
            print(f"  - {error}")

def generate_performance_report(results: Dict[str, PerformanceMetrics], output_file: str):
    """Generate comprehensive performance report."""
    report_lines = [
        "# Tibetan Text Processor Performance Report",
        "",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Batch Size | Lines/sec | Memory Peak (MB) | API Calls/min | Cache Hit Rate | Errors |",
        "|------------|-----------|------------------|---------------|----------------|--------|"
    ]
    
    for config_name, metrics in results.items():
        if metrics is None:
            continue
            
        batch_size = config_name.split('_')[1]
        cache_hit_rate = 0
        if metrics.cache_hits + metrics.cache_misses > 0:
            cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) * 100
        
        report_lines.append(
            f"| {batch_size} | {metrics.lines_per_second:.2f} | {metrics.memory_peak_mb:.1f} | "
            f"{metrics.api_calls_per_minute:.1f} | {cache_hit_rate:.1f}% | {len(metrics.errors)} |"
        )
    
    report_lines.extend([
        "",
        "## Detailed Results",
        ""
    ])
    
    for config_name, metrics in results.items():
        if metrics is None:
            continue
            
        batch_size = config_name.split('_')[1]
        report_lines.extend([
            f"### Batch Size: {batch_size}",
            "",
            f"- **Total lines processed:** {metrics.total_lines}",
            f"- **Processing time:** {metrics.processing_time:.2f} seconds",
            f"- **Throughput:** {metrics.lines_per_second:.2f} lines/second",
            f"- **Memory usage:** {metrics.memory_start_mb:.1f} MB → {metrics.memory_peak_mb:.1f} MB",
            f"- **API calls:** {metrics.api_calls_made} calls in {metrics.api_call_time:.2f} seconds",
            f"- **API rate:** {metrics.api_calls_per_minute:.1f} calls/minute",
            f"- **Cache performance:** {metrics.cache_hits} hits, {metrics.cache_misses} misses",
            f"- **Errors:** {len(metrics.errors)} errors encountered",
            ""
        ])
        
        if metrics.errors:
            report_lines.extend([
                "**Error details:**",
                ""
            ])
            for i, error in enumerate(metrics.errors[:10]):  # Show first 10 errors
                report_lines.append(f"{i+1}. {error}")
            if len(metrics.errors) > 10:
                report_lines.append(f"... and {len(metrics.errors) - 10} more errors")
            report_lines.append("")
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Performance report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Performance testing for Tibetan text processor')
    parser.add_argument('--input', '-i', default='saras-tibetan.txt', help='Input file to test with')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 5, 10, 20], 
                       help='Batch sizes to test')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--report', '-r', default='performance_report.md', 
                       help='Output file for performance report')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    try:
        logger.info("Starting performance testing...")
        results = run_performance_test(
            input_file=args.input,
            batch_sizes=args.batch_sizes,
            enable_cache=not args.no_cache
        )
        
        # Generate comprehensive report
        generate_performance_report(results, args.report)
        
        # Find optimal configuration
        best_config = None
        best_throughput = 0
        
        for config_name, metrics in results.items():
            if metrics and metrics.lines_per_second > best_throughput:
                best_throughput = metrics.lines_per_second
                best_config = config_name
        
        if best_config:
            batch_size = best_config.split('_')[1]
            logger.info(f"\nOptimal configuration: Batch size {batch_size} "
                       f"({best_throughput:.2f} lines/second)")
        
        logger.info("Performance testing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())