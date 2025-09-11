#!/usr/bin/env python3
"""
Optimized Tibetan text processor with performance improvements:
- API call batching and caching
- Memory usage monitoring
- Progress saving and resumption
- Parallel processing where safe
"""

import os
import argparse
import logging
import traceback
import time
import json
import psutil
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv
from .config import TibetanConfig as cfg
from .tibetan_line_parser import TibetanLineParser
from .wylie_converter import WylieConverter
from .phonetic_renderer import PhoneticRenderer
from .tibetan_translation_client import TibetanTranslationClient
from .tibetan_file_handler import TibetanFileHandler
from .tibetan_output_formatter import TibetanOutputFormatter
from .tibetan_models import TibetanLine, ProcessedLine, ProcessingStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tibetan_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedTibetanProcessor:
    """Optimized Tibetan processor with caching, batching, and monitoring."""
    
    def __init__(self, client: Anthropic, batch_size: int = None, enable_cache: bool = None):
        self.client = client
        self.batch_size = batch_size or cfg.BATCH_SIZE
        self.enable_cache = enable_cache if enable_cache is not None else cfg.ENABLE_CACHING
        
        # Initialize components
        self.line_parser = TibetanLineParser()
        self.wylie_converter = WylieConverter()
        self.phonetic_renderer = PhoneticRenderer()
        self.translation_client = TibetanTranslationClient(client)
        self.file_handler = TibetanFileHandler()
        self.output_formatter = TibetanOutputFormatter()
        
        # Caching system
        self.translation_cache = {}
        self.wylie_cache = {}
        self.phonetic_cache = {}
        self.cache_lock = threading.Lock()
        
        # Performance monitoring
        self.start_time = None
        self.memory_samples = []
        self.api_call_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load existing cache
        if self.enable_cache:
            self.load_cache()
    
    def load_cache(self):
        """Load existing cache from disk."""
        cache_file = cfg.CACHE_FILE
        if not os.path.exists(cache_file):
            logger.info("No cache file found, starting with empty cache")
            return
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.translation_cache = cache_data.get('translation_cache', {})
            self.wylie_cache = cache_data.get('wylie_cache', {})
            self.phonetic_cache = cache_data.get('phonetic_cache', {})
            
            total_cached = len(self.translation_cache) + len(self.wylie_cache) + len(self.phonetic_cache)
            logger.info(f"Loaded cache with {total_cached} entries")
            
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
    
    def save_cache(self):
        """Save cache to disk."""
        if not self.enable_cache:
            return
            
        cache_data = {
            'translation_cache': self.translation_cache,
            'wylie_cache': self.wylie_cache,
            'phonetic_cache': self.phonetic_cache,
            'metadata': {
                'created': time.time(),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses
            }
        }
        
        try:
            with open(cfg.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache saved to {cfg.CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def get_cached_translation(self, wylie_text: str, tibetan_script: str = "") -> str:
        """Get cached translation or make API call."""
        if not self.enable_cache:
            return self._make_translation_call(wylie_text, tibetan_script)
        
        cache_key = f"{wylie_text}|{tibetan_script}"
        
        with self.cache_lock:
            if cache_key in self.translation_cache:
                self.cache_hits += 1
                return self.translation_cache[cache_key]
        
        # Cache miss
        self.cache_misses += 1
        translation = self._make_translation_call(wylie_text, tibetan_script)
        
        with self.cache_lock:
            self.translation_cache[cache_key] = translation
        
        return translation
    
    def _make_translation_call(self, wylie_text: str, tibetan_script: str = "") -> str:
        """Make API call with timing and rate limiting."""
        start_time = time.time()
        
        try:
            # Add small delay to respect rate limits
            time.sleep(cfg.RATE_LIMIT_BUFFER)
            
            translation = self.translation_client.translate_vajrayana_context(wylie_text, tibetan_script)
            
            call_time = time.time() - start_time
            self.api_call_times.append(call_time)
            
            return translation
            
        except Exception as e:
            logger.warning(f"API call failed: {e}")
            return f"[Translation failed: {str(e)}]"
    
    def get_cached_wylie_conversion(self, wylie_text: str) -> str:
        """Get cached Wylie conversion."""
        if not self.enable_cache or wylie_text not in self.wylie_cache:
            result = self.wylie_converter.convert_to_tibetan(wylie_text)
            if self.enable_cache:
                self.wylie_cache[wylie_text] = result
            return result
        
        return self.wylie_cache[wylie_text]
    
    def get_cached_phonetic(self, wylie_text: str) -> str:
        """Get cached phonetic rendering."""
        if not self.enable_cache or wylie_text not in self.phonetic_cache:
            result = self.phonetic_renderer.generate_phonetic(wylie_text)
            if self.enable_cache:
                self.phonetic_cache[wylie_text] = result
            return result
        
        return self.phonetic_cache[wylie_text]
    
    def monitor_memory(self):
        """Sample current memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append({
                'time': time.time() - self.start_time,
                'memory_mb': memory_mb
            })
        except Exception as e:
            logger.debug(f"Memory monitoring failed: {e}")
    
    def process_batch_optimized(self, lines: List[TibetanLine]) -> List[ProcessedLine]:
        """Process batch with optimizations."""
        processed_lines = []
        translation_tasks = []
        
        # First pass: handle non-API operations
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
            
            # Fast operations (Wylie conversion and phonetics)
            try:
                tibetan_script = self.get_cached_wylie_conversion(line.wylie_content)
                phonetic = self.get_cached_phonetic(line.wylie_content)
            except Exception as e:
                logger.warning(f"Conversion failed for line {line.line_number}: {e}")
                tibetan_script = f"[Conversion failed: {line.wylie_content}]"
                phonetic = f"[Phonetic failed: {line.wylie_content}]"
            
            # Prepare for translation (slow operation)
            translation_tasks.append({
                'line': line,
                'tibetan_script': tibetan_script,
                'phonetic': phonetic,
                'index': len(processed_lines)
            })
            
            # Placeholder for now
            processed_lines.append(None)
        
        # Second pass: handle translations with controlled concurrency
        if translation_tasks:
            with ThreadPoolExecutor(max_workers=cfg.MAX_CONCURRENT_API_CALLS) as executor:
                future_to_task = {}
                
                for task in translation_tasks:
                    future = executor.submit(
                        self.get_cached_translation,
                        task['line'].wylie_content,
                        task['tibetan_script']
                    )
                    future_to_task[future] = task
                
                # Collect results
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        translation = future.result()
                        
                        # Determine status
                        status = ProcessingStatus.SUCCESS
                        error_msg = None
                        
                        if any(fail_marker in text for text in [task['tibetan_script'], task['phonetic'], translation]
                               for fail_marker in ["[Conversion failed:", "[Phonetic failed:", "[Translation failed:"]):
                            status = ProcessingStatus.PARTIAL_SUCCESS
                            error_msg = "Some processing steps failed"
                        
                        processed_line = ProcessedLine(
                            source_line=task['line'],
                            tibetan_script=task['tibetan_script'],
                            phonetic_rendering=task['phonetic'],
                            english_translation=translation,
                            processing_status=status,
                            error_message=error_msg
                        )
                        
                        processed_lines[task['index']] = processed_line
                        
                    except Exception as e:
                        logger.error(f"Translation failed for line {task['line'].line_number}: {e}")
                        
                        processed_line = ProcessedLine(
                            source_line=task['line'],
                            tibetan_script=task['tibetan_script'],
                            phonetic_rendering=task['phonetic'],
                            english_translation=f"[Translation failed: {str(e)}]",
                            processing_status=ProcessingStatus.FAILED,
                            error_message=str(e)
                        )
                        
                        processed_lines[task['index']] = processed_line
        
        # Filter out None placeholders
        return [line for line in processed_lines if line is not None]
    
    def save_progress(self, processed_lines: List[ProcessedLine], output_file: str):
        """Save processing progress."""
        if not cfg.ENABLE_PROGRESS_SAVING:
            return
        
        progress_file = f"{output_file}.progress"
        
        try:
            progress_data = {
                'processed_lines': [
                    {
                        'source_line': {
                            'line_number': pl.source_line.line_number,
                            'original_content': pl.source_line.original_content,
                            'wylie_content': pl.source_line.wylie_content,
                            'is_structural': pl.source_line.is_structural,
                            'markers': pl.source_line.markers
                        },
                        'tibetan_script': pl.tibetan_script,
                        'phonetic_rendering': pl.phonetic_rendering,
                        'english_translation': pl.english_translation,
                        'processing_status': pl.processing_status.value,
                        'error_message': pl.error_message
                    } for pl in processed_lines
                ],
                'timestamp': time.time(),
                'performance_stats': {
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'api_calls': len(self.api_call_times),
                    'avg_api_time': sum(self.api_call_times) / len(self.api_call_times) if self.api_call_times else 0
                }
            }
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def load_progress(self, output_file: str) -> List[ProcessedLine]:
        """Load previous progress if available."""
        progress_file = f"{output_file}.progress"
        
        if not cfg.ENABLE_PROGRESS_SAVING or not os.path.exists(progress_file):
            return []
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            processed_lines = []
            for item in progress_data.get('processed_lines', []):
                # Reconstruct TibetanLine
                source_data = item['source_line']
                source_line = TibetanLine(
                    line_number=source_data['line_number'],
                    original_content=source_data['original_content'],
                    wylie_content=source_data['wylie_content'],
                    is_structural=source_data['is_structural'],
                    markers=source_data['markers']
                )
                
                # Reconstruct ProcessedLine
                processed_line = ProcessedLine(
                    source_line=source_line,
                    tibetan_script=item['tibetan_script'],
                    phonetic_rendering=item['phonetic_rendering'],
                    english_translation=item['english_translation'],
                    processing_status=ProcessingStatus(item['processing_status']),
                    error_message=item['error_message']
                )
                
                processed_lines.append(processed_line)
            
            logger.info(f"Resumed from line {len(processed_lines) + 1}")
            return processed_lines
            
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")
            return []
    
    def process_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process file with all optimizations."""
        self.start_time = time.time()
        
        # Parse input
        logger.info("Parsing input file...")
        tibetan_lines = self.line_parser.parse_file(input_file)
        
        if not tibetan_lines:
            logger.warning("No lines found to process")
            return {'status': 'no_content'}
        
        logger.info(f"Found {len(tibetan_lines)} lines to process")
        
        # Load previous progress
        processed_lines = self.load_progress(output_file)
        start_index = len(processed_lines)
        
        if start_index > 0:
            logger.info(f"Resuming from line {start_index + 1}")
        
        # Process remaining lines in batches
        for i in range(start_index, len(tibetan_lines), self.batch_size):
            batch = tibetan_lines[i:i + self.batch_size]
            batch_num = ((i - start_index) // self.batch_size) + 1
            total_batches = ((len(tibetan_lines) - start_index) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} lines)")
            
            # Monitor memory periodically
            if i % cfg.MEMORY_MONITORING_INTERVAL == 0:
                self.monitor_memory()
            
            # Process batch
            batch_results = self.process_batch_optimized(batch)
            processed_lines.extend(batch_results)
            
            # Progress reporting
            progress = len(processed_lines) / len(tibetan_lines) * 100
            elapsed = time.time() - self.start_time
            rate = len(processed_lines) / elapsed if elapsed > 0 else 0
            eta = (len(tibetan_lines) - len(processed_lines)) / rate if rate > 0 else 0
            
            logger.info(f"Progress: {progress:.1f}% - Rate: {rate:.2f} lines/sec - ETA: {eta/60:.1f} min")
            
            # Auto-save progress
            if len(processed_lines) % cfg.AUTO_SAVE_INTERVAL == 0:
                self.save_progress(processed_lines, output_file)
                self.save_cache()
        
        # Final memory sample
        self.monitor_memory()
        
        # Generate output
        logger.info("Writing output file...")
        self.file_handler.write_tibetan_output(output_file, processed_lines)
        
        # Save final state
        self.save_progress(processed_lines, output_file)
        self.save_cache()
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        
        stats = {
            'status': 'completed',
            'total_lines': len(tibetan_lines),
            'processed_lines': len(processed_lines),
            'processing_time': total_time,
            'lines_per_second': len(processed_lines) / total_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'api_calls': len(self.api_call_times),
            'avg_api_time': sum(self.api_call_times) / len(self.api_call_times) if self.api_call_times else 0,
            'memory_peak_mb': max(sample['memory_mb'] for sample in self.memory_samples) if self.memory_samples else 0
        }
        
        # Clean up progress file on successful completion
        progress_file = f"{output_file}.progress"
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
            except Exception as e:
                logger.warning(f"Could not remove progress file: {e}")
        
        return stats

def main() -> int:
    parser = argparse.ArgumentParser(description='Optimized Tibetan text processor')
    parser.add_argument('--input', '-i', help='Input file path', default=cfg.INPUT_FILE)
    parser.add_argument('--output', '-o', help='Output file path', default=cfg.OUTPUT_FILE)
    parser.add_argument('--batch-size', '-b', type=int, help='Batch size', default=cfg.BATCH_SIZE)
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--performance-report', help='Generate performance report')
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return 1
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    try:
        client = Anthropic(api_key=api_key)
        
        processor = OptimizedTibetanProcessor(
            client=client,
            batch_size=args.batch_size,
            enable_cache=not args.no_cache
        )
        
        logger.info("Starting optimized Tibetan text processing...")
        stats = processor.process_file(args.input, args.output)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Lines processed: {stats['processed_lines']}/{stats['total_lines']}")
        logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
        logger.info(f"Throughput: {stats['lines_per_second']:.2f} lines/second")
        logger.info(f"Cache performance: {stats['cache_hits']} hits, {stats['cache_misses']} misses")
        logger.info(f"API calls: {stats['api_calls']} (avg: {stats['avg_api_time']:.2f}s)")
        logger.info(f"Peak memory: {stats['memory_peak_mb']:.1f} MB")
        
        # Generate performance report if requested
        if args.performance_report:
            with open(args.performance_report, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Performance report saved to {args.performance_report}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())