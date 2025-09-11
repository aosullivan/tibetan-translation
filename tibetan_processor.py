import os
import argparse
import logging
import traceback
import time
import json
from typing import List, Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv
from src.config import TibetanConfig as cfg
from src.tibetan_line_parser import TibetanLineParser
from src.wylie_converter import WylieConverter
from src.tibetan_translation_client import TibetanTranslationClient
from src.tibetan_file_handler import TibetanFileHandler
from src.tibetan_output_formatter import TibetanOutputFormatter
from src.tibetan_models import TibetanLine, ProcessedLine, ProcessingStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tibetan_processing.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main() -> int:
    parser = argparse.ArgumentParser(description='Process Tibetan text: convert Wylie to Tibetan script, generate phonetics, and translate')
    parser.add_argument('--input', '-i', help='Input file path (Wylie format)', default=cfg.INPUT_FILE)
    parser.add_argument('--output', '-o', help='Output file path', default=cfg.OUTPUT_FILE)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return 1

    try:
        # Clean up previous output files
        output_file_path = args.output
        progress_file_path = f"{args.output}.progress"
        
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
            logger.info(f"Deleted existing output file: {output_file_path}")
        if os.path.exists(progress_file_path):
            os.remove(progress_file_path)
            logger.info(f"Deleted existing progress file: {progress_file_path}")
        
        logger.info(f"Starting Tibetan text processing: {args.input} -> {args.output}")
        
        # Validate input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        # Initialize API client
        try:
            client = Anthropic(api_key=api_key)
            logger.info("API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            return 1
        
        # Initialize processing components
        try:
            line_parser = TibetanLineParser()
            wylie_converter = WylieConverter()
            translation_client = TibetanTranslationClient(client)
            file_handler = TibetanFileHandler()
            output_formatter = TibetanOutputFormatter()
            logger.info("All processing components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processing components: {e}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return 1
        
        # Parse input file
        logger.info("Parsing input file...")
        try:
            tibetan_lines = line_parser.parse_file(args.input)
            logger.info(f"Found {len(tibetan_lines)} lines to process")
        except Exception as e:
            logger.error(f"Failed to parse input file: {e}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return 1
        
        if not tibetan_lines:
            logger.warning("No lines found to process in input file")
            return 0
        
        processed_lines = []
        processing_stats = {
            'total_lines': len(tibetan_lines),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'start_time': time.time(),
            'errors': []
        }
        
        # Load progress if resuming
        progress_file = f"{args.output}.progress"
        if cfg.ENABLE_PROGRESS_SAVING and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    saved_progress = json.load(f)
                    processed_lines = [ProcessedLine(**item) for item in saved_progress.get('processed_lines', [])]
                    processing_stats.update(saved_progress.get('stats', {}))
                    logger.info(f"Resuming from line {len(processed_lines) + 1}")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}. Starting fresh.")
        
        # Process lines
        for line_idx, line in enumerate(tibetan_lines[len(processed_lines):]):
            current_line_num = len(processed_lines) + line_idx + 1
            
            # Progress indicator
            progress_percent = (current_line_num / len(tibetan_lines)) * 100
            elapsed_time = time.time() - processing_stats['start_time']
            if current_line_num > 0:
                estimated_total_time = elapsed_time * len(tibetan_lines) / current_line_num
                remaining_time = estimated_total_time - elapsed_time
                logger.info(f"Progress: {progress_percent:.1f}% - Estimated time remaining: {remaining_time/60:.1f} minutes")
            
            try:
                # Skip structural markers if configured
                if line.is_structural and cfg.SKIP_EMPTY_LINES:
                    logger.debug(f"Skipping structural line {line.line_number}")
                    processing_stats['skipped'] += 1
                    continue
                
                # Convert Wylie to Tibetan script
                try:
                    tibetan_script = wylie_converter.convert_to_tibetan(line.wylie_content)
                except Exception as e:
                    logger.warning(f"Wylie conversion failed for line {line.line_number}: {e}")
                    tibetan_script = f"[Conversion failed: {line.wylie_content}]"
                
                # Get English translation
                try:
                    translation, _ = translation_client.translate_vajrayana_context(
                        line.wylie_content, tibetan_script
                    )
                except Exception as e:
                    logger.warning(f"Translation failed for line {line.line_number}: {e}")
                    translation = f"[Translation failed: {str(e)}]"
                
                # Determine processing status
                status = ProcessingStatus.SUCCESS
                error_msg = None
                if "[Conversion failed:" in tibetan_script or "[Translation failed:" in translation:
                    status = ProcessingStatus.PARTIAL_SUCCESS
                    error_msg = "Some processing steps failed"
                
                # Create processed line
                processed_line = ProcessedLine(
                    source_line=line,
                    tibetan_script=tibetan_script,
                    english_translation=translation,
                    processing_status=status,
                    error_message=error_msg
                )
                
                processed_lines.append(processed_line)
                processing_stats['processed'] += 1
                
            except Exception as e:
                error_msg = f"Complete failure processing line {line.line_number}: {str(e)}"
                logger.error(error_msg)
                logger.debug(f"Stack trace for line {line.line_number}:\n{traceback.format_exc()}")
                
                # Create failed processed line for record keeping
                processed_line = ProcessedLine(
                    source_line=line,
                    tibetan_script="[Processing failed]",
                    english_translation="[Processing failed]",
                    processing_status=ProcessingStatus.FAILED,
                    error_message=str(e)
                )
                processed_lines.append(processed_line)
                processing_stats['failed'] += 1
                processing_stats['errors'].append({
                    'line_number': line.line_number,
                    'error': str(e),
                    'content': line.wylie_content[:100]  # First 100 chars for context
                })
            
            # Save progress after each line
            if cfg.ENABLE_PROGRESS_SAVING:
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
                                'english_translation': pl.english_translation,
                                'processing_status': pl.processing_status.value,
                                'error_message': pl.error_message
                            } for pl in processed_lines
                        ],
                        'stats': processing_stats
                    }
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Could not save progress: {e}")
        
        # Calculate final statistics
        processing_stats['end_time'] = time.time()
        processing_stats['total_time'] = processing_stats['end_time'] - processing_stats['start_time']
        
        # Generate output
        logger.info("Generating output file...")
        try:
            file_handler.write_tibetan_output(args.output, processed_lines)
            logger.info(f"Output written to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            return 1
        
        # Generate and display comprehensive summary report
        
        # Additional detailed statistics
        success_count = sum(1 for pl in processed_lines if pl.processing_status == ProcessingStatus.SUCCESS)
        partial_count = sum(1 for pl in processed_lines if pl.processing_status == ProcessingStatus.PARTIAL_SUCCESS)
        failed_count = sum(1 for pl in processed_lines if pl.processing_status == ProcessingStatus.FAILED)
        skipped_count = sum(1 for pl in processed_lines if pl.processing_status == ProcessingStatus.SKIPPED)
        
        logger.info("=" * 60)
        logger.info("TIBETAN PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total lines processed: {len(processed_lines)}")
        logger.info(f"✓ Successful: {success_count}")
        logger.info(f"⚠ Partial success: {partial_count}")
        logger.info(f"✗ Failed: {failed_count}")
        logger.info(f"⊘ Skipped: {skipped_count}")
        logger.info(f"Processing time: {processing_stats['total_time']/60:.2f} minutes")
        logger.info(f"Average time per line: {processing_stats['total_time']/len(processed_lines):.2f} seconds")
        
        if processing_stats['errors']:
            logger.info(f"\nErrors encountered: {len(processing_stats['errors'])}")
            for error in processing_stats['errors'][:5]:  # Show first 5 errors
                logger.info(f"  Line {error['line_number']}: {error['error']}")
            if len(processing_stats['errors']) > 5:
                logger.info(f"  ... and {len(processing_stats['errors']) - 5} more errors")
        
        # Clean up progress file on successful completion
        if cfg.ENABLE_PROGRESS_SAVING and os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                logger.info("Progress file cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove progress file: {e}")
        
        # Return appropriate exit code
        if failed_count > 0:
            logger.warning(f"Processing completed with {failed_count} failures")
            return 2  # Partial failure
        elif partial_count > 0:
            logger.info(f"Processing completed with {partial_count} partial successes")
            return 0  # Success with warnings
        else:
            logger.info("All lines processed successfully!")
            return 0
        
    except KeyboardInterrupt:
        logger.info("Tibetan processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error during Tibetan processing: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(code=main())