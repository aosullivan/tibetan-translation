#!/usr/bin/env python3
"""
Integration test for TibetanLineParser and TibetanFileHandler.
"""

import tempfile
import os
import sys
sys.path.append('../src')
from src.tibetan_line_parser import TibetanLineParser
from src.tibetan_file_handler import TibetanFileHandler, ProcessedLine

def main():
    # Create sample Tibetan content
    sample_content = """(kha)@#/[ ]/bo dong paN chen phyogs las rnam rgyal/dbyangs can ma sgrub par 'dod pas/_tshangs par spyad_/khyad par rdzun mi smra/"""
    
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt') as f:
        f.write(sample_content)
        temp_input = f.name
    
    try:
        print("Testing TibetanLineParser...")
        parser = TibetanLineParser()
        lines = parser.parse_file(temp_input)
        
        print(f"Parsed {len(lines)} lines:")
        for i, line in enumerate(lines):
            print(f"  {i+1}: {'[S]' if line.is_structural else '[T]'} '{line.original_content}'")
            if line.wylie_content and line.wylie_content != line.original_content:
                print(f"      Wylie: '{line.wylie_content}'")
        
        print("\nTesting TibetanFileHandler...")
        handler = TibetanFileHandler()
        
        # Read using the handler
        handler_lines = handler.read_wylie_file(temp_input)
        print(f"Handler read {len(handler_lines)} lines")
        
        # Create processed lines for testing output
        processed_lines = []
        for line in handler_lines[:3]:  # Just test first 3 lines
            processed_line = ProcessedLine(line)
            
            if not line.is_structural:
                # Simulate processing results
                processed_line.tibetan_script = f"[Tibetan: {line.wylie_content[:20]}...]"
                processed_line.phonetic_rendering = f"[Phonetic: {line.wylie_content[:20]}...]"
                processed_line.english_translation = f"[Translation: {line.wylie_content[:20]}...]"
                processed_line.processing_status = "completed"
            else:
                processed_line.processing_status = "skipped"
            
            processed_lines.append(processed_line)
        
        # Test output writing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.md') as f:
            temp_output = f.name
        
        handler.write_tibetan_output(temp_output, processed_lines)
        
        print(f"\nOutput written to {temp_output}")
        print("Output preview:")
        with open(temp_output, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content[:400] + "..." if len(content) > 400 else content)
        
        print("\n✅ Integration test completed successfully!")
        
    finally:
        # Clean up
        if os.path.exists(temp_input):
            os.unlink(temp_input)
        if 'temp_output' in locals() and os.path.exists(temp_output):
            os.unlink(temp_output)

if __name__ == "__main__":
    main()