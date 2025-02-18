#!/usr/bin/env python3
import os
from pathlib import Path
from config import TranslationConfig as cfg

def clean_output_files():
    """Remove translation output files and logs."""
    project_root = Path(__file__).parent
    
    files_to_remove = [
        "progress.json",
        "translation.log",
        cfg.OUTPUT_FILE
    ]
    
    removed = 0
    for filename in files_to_remove:
        file_path = project_root / filename
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"Removed: {filename}")
                removed += 1
            except Exception as e:
                print(f"Error removing {filename}: {e}")
                
    # Also remove any .translated files
    for translated_file in project_root.glob("*.translated"):
        try:
            translated_file.unlink()
            print(f"Removed: {translated_file.name}")
            removed += 1
        except Exception as e:
            print(f"Error removing {translated_file.name}: {e}")

    print(f"\nCleanup complete. Removed {removed} files.")

if __name__ == "__main__":
    clean_output_files()
