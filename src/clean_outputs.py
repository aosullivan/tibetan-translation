#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import TranslationConfig as cfg


def clean_output_files(extra_outputs: list[str] | None = None) -> None:
    """Remove translation output files and logs."""
    project_root = Path(__file__).parent.parent

    files_to_remove = [
        "progress.json",
        "translation.log",
        cfg.OUTPUT_FILE,
    ]
    if extra_outputs:
        files_to_remove.extend(extra_outputs)
    seen: set[str] = set()
    deduped: list[str] = []
    for f in files_to_remove:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    files_to_remove = deduped
    
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
    parser = argparse.ArgumentParser(
        description="Remove progress, logs, default output, and optional extra output paths."
    )
    parser.add_argument(
        "-o",
        "--output",
        action="append",
        default=[],
        dest="extra_outputs",
        help="Additional output file path (relative to repo root) to delete; repeat for multiple.",
    )
    args = parser.parse_args()
    clean_output_files(extra_outputs=args.extra_outputs)
