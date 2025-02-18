from dataclasses import dataclass

@dataclass
class TranslationConfig:
    # Input/Output settings
    INPUT_FILE: str = "input.txt"
    OUTPUT_FILE: str = "translations.md"  # Change from .txt to .md
    OUTPUT_FORMAT: str = "markdown"  # or "txt"
    PROGRESS_FILE: str = "progress.json"

    # Translation settings
    CHUNK_SIZE: int = 1000  # characters per chunk
    SUMMARY_INTERVAL: int = 5  # generate summary every N chunks
    MAX_RETRIES: int = 3  # max retries for file operations
    INITIAL_DELAY: int = 1
    MODEL_NAME: str = "claude-3-5-sonnet-latest"
    MAX_TOKENS: int = 1000
