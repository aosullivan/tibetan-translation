from dataclasses import dataclass

@dataclass
class TranslationConfig:
    # Input/Output settings
    INPUT_FILE: str = "input.txt"
    OUTPUT_FILE: str = "translation.md"
    OUTPUT_FORMAT: str = "markdown"
    PROGRESS_FILE: str = "progress.json"

    # Translation settings
    CHUNK_SIZE: int = 1500
    SUMMARY_INTERVAL: int = 5
    MAX_RETRIES: int = 3
    INITIAL_DELAY: int = 1
    MODEL: str = "claude-sonnet-4-20250514"
    MAX_TOKENS: int = 1000
