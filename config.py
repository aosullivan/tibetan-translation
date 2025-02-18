from dataclasses import dataclass

@dataclass
class TranslationConfig:
    # Input/Output settings
    INPUT_FILE: str = "input.txt"
    OUTPUT_FILE: str = "translations.md"
    OUTPUT_FORMAT: str = "markdown"
    PROGRESS_FILE: str = "progress.json"

    # Translation settings
    CHUNK_SIZE: int = 1000
    SUMMARY_INTERVAL: int = 5
    MAX_RETRIES: int = 3
    INITIAL_DELAY: int = 1
    MODEL: str = "claude-3-5-sonnet-latest"
    MAX_TOKENS: int = 1000
