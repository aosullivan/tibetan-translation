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


@dataclass
class TibetanConfig(TranslationConfig):
    # Tibetan-specific input/output settings
    INPUT_FILE: str = "saras-tibetan.txt"
    OUTPUT_FILE: str = "tibetan_output.md"
    WYLIE_OUTPUT_FILE: str = "tibetan_wylie.txt"
    PHONETIC_OUTPUT_FILE: str = "tibetan_phonetic.txt"
    TRANSLATION_OUTPUT_FILE: str = "tibetan_translation.txt"
    
    # Tibetan conversion options
    WYLIE_DELIMITER: str = "//"
    PRESERVE_STRUCTURAL_MARKERS: bool = False
    UNICODE_ENCODING: str = "utf-8"
    
    # Buddhist translation context
    TRANSLATION_CONTEXT: str = "vajrayana_buddhist"
    PRESERVE_SANSKRIT_TERMS: bool = True
    HANDLE_MANTRAS_SPECIALLY: bool = True
    
    # Processing options
    SKIP_EMPTY_LINES: bool = True
    BATCH_SIZE: int = 10  # Process lines in batches for efficiency
    ENABLE_PROGRESS_SAVING: bool = True
    
    # Performance optimization settings
    ENABLE_CACHING: bool = True
    CACHE_FILE: str = "tibetan_processor_cache.json"
    MAX_CONCURRENT_API_CALLS: int = 3  # Limit concurrent API calls
    MEMORY_MONITORING_INTERVAL: int = 10  # Monitor memory every N lines
    AUTO_SAVE_INTERVAL: int = 50  # Auto-save progress every N lines
    
    # API optimization
    API_RETRY_DELAY: float = 1.0  # Base delay for API retries
    API_TIMEOUT: int = 30  # API call timeout in seconds
    RATE_LIMIT_BUFFER: float = 0.1  # Buffer time between API calls
