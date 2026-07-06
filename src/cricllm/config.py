"""All the knobs for this pipeline, pulled from env vars / a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]


class Settings(BaseSettings):
    """Every setting you can tweak, loaded from the environment or .env."""

    model_config = SettingsConfigDict(env_prefix="CRICLLM_", env_file=".env", extra="ignore")

    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")

    embedding_model: str = "models/gemini-embedding-001"
    embedding_task_type: str = "RETRIEVAL_DOCUMENT"
    # gemini-embedding-001's output size — has to match the Pinecone index's
    # dimension exactly, both are set from this one place.
    embedding_dimension: int = 3072

    # Only used at query time (ask.py) — this is what actually writes the
    # answer once we've already fetched the relevant chunks.
    generation_model: str = "gemini-3.5-flash"
    retrieval_top_k: int = 5

    min_chunk_tokens: int = 400
    max_chunk_tokens: int = 700
    hard_max_chunk_tokens: int = 1400
    # Gemini just flat-out rejects embedContent calls over ~2048 tokens —
    # retrying won't save you. Anything bigger gets force-split as a last
    # resort instead of guaranteed-failing forever.
    max_embedding_input_tokens: int = 2048

    embed_batch_size: int = 32
    max_retries: int = 6
    retry_min_seconds: float = 1.0
    retry_max_seconds: float = 60.0

    use_semantic_chunking: bool = True
    semantic_breakpoint_threshold_type: BreakpointThresholdType = "percentile"

    cache_db: Path = Path(".cache/cricllm_cache.sqlite3")
    log_dir: Path = Path("logs")
    dead_letter_dir: Path = Path("logs/dead_letter")

    # Pinecone index names are restricted to lowercase alphanumerics + hyphens.
    pinecone_index_name: str = "cricllm-icc-laws"

    def ensure_directories(self) -> None:
        """Make sure the output dirs exist before anything tries to write to them."""
        for directory in (self.cache_db.parent, self.log_dir, self.dead_letter_dir):
            directory.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Read everything from the environment / .env and build a Settings object."""
    return Settings()  # type: ignore[call-arg]
