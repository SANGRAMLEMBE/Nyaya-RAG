"""Project-wide settings. Override anything via env vars (NYAYA_*) or .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NYAYA_", env_file=".env", extra="ignore")

    # --- paths ---
    data_dir: Path = REPO_ROOT / "data"
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    interim_dir: Path = REPO_ROOT / "data" / "interim"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    catalog_path: Path = REPO_ROOT / "configs" / "acts_catalog.yaml"
    manifest_db: Path = REPO_ROOT / "data" / "manifest.sqlite"
    qdrant_path: Path = REPO_ROOT / "data" / "qdrant"

    # --- scraper politeness (government sites: be a good citizen) ---
    request_delay_s: float = 3.0
    request_timeout_s: float = 60.0
    max_retries: int = 5
    user_agent: str = (
        "NyayaRAG-research/0.1 (M.Tech academic project; respects robots.txt; "
        "contact: FILL_YOUR_EMAIL)"
    )

    # --- models (all local; nothing leaves the machine) ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    generation_model: str = "Qwen/Qwen2.5-14B-Instruct"
    judge_model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"  # eval-only, served locally
    # vLLM serves the generation model on localhost. This is a local process,
    # not an external API: the OpenAI *client library* is used purely as the
    # wire protocol to talk to our own GPU.
    llm_base_url: str = "http://127.0.0.1:8000/v1"

    # --- retrieval defaults (ablated in eval/, see RESULTS.md) ---
    dense_top_k: int = 30
    bm25_top_k: int = 30
    rerank_top_k: int = 8
    rrf_k: int = 60


settings = Settings()
