"""
Pneumonia Diagnostic Hub — FastAPI Backend (Tier 2)
Pydantic-settings configuration loaded from environment variables / .env file.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── Hugging Face Inference Engine (Tier 3) ───────────────────────────────
    hf_space_url: str = "https://shahabkhan396-pneumonia-hub.hf.space"
    hf_api_token: str = ""


    # ─── Internal security (shared with Next.js API Routes) ──────────────────
    internal_api_key: str = ""

    # ─── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: List[str] = ["http://localhost:3000"]

    # ─── Upload / Storage ─────────────────────────────────────────────────────
    max_upload_bytes: int = 33_554_432          # 32 MB
    upload_dir: str = "static/reports"
    samples_dir: str = "static/samples"

    # ─── Samples catalog ──────────────────────────────────────────────────────
    samples_catalog: dict = {
        "sample_normal": {
            "id": "sample_normal",
            "label": "Normal Clear CXR",
            "filename": "sample_normal.jpg",
            "description": "Clear bilateral lung fields — no consolidation or infiltrates.",
            "category": "normal",
        },
        "sample_bacterial": {
            "id": "sample_bacterial",
            "label": "Bacterial Lobar Pneumonia",
            "filename": "sample_bacterial.jpg",
            "description": "Right middle/lower lobe dense consolidation pattern.",
            "category": "bacterial",
        },
        "sample_viral": {
            "id": "sample_viral",
            "label": "Viral Interstitial Pneumonia",
            "filename": "sample_viral.jpg",
            "description": "Bilateral diffuse interstitial infiltrates / reticular opacities.",
            "category": "viral",
        },
    }

    # ─── Allowed file extensions ───────────────────────────────────────────────
    allowed_extensions: List[str] = ["png", "jpg", "jpeg", "webp", "dcm"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
