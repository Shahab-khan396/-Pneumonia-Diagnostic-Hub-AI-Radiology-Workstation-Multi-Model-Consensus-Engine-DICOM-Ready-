"""
Pneumonia Diagnostic Hub — FastAPI Backend (Tier 2)
FastAPI application factory with CORS, rate limiting, and router registration.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import get_settings
from core.sample_manager import ensure_samples_generated
from routers import compare, health, predict, report, samples


# ─── Lifespan: runs once on startup ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Generate synthetic sample CXRs and ensure output directories exist."""
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.samples_dir).mkdir(parents=True, exist_ok=True)
    ensure_samples_generated()
    yield


# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ─── App factory ──────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Pneumonia Diagnostic Hub API",
        description=(
            "Multi-model CNN inference engine for chest radiograph pneumonia screening. "
            "Tier 2 backend — bridges the Next.js frontend (Vercel) and the "
            "HF Spaces AI inference engine (Tier 3)."
        ),
        version="2.4.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        max_age=3600,
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Global 422 → readable JSON ────────────────────────────────────────────
    @app.exception_handler(422)
    async def validation_exception_handler(request: Request, exc):
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": "Request validation failed.", "detail": str(exc)},
        )

    # ── Static files ──────────────────────────────────────────────────────────
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.samples_dir).mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # ── Routers ───────────────────────────────────────────────────────────────
    PREFIX = "/api/v1"
    app.include_router(health.router,  prefix=PREFIX)
    app.include_router(predict.router, prefix=PREFIX)
    app.include_router(compare.router, prefix=PREFIX)
    app.include_router(report.router,  prefix=PREFIX)
    app.include_router(samples.router, prefix=PREFIX)

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "Pneumonia Diagnostic Hub API", "docs": "/docs", "version": "2.4.0"}

    return app


app = create_app()
