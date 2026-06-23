"""FastAPI server — exposes the Nyaya-RAG pipeline as a REST API.

Endpoints:
    GET  /health          — liveness check
    POST /query           — main Q&A endpoint
    GET  /query/{id}      — retrieve a specific chunk by ID (for debugging)

Run locally (after indexing is done):
    uvicorn nyaya.api.app:app --host 0.0.0.0 --port 8080 --reload

On CHAMP (after vLLM is running on port 8000):
    uvicorn nyaya.api.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nyaya.config import settings
from nyaya.schema import Era

log = logging.getLogger("nyaya.api")

# --- request / response models -----------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    era: str | None = Field(
        default=None,
        description="'old_code', 'new_code', or null for auto-detect",
    )
    top_k: int = Field(default=8, ge=1, le=20)


class ChunkOut(BaseModel):
    id: str
    act: str | None
    section: str | None
    era: str
    text: str


class VerificationOut(BaseModel):
    """Citation-verification summary for the answer (ADR-005)."""

    total: int
    verified: int
    ungrounded: int
    hallucinated: int
    precision: float
    hallucination_rate: float


class QueryResponse(BaseModel):
    question: str
    era_used: str | None
    answer: str  # verified answer — unverifiable citations stripped (ADR-005)
    citations: list[str]  # only citations that passed verification
    verification: VerificationOut
    chunks: list[ChunkOut]
    model: str
    prompt_tokens: int
    completion_tokens: int


# --- era auto-detection -------------------------------------------------------

_OLD_KEYWORDS = re.compile(
    r"\b(ipc|crpc|iea|indian penal code|criminal procedure|evidence act"
    r"|before\s+2024|before\s+july|pre.?2024|old\s+law|old\s+code)\b",
    re.IGNORECASE,
)
_NEW_KEYWORDS = re.compile(
    r"\b(bns|bnss|bsa|bharatiya|after\s+2024|after\s+july|post.?2024"
    r"|new\s+law|new\s+code)\b",
    re.IGNORECASE,
)


def _detect_era(question: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if _OLD_KEYWORDS.search(question):
        return Era.OLD_CODE.value
    if _NEW_KEYWORDS.search(question):
        return Era.NEW_CODE.value
    return Era.NEW_CODE.value  # default to current law


# --- app lifecycle ------------------------------------------------------------

_retriever = None
_answerer = None
_verifier = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _retriever, _answerer, _verifier
    log.info("loading retriever (bge-m3 + Qdrant + BM25)…")
    from nyaya.retrieval.hybrid import HybridRetriever
    _retriever = HybridRetriever()

    log.info("loading answerer (vLLM client)…")
    from nyaya.generation.answer import LegalAnswerer
    _answerer = LegalAnswerer()

    log.info("loading citation verifier (corpus section index)…")
    from nyaya.eval.verify import CitationVerifier
    _verifier = CitationVerifier.from_processed_dir(settings.processed_dir)

    log.info("Nyaya-RAG API ready")
    yield
    log.info("shutting down")


app = FastAPI(
    title="Nyaya-RAG",
    description="Era-aware RAG over Indian statutes — fully local, zero external APIs",
    version="0.1.0",
    lifespan=lifespan,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)


# --- endpoints ----------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "retriever": "loaded" if _retriever else "not loaded"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if _retriever is None or _answerer is None or _verifier is None:
        raise HTTPException(status_code=503, detail="retriever not ready")

    era_used = _detect_era(req.question, req.era)
    log.info("query: era=%s q=%r", era_used, req.question[:80])

    chunks = _retriever.retrieve(req.question, era=era_used, final_k=req.top_k)
    result = _answerer.answer(req.question, chunks)

    # ADR-005: verify citations against the corpus + retrieved context, strip
    # the unverifiable ones, and report the rates.
    verdict = _verifier.verify(result.answer, chunks)
    if verdict.n_hallucinated or verdict.n_ungrounded:
        log.warning(
            "stripped %d unverifiable citation(s): %d hallucinated, %d ungrounded",
            verdict.n_hallucinated + verdict.n_ungrounded,
            verdict.n_hallucinated,
            verdict.n_ungrounded,
        )

    return QueryResponse(
        question=req.question,
        era_used=era_used,
        answer=verdict.clean_answer,
        citations=[c.raw for c in verdict.citations if c.verified],
        verification=VerificationOut(
            total=verdict.total,
            verified=verdict.n_verified,
            ungrounded=verdict.n_ungrounded,
            hallucinated=verdict.n_hallucinated,
            precision=verdict.precision,
            hallucination_rate=verdict.hallucination_rate,
        ),
        chunks=[
            ChunkOut(
                id=c.id,
                act=c.act,
                section=c.section,
                era=c.era.value,
                text=c.text[:400] + "…" if len(c.text) > 400 else c.text,
            )
            for c in chunks
        ],
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
