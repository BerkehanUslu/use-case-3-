"""
06_api_deployment.py — FastAPI REST API for the French real estate assistant.

Wraps modules 03 (Q&A), 04 (summarization), 05 (sentiment), and 02 (vector search)
behind a clean HTTP interface. All ML imports are lazy so the server starts even if
models have not been downloaded yet.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Assistant Immobilier API",
    description="API pour l'analyse du marché immobilier français",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QuestionRequest(BaseModel):
    question: str
    context: str | None = None


class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 150


class SentimentRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5


# ---------------------------------------------------------------------------
# Mock / fallback data
# ---------------------------------------------------------------------------

MOCK_COMMUNES = [
    {
        "commune": "Paris",
        "avg_price_m2": 10500,
        "transactions": 18200,
        "region": "Île-de-France",
    },
    {
        "commune": "Lyon",
        "avg_price_m2": 5200,
        "transactions": 8400,
        "region": "Auvergne-Rhône-Alpes",
    },
    {
        "commune": "Marseille",
        "avg_price_m2": 3800,
        "transactions": 7100,
        "region": "Provence-Alpes-Côte d'Azur",
    },
    {
        "commune": "Bordeaux",
        "avg_price_m2": 4900,
        "transactions": 6300,
        "region": "Nouvelle-Aquitaine",
    },
    {
        "commune": "Toulouse",
        "avg_price_m2": 3600,
        "transactions": 6800,
        "region": "Occitanie",
    },
    {
        "commune": "Nice",
        "avg_price_m2": 5500,
        "transactions": 5200,
        "region": "Provence-Alpes-Côte d'Azur",
    },
    {
        "commune": "Nantes",
        "avg_price_m2": 4100,
        "transactions": 5900,
        "region": "Pays de la Loire",
    },
    {
        "commune": "Strasbourg",
        "avg_price_m2": 3700,
        "transactions": 4800,
        "region": "Grand Est",
    },
    {
        "commune": "Montpellier",
        "avg_price_m2": 3500,
        "transactions": 5100,
        "region": "Occitanie",
    },
    {
        "commune": "Rennes",
        "avg_price_m2": 3900,
        "transactions": 4600,
        "region": "Bretagne",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_META_PATH = os.path.join(os.path.dirname(__file__), "indexes", "real_estate_meta.json")


def _load_communes() -> list[dict]:
    """Load communes from the metadata index, fall back to mock data."""
    if os.path.exists(_META_PATH):
        try:
            with open(_META_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Accept either a list or a dict with a "communes" key
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("communes", list(data.values()))
        except Exception:
            pass
    return MOCK_COMMUNES


def _model_status() -> dict[str, Any]:
    """Return a dict describing which ML components are available."""
    status: dict[str, Any] = {}

    for label, module_name in [
        ("qa", "03_qa_system"),
        ("summarization", "04_summarization"),
        ("sentiment", "05_sentiment_analysis"),
        ("vector_index", "02_vector_indexing"),
    ]:
        try:
            # importlib avoids polluting sys.modules with partial imports
            import importlib.util

            spec = importlib.util.find_spec(module_name)
            status[label] = "available" if spec is not None else "not_found"
        except Exception:
            status[label] = "error"

    return status


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness / readiness probe."""
    return {"status": "ok", "version": "1.0.0", "models": _model_status()}


@app.post("/qa")
def qa(request: QuestionRequest):
    """Answer a real-estate question using the Q&A module."""
    try:
        from importlib import import_module

        qa_mod = import_module("03_qa_system")
        result = qa_mod.answer_question(request.question, request.context)
        return result
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Le module Q&A (03_qa_system) n'est pas disponible : {exc}. "
                "Assurez-vous que les modèles sont téléchargés."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur lors du traitement Q&A : {exc}",
        ) from exc


@app.post("/summarize")
def summarize(request: SummarizeRequest):
    """Summarize a piece of text using the summarization module."""
    try:
        from importlib import import_module

        summ_mod = import_module("04_summarization")
        result = summ_mod.summarize_text(request.text, max_length=request.max_length)
        return result
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Le module de résumé (04_summarization) n'est pas disponible : {exc}. "
                "Assurez-vous que les modèles sont téléchargés."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur lors du résumé : {exc}",
        ) from exc


@app.post("/sentiment")
def sentiment(request: SentimentRequest):
    """Analyse le sentiment d'un texte."""
    try:
        from importlib import import_module

        sent_mod = import_module("05_sentiment_analysis")
        result = sent_mod.analyze_sentiment(request.text)
        return result
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Le module d'analyse de sentiment (05_sentiment_analysis) "
                f"n'est pas disponible : {exc}. "
                "Assurez-vous que les modèles sont téléchargés."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur lors de l'analyse de sentiment : {exc}",
        ) from exc


@app.get("/communes")
def list_communes():
    """Return the list of communes with real-estate statistics."""
    return _load_communes()


@app.get("/communes/{commune}")
def get_commune(commune: str):
    """Return statistics for a specific commune (case-insensitive)."""
    communes = _load_communes()
    commune_lower = commune.lower()
    for entry in communes:
        name = entry.get("commune", entry.get("name", ""))
        if name.lower() == commune_lower:
            return entry
    raise HTTPException(
        status_code=404,
        detail=f"Commune '{commune}' introuvable.",
    )


@app.post("/search")
def search(request: SearchRequest):
    """Semantic search over the vector index; falls back to keyword search."""
    # Try vector search first
    try:
        from importlib import import_module

        vec_mod = import_module("02_vector_indexing")
        results = vec_mod.search(request.query, k=request.k)
        return {"results": results, "backend": "vector"}
    except ImportError:
        pass  # Fall through to keyword search
    except Exception:
        pass  # Fall through to keyword search

    # Keyword fallback on mock commune data
    query_lower = request.query.lower()
    communes = _load_communes()
    matches = [
        c
        for c in communes
        if query_lower in c.get("commune", "").lower()
        or query_lower in c.get("region", "").lower()
    ]
    return {
        "results": matches[: request.k],
        "backend": "keyword_fallback",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
