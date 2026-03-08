"""
02_vector_indexing.py
---------------------
FAISS semantic search index for French commune real estate data.

Builds embeddings using sentence-transformers and indexes commune summaries
so they can be retrieved via semantic similarity search.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / "data"
INDEXES_DIR = BACKEND_DIR / "indexes"
PARQUET_PATH = DATA_DIR / "dvf_communes.parquet"
FAISS_INDEX_PATH = INDEXES_DIR / "real_estate.faiss"
META_PATH = INDEXES_DIR / "real_estate_meta.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_COMMUNES = [
    {"commune": "Paris",        "code": "75056", "prix_moyen_m2": 10500.0, "prix_median_m2": 9800.0,  "nb_transactions": 45230, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Lyon",         "code": "69123", "prix_moyen_m2": 5200.0,  "prix_median_m2": 4900.0,  "nb_transactions": 18750, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Marseille",    "code": "13055", "prix_moyen_m2": 3800.0,  "prix_median_m2": 3500.0,  "nb_transactions": 15600, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Bordeaux",     "code": "33063", "prix_moyen_m2": 4600.0,  "prix_median_m2": 4300.0,  "nb_transactions": 12400, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Toulouse",     "code": "31555", "prix_moyen_m2": 3900.0,  "prix_median_m2": 3700.0,  "nb_transactions": 14100, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Nantes",       "code": "44109", "prix_moyen_m2": 4100.0,  "prix_median_m2": 3900.0,  "nb_transactions": 11200, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Strasbourg",   "code": "67482", "prix_moyen_m2": 3700.0,  "prix_median_m2": 3500.0,  "nb_transactions": 9800,  "type_bien": "Appartement", "annee": 2023},
    {"commune": "Montpellier",  "code": "34172", "prix_moyen_m2": 3600.0,  "prix_median_m2": 3400.0,  "nb_transactions": 10500, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Nice",         "code": "06088", "prix_moyen_m2": 5800.0,  "prix_median_m2": 5400.0,  "nb_transactions": 13300, "type_bien": "Appartement", "annee": 2023},
    {"commune": "Rennes",       "code": "35238", "prix_moyen_m2": 3800.0,  "prix_median_m2": 3600.0,  "nb_transactions": 9100,  "type_bien": "Appartement", "annee": 2023},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_price(value: float) -> str:
    """Format a price value with French thousand-separator style."""
    return f"{value:,.0f}".replace(",", " ")


def _build_text(row: dict) -> str:
    """Build a French text description for a commune row."""
    return (
        f"Commune: {row['commune']} ({row['code']}). "
        f"Prix moyen: {_format_price(row['prix_moyen_m2'])} €/m². "
        f"Transactions: {_format_price(row['nb_transactions'])}. "
        f"Prix médian: {_format_price(row['prix_median_m2'])} €/m². "
        f"Type: {row['type_bien']}. "
        f"Année: {row['annee']}."
    )


def _load_parquet_data() -> list[dict]:
    """Try to load commune data from Parquet; return None if unavailable."""
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(PARQUET_PATH)
        records = df.to_dict(orient="records")
        print(f"[info] Loaded {len(records)} rows from {PARQUET_PATH}")
        return records
    except Exception as exc:
        print(f"[warn] Could not read parquet ({exc}); using mock data.")
        return None


def _get_model():
    """Load (and cache) the sentence-transformer model."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print(
            "[error] sentence-transformers is not installed.\n"
            "Install it with:  pip install sentence-transformers"
        )
        sys.exit(1)
    print(f"[info] Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def _get_faiss():
    """Import faiss, raising a clear error if not available."""
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is not installed. Install it with:  pip install faiss-cpu"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index(data: list[dict] | None = None) -> None:
    """Build and save FAISS index from data or mock data.

    Parameters
    ----------
    data:
        Optional list of dicts with commune real estate fields.  When None the
        function first attempts to read the Parquet file and falls back to the
        built-in mock data.
    """
    faiss = _get_faiss()

    # ---- resolve data -------------------------------------------------------
    if data is None:
        if PARQUET_PATH.exists():
            data = _load_parquet_data()
        if data is None:
            print("[info] Using built-in mock data for 10 French communes.")
            data = MOCK_COMMUNES

    # ---- build metadata records & texts ------------------------------------
    metadata: list[dict] = []
    texts: list[str] = []

    for idx, row in enumerate(data):
        # normalise key names that might differ in real parquet
        commune = row.get("commune") or row.get("nom_commune", f"Commune_{idx}")
        code = row.get("code") or row.get("code_commune", "00000")
        prix_moyen = float(row.get("prix_moyen_m2") or row.get("prix_moyen", 0.0))
        prix_median = float(row.get("prix_median_m2") or row.get("prix_median", prix_moyen))
        nb_trans = int(row.get("nb_transactions") or row.get("transactions", 0))
        type_bien = row.get("type_bien", "Appartement")
        annee = int(row.get("annee") or row.get("year", 2023))

        normalised = {
            "commune": commune,
            "code": code,
            "prix_moyen_m2": prix_moyen,
            "prix_median_m2": prix_median,
            "nb_transactions": nb_trans,
            "type_bien": type_bien,
            "annee": annee,
        }
        text = _build_text(normalised)
        texts.append(text)
        metadata.append(
            {
                "id": idx,
                "commune": commune,
                "code": code,
                "text": text,
                "prix_moyen_m2": prix_moyen,
                "annee": annee,
                "type_bien": type_bien,
            }
        )

    # ---- embed -------------------------------------------------------------
    model = _get_model()
    print(f"[info] Encoding {len(texts)} texts …")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # ---- normalise for cosine similarity via inner product -----------------
    faiss.normalize_L2(embeddings)

    # ---- build FAISS index -------------------------------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalised = cosine
    index.add(embeddings)
    print(f"[info] FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ---- persist -----------------------------------------------------------
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"[info] Index saved to {FAISS_INDEX_PATH}")

    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[info] Metadata saved to {META_PATH}")


def load_index() -> tuple:
    """Load saved index and metadata.

    Returns
    -------
    (index, metadata)
        index    – faiss index object
        metadata – list of dicts
    """
    faiss = _get_faiss()

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run build_index() first."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {META_PATH}. "
            "Run build_index() first."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = json.loads(META_PATH.read_text())
    print(f"[info] Loaded index ({index.ntotal} vectors) and {len(metadata)} metadata records.")
    return index, metadata


def search(query: str, k: int = 5) -> list[dict]:
    """Search the index for the k most similar entries to the query.

    Parameters
    ----------
    query:
        Free-text French (or multilingual) search query.
    k:
        Number of results to return.

    Returns
    -------
    List of metadata dicts, each with an added ``"score"`` field (cosine
    similarity, higher is better).
    """
    faiss = _get_faiss()
    index, metadata = load_index()
    model = _get_model()

    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    k_clamped = min(k, index.ntotal)
    scores, indices = index.search(query_vec, k_clamped)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(score)
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Building FAISS index ===")
    build_index()

    print("\n=== Test search: 'prix immobilier Paris appartement' ===")
    results = search("prix immobilier Paris appartement", k=5)
    for rank, r in enumerate(results, 1):
        print(
            f"  [{rank}] {r['commune']} ({r['code']}) | "
            f"score={r['score']:.4f} | "
            f"{r['prix_moyen_m2']:,.0f} €/m² | "
            f"{r['type_bien']} {r['annee']}"
        )
