"""
02_vector_indexing.py
---------------------
FAISS semantic search index for French commune real estate data.

Builds embeddings using sentence-transformers and indexes commune summaries
so they can be retrieved via semantic similarity search.
"""

import argparse
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
BAN_COORDS_CACHE_PATH = INDEXES_DIR / "ban_coords_cache.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model_cache = None  # Reason: avoid reloading the 400 MB model on every search call
_index_cache: tuple | None = None  # Reason: avoid re-reading FAISS index from disk on every call
_ban_error_logged = False

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


def _offline_mode_enabled() -> bool:
    """Return True when Hugging Face libraries should stay fully offline."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _normalise_commune_code(code: object) -> str:
    """Return a BAN-friendly 5-digit commune code when possible."""
    if code is None:
        return ""
    raw = str(code).strip()
    if not raw or raw.lower() == "nan":
        return ""
    if raw.isdigit():
        return raw.zfill(5)
    return raw


def _ban_citycode(code: object) -> str:
    """Return a BAN citycode only when the source value is already a full INSEE code."""
    if code is None:
        return ""
    raw = str(code).strip()
    if raw.isdigit() and len(raw) == 5:
        return raw
    return ""


def _get_model():
    """Load (and cache) the sentence-transformer model."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print(
            "[error] sentence-transformers is not installed.\n"
            "Install it with:  pip install sentence-transformers"
        )
        sys.exit(1)
    offline = _offline_mode_enabled()
    if offline:
        print(f"[info] Loading model in offline mode: {MODEL_NAME}")
    else:
        print(f"[info] Loading model: {MODEL_NAME}")
    _model_cache = SentenceTransformer(MODEL_NAME, local_files_only=offline)
    return _model_cache


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
# BAN (Base Adresse Nationale) geocoding
# ---------------------------------------------------------------------------


def _fetch_ban_coordinates(commune: str, code: str) -> tuple[float, float] | None:
    """Fetch (lat, lon) for a single commune from the BAN API.

    Uses api-adresse.data.gouv.fr — free, no auth required.
    Returns None on any network/parse error so callers can degrade gracefully.
    """
    global _ban_error_logged
    try:
        import certifi
        import socket
        import ssl
        import time
        import urllib.error
        import urllib.parse
        import urllib.request

        params = urllib.parse.urlencode({"q": commune, "type": "municipality", "limit": "1"})
        citycode = _ban_citycode(code)
        if citycode and citycode != "00000":
            params += f"&citycode={citycode}"
        url = f"https://api-adresse.data.gouv.fr/search/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "use-case-3-real-estate-assistant/1.0"})
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=5, context=ssl_context) as resp:
                    payload = json.loads(resp.read().decode())
                features = payload.get("features", [])
                if features:
                    lon, lat = features[0]["geometry"]["coordinates"]
                    return float(lat), float(lon)
                return None
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt == 2:
                    raise exc
            time.sleep(0.5 * (attempt + 1))
    except Exception as exc:
        if not _ban_error_logged:
            print(f"[warn] BAN lookup failed for '{commune}' ({code}): {exc}")
            _ban_error_logged = True
    return None


def _geocode_communes(data: list[dict], *, retry_empty_cache: bool = True) -> dict[str, tuple[float, float]]:
    """Geocode unique communes using BAN API, persisting results in a local cache.

    Returns a dict mapping commune name → (lat, lon).
    Communes that cannot be geocoded are absent from the result.
    """
    import time

    # Load existing cache from disk
    cache: dict[str, list[float]] = {}
    if BAN_COORDS_CACHE_PATH.exists():
        try:
            cache = json.loads(BAN_COORDS_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # Identify unique communes not yet cached. Optionally retry stale empty values,
    # which usually come from a previous offline / network-failed run.
    to_fetch: dict[str, str] = {}  # commune → code
    for row in data:
        commune = str(row.get("commune") or row.get("nom_commune", "")).strip()
        code = _normalise_commune_code(row.get("code") or row.get("code_commune", ""))
        cached = cache.get(commune)
        should_fetch = commune and (
            commune not in cache or (retry_empty_cache and (not cached or len(cached) != 2))
        )
        if should_fetch and commune not in to_fetch:
            to_fetch[commune] = code

    if to_fetch:
        print(f"[info] Fetching BAN coordinates for {len(to_fetch)} new commune(s)…")
        failures = 0
        for i, (commune, code) in enumerate(to_fetch.items(), 1):
            coords = _fetch_ban_coordinates(commune, code)
            cache[commune] = list(coords) if coords else []
            if coords is None:
                failures += 1
            if i % 50 == 0:
                print(f"  … {i}/{len(to_fetch)} done")
            if i % 500 == 0:
                INDEXES_DIR.mkdir(parents=True, exist_ok=True)
                BAN_COORDS_CACHE_PATH.write_text(
                    json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            time.sleep(0.05)  # Reason: polite rate-limit — 20 req/s max

        INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        BAN_COORDS_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        geocoded = sum(1 for v in cache.values() if v)
        print(f"[info] BAN geocoding done: {geocoded}/{len(cache)} communes with coordinates")
        if failures:
            print(f"[warn] BAN geocoding failed for {failures}/{len(to_fetch)} requested communes.")
    else:
        print(f"[info] BAN cache up-to-date ({len(cache)} communes).")

    return {k: (v[0], v[1]) for k, v in cache.items() if v}


def _attach_coordinates_to_metadata(
    metadata: list[dict],
    coords_cache: dict[str, tuple[float, float]],
) -> list[dict]:
    """Return metadata rows with lat/lon refreshed from the BAN cache."""
    updated: list[dict] = []
    with_coords = 0
    for row in metadata:
        entry = dict(row)
        coords = coords_cache.get(str(entry.get("commune", "")).strip())
        if coords:
            entry["lat"], entry["lon"] = coords
            with_coords += 1
        else:
            entry["lat"] = None
            entry["lon"] = None
        updated.append(entry)
    print(f"[info] Metadata coordinate refresh: {with_coords}/{len(updated)} rows now have lat/lon.")
    return updated


def refresh_metadata_coordinates(*, retry_empty_cache: bool = True) -> int:
    """Refresh lat/lon in existing metadata without rebuilding embeddings."""
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {META_PATH}. Build the index once before refreshing coordinates."
        )

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    coords_cache = _geocode_communes(metadata, retry_empty_cache=retry_empty_cache)
    updated = _attach_coordinates_to_metadata(metadata, coords_cache)
    META_PATH.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    geocoded = sum(1 for row in updated if row.get("lat") is not None and row.get("lon") is not None)
    print(f"[info] Metadata saved to {META_PATH}")
    return geocoded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index(
    data: list[dict] | None = None,
    *,
    skip_geocoding: bool = False,
    retry_empty_cache: bool = True,
) -> None:
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

    # ---- geocode communes via BAN API (results cached to disk) -------------
    coords_cache = {}
    if skip_geocoding:
        print("[info] Skipping BAN geocoding; metadata lat/lon will remain empty.")
    else:
        coords_cache = _geocode_communes(data, retry_empty_cache=retry_empty_cache)

    # ---- build metadata records & texts ------------------------------------
    metadata: list[dict] = []
    texts: list[str] = []

    for idx, row in enumerate(data):
        # normalise key names that might differ in real parquet
        commune = row.get("commune") or row.get("nom_commune", f"Commune_{idx}")
        code = _normalise_commune_code(row.get("code") or row.get("code_commune", "00000"))
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
        coords = coords_cache.get(commune)
        metadata.append(
            {
                "id": idx,
                "commune": commune,
                "code": code,
                "text": text,
                "prix_moyen_m2": prix_moyen,
                "prix_median_m2": prix_median,
                "nb_transactions": nb_trans,
                "annee": annee,
                "type_bien": type_bien,
                "lat": coords[0] if coords else None,
                "lon": coords[1] if coords else None,
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
    global _index_cache
    if _index_cache is not None:
        return _index_cache

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
    _index_cache = (index, metadata)
    return _index_cache


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
    parser = argparse.ArgumentParser(description="Build or refresh the real-estate FAISS index.")
    parser.add_argument(
        "--refresh-coords-only",
        action="store_true",
        help="Refresh lat/lon in existing metadata using BAN without rebuilding embeddings.",
    )
    parser.add_argument(
        "--skip-geocoding",
        action="store_true",
        help="Build embeddings/index without calling BAN.",
    )
    parser.add_argument(
        "--no-search-test",
        action="store_true",
        help="Skip the final example search after building the index.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face model loading to use local cache only.",
    )
    parser.add_argument(
        "--keep-empty-ban-cache",
        action="store_true",
        help="Do not retry communes that currently have empty BAN cache entries.",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    retry_empty_cache = not args.keep_empty_ban_cache

    if args.refresh_coords_only:
        print("=== Refreshing metadata coordinates ===")
        geocoded = refresh_metadata_coordinates(retry_empty_cache=retry_empty_cache)
        if geocoded == 0:
            print("[warn] No coordinates were populated. Check BAN connectivity before retrying.")
    else:
        print("=== Building FAISS index ===")
        build_index(
            skip_geocoding=args.skip_geocoding,
            retry_empty_cache=retry_empty_cache,
        )

        if not args.no_search_test:
            print("\n=== Test search: 'prix immobilier Paris appartement' ===")
            results = search("prix immobilier Paris appartement", k=5)
            for rank, r in enumerate(results, 1):
                print(
                    f"  [{rank}] {r['commune']} ({r['code']}) | "
                    f"score={r['score']:.4f} | "
                    f"{r['prix_moyen_m2']:,.0f} €/m² | "
                    f"{r['type_bien']} {r['annee']}"
                )
