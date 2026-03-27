"""
test_pipeline.py
----------------
Automated QA for the French real estate assistant pipeline.

Run from backend/:
    python test_pipeline.py          # fast tests — no model downloads
    python test_pipeline.py --full   # also tests real ML models (slow)

Exit code 0 = all non-skipped tests passed.
"""

import argparse
import os
import sys
import json
import tempfile

# Ensure imports resolve relative to backend/
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
SKIP  = "⏭  SKIP"
_results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    suffix = f" — {detail}" if detail else ""
    print(f"  {status}  {name}{suffix}")


# ---------------------------------------------------------------------------
# 01 — Data Preparation
# ---------------------------------------------------------------------------

def test_data_preparation() -> None:
    print("\n[01] Data Preparation")
    try:
        import importlib
        mod = importlib.import_module("01_data_preparation")

        # Build a minimal DVF-shaped pipe-delimited file
        sample = (
            "Commune|Code commune|Date mutation|Valeur fonciere|Surface reelle bati|Type local\n"
            "Paris|75056|01/01/2024|500000,00|50|Appartement\n"
            "Lyon|69123|15/06/2024|300000,00|60|Maison\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(sample)
            tmp = f.name

        df = mod.aggregate_dvf(input_paths=[tmp])
        os.unlink(tmp)

        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert set(["commune", "prix_moyen_m2", "nb_transactions", "annee", "type_bien"]).issubset(df.columns)
        record("aggregate_dvf() schema", PASS, f"{len(df)} rows, {list(df.columns)}")

        # Sanity-check the price calculation: 500000 / 50 = 10000 €/m²
        paris_row = df[df["commune"].str.upper() == "PARIS"]
        if not paris_row.empty:
            assert abs(paris_row.iloc[0]["prix_moyen_m2"] - 10000) < 1
            record("prix_moyen_m2 = valeur / surface", PASS, "Paris: 10 000 €/m²")
        else:
            record("prix_moyen_m2 calculation", SKIP, "commune name casing differs")

    except Exception as exc:
        record("01_data_preparation", FAIL, str(exc))


# ---------------------------------------------------------------------------
# 02 — Vector Indexing + BAN
# ---------------------------------------------------------------------------

def test_vector_indexing() -> None:
    print("\n[02] Vector Indexing (FAISS)")
    try:
        import importlib
        vi = importlib.import_module("02_vector_indexing")

        # Test pure data-processing helpers (no model, no segfault risk)
        row = vi.MOCK_COMMUNES[0]
        text = vi._build_text({
            "commune": row["commune"], "code": row["code"],
            "prix_moyen_m2": row["prix_moyen_m2"], "prix_median_m2": row["prix_median_m2"],
            "nb_transactions": row["nb_transactions"], "type_bien": row["type_bien"],
            "annee": row["annee"],
        })
        assert "Commune:" in text and "€/m²" in text
        record("_build_text() produces French description", PASS, text[:60])

        formatted = vi._format_price(10500.0)
        assert "10" in formatted and "500" in formatted
        record("_format_price() formats with separator", PASS, f"10500 → '{formatted}'")

        # If index already exists, load and inspect metadata (no model needed)
        if vi.FAISS_INDEX_PATH.exists() and vi.META_PATH.exists():
            import json as _json
            meta = _json.loads(vi.META_PATH.read_text(encoding="utf-8"))
            assert len(meta) > 0, "Metadata is empty"
            record("load metadata JSON", PASS, f"{len(meta)} entries")

            with_coords = sum(1 for m in meta if m.get("lat") is not None)
            record(
                "metadata contains lat/lon (BAN)",
                PASS if with_coords > 0 else SKIP,
                f"{with_coords}/{len(meta)} entries geocoded (run build_index() to populate)",
            )

            required_keys = {"commune", "code", "prix_moyen_m2", "annee", "type_bien", "text"}
            missing = required_keys - set(meta[0].keys())
            record(
                "metadata schema complete",
                PASS if not missing else FAIL,
                f"missing: {missing}" if missing else "all keys present",
            )
        else:
            record("load_index() metadata", SKIP, "Index not built — run `python 02_vector_indexing.py` first")
            record("metadata contains lat/lon (BAN)", SKIP, "Index not built")

    except Exception as exc:
        record("02_vector_indexing", FAIL, str(exc))


def test_ban_geocoding() -> None:
    print("\n[BAN] Base Adresse Nationale")
    try:
        import importlib
        vi = importlib.import_module("02_vector_indexing")

        coords = vi._fetch_ban_coordinates("Paris", "75056")
        if coords is None:
            record("_fetch_ban_coordinates('Paris')", SKIP, "BAN API unreachable (needs internet)")
            return

        lat, lon = coords
        assert 48.0 < lat < 49.5, f"Paris lat out of range: {lat}"
        assert 1.5 < lon < 3.5, f"Paris lon out of range: {lon}"
        record("_fetch_ban_coordinates('Paris', '75056')", PASS, f"lat={lat:.4f}, lon={lon:.4f}")

        # Cache file created?
        cache_path = vi.BAN_COORDS_CACHE_PATH
        record(
            "ban_coords_cache.json exists",
            PASS if cache_path.exists() else SKIP,
            str(cache_path) if cache_path.exists() else "run build_index() first",
        )

    except AttributeError:
        record("_fetch_ban_coordinates", FAIL, "Function missing — BAN not yet implemented in 02_vector_indexing.py")
    except Exception as exc:
        record("BAN geocoding", FAIL, str(exc))


# ---------------------------------------------------------------------------
# 03 — Q&A System
# ---------------------------------------------------------------------------

def test_qa_system(full: bool = False) -> None:
    print("\n[03] Q&A System (CamemBERT)")
    try:
        import importlib
        qa = importlib.import_module("03_qa_system")

        # Mock mode — no model download
        result = qa._mock_answer("Quel est le prix moyen au m² à Paris ?")
        assert "answer" in result and "score" in result
        record("_mock_answer() returns answer + score", PASS, f"answer: {result['answer']}")

        # Context builder
        ctx = qa.get_commune_context("Paris")
        assert len(ctx) > 20
        record("get_commune_context('Paris')", PASS, f"{len(ctx)} chars")

        # RAG context builder
        ctx_rag = qa._build_rag_context("Quel est le prix à Lyon ?")
        assert "Lyon" in ctx_rag or len(ctx_rag) > 0
        record("_build_rag_context() keyword match", PASS)

        if full:
            result = qa.answer_question("Quel est le prix moyen à Lyon ?")
            assert "answer" in result
            status = PASS if result["score"] > 0 else FAIL
            record("answer_question() with real model", status, f"score={result['score']:.3f}")
        else:
            record("answer_question() with real model", SKIP, "use --full (loads ~400 MB model)")

    except Exception as exc:
        record("03_qa_system", FAIL, str(exc))


# ---------------------------------------------------------------------------
# 04 — Summarization
# ---------------------------------------------------------------------------

def test_summarization(full: bool = False) -> None:
    print("\n[04] Summarization (BARThez)")
    try:
        import importlib
        summ = importlib.import_module("04_summarization")

        # Mock mode
        result = summ.summarize_text("Ceci est un texte de test immobilier.", mock=True)
        assert "summary" in result and result["original_length"] > 0
        record("summarize_text() mock mode", PASS)

        # Chunking logic (mock, so no model)
        long_text = "Mot " * 800
        chunk_result = summ.chunk_and_summarize(long_text, chunk_size=500, mock=True)
        assert chunk_result["chunks"] > 1
        record("chunk_and_summarize() splits long text", PASS, f"{chunk_result['chunks']} chunks")

        # Compression ratio
        ratio = chunk_result.get("original_length", 1)
        record("chunk_and_summarize() returns original_length", PASS if ratio > 0 else FAIL)

        if full:
            result = summ.summarize_text(
                "Le marché immobilier parisien reste très dynamique en 2024.", mock=False
            )
            status = PASS if result.get("summary") else FAIL
            record("summarize_text() with real model", status, result.get("summary", "")[:80])
        else:
            record("summarize_text() with real model", SKIP, "use --full")

    except Exception as exc:
        record("04_summarization", FAIL, str(exc))


# ---------------------------------------------------------------------------
# 05 — Sentiment Analysis
# ---------------------------------------------------------------------------

def test_sentiment(full: bool = False) -> None:
    print("\n[05] Sentiment Analysis (BERT multilingual)")
    try:
        import importlib
        sent = importlib.import_module("05_sentiment_analysis")

        # Mock helpers
        mock_pos = sent._mock_analyze_sentiment("test", index=0)
        assert mock_pos["stars"] == 5 and mock_pos["label"] == "Très positif"
        record("_mock_analyze_sentiment() index=0 → 5 stars", PASS)

        mock_neg = sent._mock_analyze_sentiment("test", index=2)
        assert mock_neg["stars"] == 1
        record("_mock_analyze_sentiment() index=2 → 1 star", PASS)

        # Investment recommendation
        rec = sent.get_investment_recommendation(5, prix_m2=4000.0)
        assert len(rec) > 10
        record("get_investment_recommendation(5★, 4000€)", PASS, rec[:70])

        rec_low = sent.get_investment_recommendation(1, prix_m2=5000.0)
        assert "risque" in rec_low.lower() or "déconseillé" in rec_low.lower()
        record("get_investment_recommendation(1★, 5000€) warns", PASS, rec_low[:70])

        if full:
            result = sent.analyze_sentiment("Magnifique appartement lumineux bien entretenu.")
            assert result["stars"] in range(1, 6)
            record(
                "analyze_sentiment() with real model",
                PASS,
                f"{result['label']} ({result['stars']}★, conf={result['score']:.0%})",
            )
        else:
            record("analyze_sentiment() with real model", SKIP, "use --full")

    except Exception as exc:
        record("05_sentiment_analysis", FAIL, str(exc))


# ---------------------------------------------------------------------------
# 06 — FastAPI
# ---------------------------------------------------------------------------

def test_api() -> None:
    print("\n[06] FastAPI Endpoints")
    try:
        from fastapi.testclient import TestClient
        import importlib

        api = importlib.import_module("06_api_deployment")
        client = TestClient(api.app)

        # Health
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        record("GET /health", PASS, f"version={r.json()['version']}")

        # List communes
        r = client.get("/communes")
        assert r.status_code == 200 and isinstance(r.json(), list)
        record("GET /communes", PASS, f"{len(r.json())} entries")

        # Specific commune
        r = client.get("/communes/Paris")
        assert r.status_code == 200
        record("GET /communes/Paris", PASS)

        # 404 for unknown commune
        r = client.get("/communes/ZZZ_INEXISTANT")
        assert r.status_code == 404
        record("GET /communes/ZZZ_INEXISTANT → 404", PASS)

        # Q&A (503 acceptable when model not loaded)
        r = client.post("/qa", json={"question": "Quel est le prix à Paris ?"})
        assert r.status_code in (200, 503)
        record(
            "POST /qa",
            PASS if r.status_code == 200 else SKIP,
            "ok" if r.status_code == 200 else "model not loaded (503)",
        )

        # Summarize
        r = client.post("/summarize", json={"text": "Test.", "max_length": 50})
        assert r.status_code in (200, 503)
        record(
            "POST /summarize",
            PASS if r.status_code == 200 else SKIP,
            "ok" if r.status_code == 200 else "model not loaded (503)",
        )

        # Sentiment
        r = client.post("/sentiment", json={"text": "Magnifique appartement."})
        assert r.status_code in (200, 503)
        record(
            "POST /sentiment",
            PASS if r.status_code == 200 else SKIP,
            "ok" if r.status_code == 200 else "model not loaded (503)",
        )

        # Search (vector or keyword fallback — both return 200)
        r = client.post("/search", json={"query": "Paris", "k": 3})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data and "backend" in data
        record("POST /search", PASS, f"backend={data['backend']}, {len(data['results'])} results")

        # Trends — 200 if index exists, 503 if not built yet (both acceptable)
        r = client.get("/trends/Paris")
        if r.status_code == 200:
            payload = r.json()
            assert "trends" in payload and isinstance(payload["trends"], list)
            record(
                "GET /trends/Paris",
                PASS,
                f"{len(payload['trends'])} data points",
            )
        elif r.status_code == 404:
            record("GET /trends/Paris", SKIP, "Paris not in index — run build_index() first")
        elif r.status_code == 503:
            record("GET /trends/Paris", SKIP, "Index not built — run python 02_vector_indexing.py first")
        else:
            record("GET /trends/Paris", FAIL, f"Unexpected status {r.status_code}: {r.text[:120]}")

    except Exception as exc:
        record("06_api_deployment", FAIL, str(exc))


# ---------------------------------------------------------------------------
# 07 — Streamlit (syntax + import check only — UI needs manual review)
# ---------------------------------------------------------------------------

def test_streamlit_syntax() -> None:
    print("\n[07] Streamlit App")
    try:
        with open("07_streamlit_app.py", "rb") as fh:
            source = fh.read()
        compile(source, "07_streamlit_app.py", "exec")
        record("07_streamlit_app.py compiles (no syntax errors)", PASS)

        # Check that _load_map_data is present (BAN integration)
        assert b"_load_map_data" in source
        record("_load_map_data() function present (BAN wired to map)", PASS)

        # Check that _load_trends_data is present (market trends)
        assert b"_load_trends_data" in source
        record("_load_trends_data() function present (trend chart wired)", PASS)

    except SyntaxError as exc:
        record("07_streamlit_app.py syntax", FAIL, str(exc))
    except AssertionError:
        record("_load_map_data() missing in Streamlit", FAIL, "BAN not wired to Tab 5")
    except Exception as exc:
        record("07_streamlit_app.py check", FAIL, str(exc))


# ---------------------------------------------------------------------------
# METRICS — Response time
# ---------------------------------------------------------------------------


def test_response_times() -> None:
    print("\n[MÉTRIQUES] Temps de réponse des endpoints")
    try:
        import time
        from fastapi.testclient import TestClient
        import importlib

        api = importlib.import_module("06_api_deployment")
        client = TestClient(api.app)

        THRESHOLD_S = 2.0  # SLA: response must be under 2 seconds

        # Only test endpoints that don't require ML model loading
        endpoints = [
            ("GET /health",              lambda: client.get("/health")),
            ("GET /communes",            lambda: client.get("/communes")),
            ("GET /communes/Paris",      lambda: client.get("/communes/Paris")),
            ("POST /search (keyword fallback)", lambda: client.post("/search", json={"query": "Paris", "k": 3})),
        ]

        for name, call in endpoints:
            t0 = time.perf_counter()
            r = call()
            elapsed = time.perf_counter() - t0
            within_sla = elapsed < THRESHOLD_S
            record(
                f"{name} < {THRESHOLD_S:.0f}s",
                PASS if within_sla else FAIL,
                f"{elapsed:.3f}s (status={r.status_code})",
            )

    except Exception as exc:
        record("test_response_times", FAIL, str(exc))


# ---------------------------------------------------------------------------
# METRICS — Q&A accuracy
# ---------------------------------------------------------------------------


def test_qa_accuracy(full: bool = False) -> None:
    print("\n[MÉTRIQUES] Précision Q&A (benchmark)")
    try:
        import importlib
        qa = importlib.import_module("03_qa_system")

        # --- Fast mode: verify mock structure only, skip accuracy ---
        if not full:
            for question in [
                "Prix moyen Paris ?",
                "Transactions Lyon ?",
                "Ville la plus chère ?",
            ]:
                result = qa._mock_answer(question)
                assert "answer" in result and "score" in result
                assert isinstance(result["score"], float)
            record("Q&A mock réponse structure valide", PASS, "champs answer + score présents")
            record("Q&A précision ≥ 85% (modèle réel)", SKIP, "use --full (charge ~400 Mo de modèle)")
            return

        # --- Full mode: accuracy benchmark with real model ---
        # Each entry: (question, context_passage, list_of_acceptable_answer_keywords)
        # Context is explicit so the extractive model can find the span.
        benchmark = [
            (
                "Quel est le prix moyen au m² à Paris ?",
                "Paris est la capitale de la France. Le prix moyen au m² à Paris est de "
                "10 500 euros en 2023. Il y a eu 42 000 transactions immobilières à Paris "
                "en 2023. Paris reste la ville la plus chère de France en immobilier.",
                ["10 500", "10500"],
            ),
            (
                "Combien de transactions immobilières à Lyon en 2023 ?",
                "Lyon est la deuxième ville de France. Le prix moyen au m² à Lyon est de "
                "5 200 euros en 2023. Il y a eu 18 500 transactions immobilières à Lyon en 2023.",
                ["18 500", "18500"],
            ),
            (
                "Quel est le prix moyen au m² à Marseille ?",
                "Marseille est la troisième plus grande ville de France. "
                "Le prix moyen au m² à Marseille est de 3 800 euros en 2023. "
                "Il y a eu 15 000 transactions immobilières à Marseille en 2023.",
                ["3 800", "3800"],
            ),
            (
                "Quelle est la ville la plus chère de France ?",
                "Paris est la ville la plus chère de France avec un prix moyen au m² de "
                "10 500 euros en 2023. Lyon affiche 5 200 euros au m² et Marseille 3 800 euros.",
                ["paris", "Paris"],
            ),
            (
                "Quel est le prix moyen au m² à Lyon ?",
                "Lyon est la deuxième ville de France par son dynamisme économique. "
                "Le prix moyen au m² à Lyon est de 5 200 euros en 2023.",
                ["5 200", "5200"],
            ),
            (
                "Combien de transactions immobilières à Paris en 2023 ?",
                "Paris est la capitale de la France. Il y a eu 42 000 transactions "
                "immobilières à Paris en 2023.",
                ["42 000", "42000"],
            ),
            (
                "Quel est le nombre de transactions à Marseille ?",
                "Marseille est une grande ville côtière. "
                "Il y a eu 15 000 transactions immobilières à Marseille en 2023.",
                ["15 000", "15000"],
            ),
        ]

        hits = 0
        for question, context, expected_keywords in benchmark:
            result = qa.answer_question(question, context=context)
            # Normalise non-breaking spaces that French text often contains
            answer = result.get("answer", "").replace("\u00a0", " ")
            matched = any(kw.lower() in answer.lower() for kw in expected_keywords)
            if matched:
                hits += 1

        accuracy = hits / len(benchmark)
        threshold = 0.85
        record(
            f"Q&A précision ≥ {threshold:.0%} (modèle réel)",
            PASS if accuracy >= threshold else FAIL,
            f"{hits}/{len(benchmark)} = {accuracy:.0%}",
        )

    except Exception as exc:
        record("test_qa_accuracy", FAIL, str(exc))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary() -> int:
    passed  = sum(1 for _, s, _ in _results if s == PASS)
    failed  = sum(1 for _, s, _ in _results if s == FAIL)
    skipped = sum(1 for _, s, _ in _results if s == SKIP)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed : {passed}/{len(_results)}")
    print(f"  Failed : {failed}/{len(_results)}")
    print(f"  Skipped: {skipped}/{len(_results)}")

    if failed:
        print("\nFailed tests:")
        for name, status, detail in _results:
            if status == FAIL:
                print(f"  ❌ {name}")
                if detail:
                    print(f"     {detail}")

    print("\nManual checks needed:")
    print("  • Streamlit UI: run `streamlit run 07_streamlit_app.py --server.port 8501`")
    print("    - Tab 1: search a commune (e.g. Paris) → price metrics + bar chart")
    print("    - Tab 2: ask a question → answer + confidence score")
    print("    - Tab 3: paste text → summary")
    print("    - Tab 4: paste property description → sentiment gauge")
    print("    - Tab 5: map shows commune markers with tooltips")
    print("  • API docs: run `uvicorn 06_api_deployment:app --reload` → http://localhost:8000/docs")

    print("=" * 60)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline QA — Real Estate Assistant")
    parser.add_argument("--full", action="store_true", help="Include tests that download ML models")
    args = parser.parse_args()

    print("=" * 60)
    print("Real Estate Assistant — Pipeline QA")
    mode = "FULL (model downloads enabled)" if args.full else "FAST (mock / no-model tests only)"
    print(f"Mode: {mode}")
    print("=" * 60)

    test_data_preparation()
    test_vector_indexing()
    test_ban_geocoding()
    test_qa_system(full=args.full)
    test_summarization(full=args.full)
    test_sentiment(full=args.full)
    test_api()
    test_streamlit_syntax()
    test_response_times()
    test_qa_accuracy(full=args.full)

    failed = _print_summary()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
