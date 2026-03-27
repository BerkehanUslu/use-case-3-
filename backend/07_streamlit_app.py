"""
Assistant Immobilier IA — CY Tech
Streamlit Interface with 5 tabs:
1. Prix par Commune
2. Q&A Immobilier
3. Résumé de Rapport
4. Analyse de Bien
5. Carte Interactive

Works in two modes:
- API mode: calls http://localhost:8000 endpoints
- Demo mode: uses inline mock data if API is unavailable
"""

import streamlit as st
import httpx
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant Immobilier IA — CY Tech",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Mock Data ──────────────────────────────────────────────────────────────────
MOCK_COMMUNES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522, "prix_moyen_m2": 10500, "prix_median_m2": 9800, "nb_transactions": 45230, "code": "75056"},
    "Lyon": {"lat": 45.7640, "lon": 4.8357, "prix_moyen_m2": 5200, "prix_median_m2": 4900, "nb_transactions": 18750, "code": "69123"},
    "Marseille": {"lat": 43.2965, "lon": 5.3698, "prix_moyen_m2": 3800, "prix_median_m2": 3500, "nb_transactions": 15200, "code": "13055"},
    "Bordeaux": {"lat": 44.8378, "lon": -0.5792, "prix_moyen_m2": 4900, "prix_median_m2": 4600, "nb_transactions": 12400, "code": "33063"},
    "Toulouse": {"lat": 43.6047, "lon": 1.4442, "prix_moyen_m2": 4100, "prix_median_m2": 3900, "nb_transactions": 14800, "code": "31555"},
    "Nantes": {"lat": 47.2184, "lon": -1.5536, "prix_moyen_m2": 4300, "prix_median_m2": 4100, "nb_transactions": 11200, "code": "44109"},
    "Strasbourg": {"lat": 48.5734, "lon": 7.7521, "prix_moyen_m2": 3600, "prix_median_m2": 3400, "nb_transactions": 9800, "code": "67482"},
    "Montpellier": {"lat": 43.6110, "lon": 3.8767, "prix_moyen_m2": 3900, "prix_median_m2": 3700, "nb_transactions": 10500, "code": "34172"},
    "Nice": {"lat": 43.7102, "lon": 7.2620, "prix_moyen_m2": 5800, "prix_median_m2": 5400, "nb_transactions": 11800, "code": "06088"},
    "Rennes": {"lat": 48.1173, "lon": -1.6778, "prix_moyen_m2": 3700, "prix_median_m2": 3500, "nb_transactions": 8900, "code": "35238"},
}

# ─── API Availability Check ──────────────────────────────────────────────────────
def check_api_availability(api_url: str) -> bool:
    """Check if the API is available by hitting /health endpoint."""
    try:
        response = httpx.get(f"{api_url}/health", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 CY Tech")
    st.markdown("**Assistant Immobilier IA**")
    st.markdown("_Analyse intelligente du marché immobilier français_")
    st.divider()

    api_url = st.text_input(
        "URL de l'API",
        value=(
            os.getenv("API_URL")
            or f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"
        ),
        help="Adresse du backend FastAPI"
    )

    # Check API at startup (cached in session state)
    if "api_available" not in st.session_state:
        st.session_state.api_available = check_api_availability(api_url)

    if st.button("🔄 Vérifier la connexion"):
        st.session_state.api_available = check_api_availability(api_url)

    if st.session_state.api_available:
        st.success("🟢 Connected")
        st.caption(f"API disponible sur {api_url}")
    else:
        st.warning("🔴 Demo mode")
        st.caption("API non disponible — données de démonstration utilisées")

    st.divider()
    st.markdown("### À propos")
    st.markdown(
        "Cette application exploite des modèles de ML pour analyser "
        "le marché immobilier français (DVF, PLU, etc.)."
    )
    st.caption("CY Tech — Projet IA Immobilier 2024")

API_AVAILABLE = st.session_state.api_available
BASE_DIR = Path(__file__).resolve().parent
META_PATH = BASE_DIR / "indexes" / "real_estate_meta.json"

# ─── Helper functions ───────────────────────────────────────────────────────────
def api_get(endpoint: str, params: dict = None):
    """Perform a GET request to the API."""
    try:
        r = httpx.get(f"{api_url}{endpoint}", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None

def api_post(endpoint: str, payload: dict = None, files=None):
    """Perform a POST request to the API."""
    try:
        if files:
            r = httpx.post(f"{api_url}{endpoint}", files=files, timeout=30.0)
        else:
            r = httpx.post(f"{api_url}{endpoint}", json=payload, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None

def _aggregate_reference_map_data(records: list[dict]) -> dict:
    """Build a fast map dataset using real prices with reference coordinates.

    This avoids waiting for nationwide BAN geocoding: we keep curated lat/lon for
    the main demo cities and replace only the market statistics from real metadata.
    """
    aggregated: dict = {}
    for city, ref in MOCK_COMMUNES.items():
        city_upper = city.upper()
        matches = [r for r in records if str(r.get("commune", "")).upper() == city_upper]
        if not matches:
            prefix = city_upper + " "
            matches = [r for r in records if str(r.get("commune", "")).upper().startswith(prefix)]
        if not matches:
            continue

        total_tx = sum(int(r.get("nb_transactions", 0) or 0) for r in matches)
        weight = total_tx if total_tx > 0 else len(matches)

        def _weighted(field: str, fallback: float = 0.0) -> float:
            numerator = 0.0
            for row in matches:
                row_tx = int(row.get("nb_transactions", 0) or 0)
                row_weight = row_tx if total_tx > 0 else 1
                numerator += float(row.get(field, fallback) or fallback) * row_weight
            return round(numerator / weight, 0)

        aggregated[city] = {
            "lat": ref["lat"],
            "lon": ref["lon"],
            "prix_moyen_m2": _weighted("prix_moyen_m2", ref["prix_moyen_m2"]),
            "prix_median_m2": _weighted("prix_median_m2", ref["prix_median_m2"]),
            "nb_transactions": total_tx,
            "code": ref["code"],
        }
    return aggregated


def _load_map_data() -> tuple[dict, str]:
    """Load commune map data from metadata if available.

    Returns (data, source_label), where data is a dict of commune →
    {lat, lon, prix_moyen_m2, prix_median_m2, nb_transactions, code}.
    """
    try:
        with open(META_PATH, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        # Deduplicate: keep one entry per commune (highest prix_moyen_m2 row)
        communes: dict = {}
        for r in records:
            name = r.get("commune", "")
            lat = r.get("lat")
            lon = r.get("lon")
            if not name or lat is None or lon is None:
                continue
            prev = communes.get(name)
            if prev is None or r.get("prix_moyen_m2", 0) > prev.get("prix_moyen_m2", 0):
                communes[name] = {
                    "lat": lat,
                    "lon": lon,
                    "prix_moyen_m2": r.get("prix_moyen_m2", 0),
                    "prix_median_m2": r.get("prix_median_m2", r.get("prix_moyen_m2", 0)),
                    "nb_transactions": r.get("nb_transactions", 0),
                    "code": r.get("code", ""),
                }

        if communes:
            return communes, "index FAISS"

        reference_data = _aggregate_reference_map_data(records)
        if reference_data:
            return reference_data, "index réel + coordonnées de référence"
    except Exception:
        pass

    # Fallback to hardcoded mock data when index is not built
    return MOCK_COMMUNES, "données mock"


def _load_trends_data(commune: str) -> list[dict]:
    """Load year-over-year trend data for a commune directly from the metadata index.

    Returns a list of {annee, type_bien, prix_moyen_m2, prix_median_m2, nb_transactions},
    sorted by (annee, type_bien). Empty list if index not available or commune not found.
    """
    try:
        with open(META_PATH, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        commune_lower = commune.lower()
        # Exact match first, then prefix+space to avoid false matches (e.g. PARISOT vs PARIS 01)
        matches = [r for r in records if r.get("commune", "").lower() == commune_lower]
        if not matches:
            prefix = commune_lower + " "
            matches = [r for r in records if r.get("commune", "").lower().startswith(prefix)]

        # Group by (annee, type_bien)
        from collections import defaultdict
        buckets: dict = defaultdict(list)
        for r in matches:
            key = (r.get("annee"), r.get("type_bien", "Tous"))
            buckets[key].append(r)

        trends = []
        for (annee, type_bien), rows in sorted(buckets.items()):
            avg_prix = sum(r.get("prix_moyen_m2", 0) for r in rows) / len(rows)
            avg_med = sum(r.get("prix_median_m2", r.get("prix_moyen_m2", 0)) for r in rows) / len(rows)
            total_tx = sum(r.get("nb_transactions", 0) for r in rows)
            trends.append({
                "annee": annee,
                "type_bien": type_bien,
                "prix_moyen_m2": round(avg_prix, 0),
                "prix_median_m2": round(avg_med, 0),
                "nb_transactions": total_tx,
            })
        return trends
    except Exception:
        return []


def price_color(prix: float) -> str:
    """Return color based on price per m²."""
    if prix >= 7000:
        return "#d62728"   # red — expensive
    elif prix >= 4500:
        return "#ff7f0e"   # orange — moderate-high
    elif prix >= 3500:
        return "#2ca02c"   # green — reasonable
    else:
        return "#1f77b4"   # blue — affordable

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏙️ Prix par Commune",
    "💬 Q&A Immobilier",
    "📄 Résumé de Rapport",
    "🔍 Analyse de Bien",
    "🗺️ Carte Interactive",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prix par Commune
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏙️ Prix par Commune")
    st.markdown("Recherchez les prix immobiliers d'une commune française.")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        commune_query = st.text_input(
            "Nom de la commune",
            placeholder="Ex : Paris, Lyon, Bordeaux…",
            label_visibility="collapsed"
        )
    with col_btn:
        search_btn = st.button("🔍 Rechercher", width="stretch")

    if search_btn and commune_query:
        commune_key = commune_query.strip().title()

        if API_AVAILABLE:
            data = api_get(f"/communes/{commune_query.strip()}")
        else:
            data = None
            for key, val in MOCK_COMMUNES.items():
                if key.lower() == commune_key.lower():
                    data = {
                        "commune": key,
                        "prix_moyen_m2": val["prix_moyen_m2"],
                        "prix_median_m2": val["prix_median_m2"],
                        "nb_transactions": val["nb_transactions"],
                    }
                    break
            if data is None:
                st.info(f"Commune « {commune_key} » non trouvée dans les données de démo. Essayez : Paris, Lyon, Marseille, etc.")

        if data:
            # Normalise field names: API returns avg_price_m2/transactions,
            # mock data uses prix_moyen_m2/prix_median_m2/nb_transactions.
            prix_moyen = data.get("prix_moyen_m2") or data.get("avg_price_m2", 0)
            prix_median = data.get("prix_median_m2") or data.get("avg_price_m2", 0)
            nb_trans = data.get("nb_transactions") or data.get("transactions", 0)
            prix_reference = prix_median or prix_moyen

            st.markdown("---")
            st.subheader(f"Résultats pour **{data.get('commune', commune_key)}**")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Prix de référence (médian €/m²)", f"{prix_reference:,.0f} €")
            with c2:
                st.metric("Prix moyen (€/m²)", f"{prix_moyen:,.0f} €")
            with c3:
                st.metric("Nb transactions", f"{nb_trans:,}")

            if prix_median and prix_moyen and prix_moyen > prix_median * 1.35:
                st.caption(
                    "Le prix moyen est sensible aux valeurs extrêmes, surtout pour les "
                    "locaux commerciaux. Le prix médian est affiché comme référence."
                )

            # Bar chart — commune vs top 5 from mock using median to reduce outlier distortion
            st.markdown("#### Comparaison avec d'autres communes")
            top5 = sorted(MOCK_COMMUNES.items(), key=lambda x: x[1]["prix_median_m2"], reverse=True)[:5]
            chart_data = {c: v["prix_median_m2"] for c, v in top5}
            # Add searched commune if not already there
            if commune_key not in chart_data:
                chart_data[commune_key] = prix_reference

            communes_list = list(chart_data.keys())
            prices_list = list(chart_data.values())
            colors_list = [
                "#e63946" if c == commune_key else "#457b9d"
                for c in communes_list
            ]

            fig = go.Figure(
                go.Bar(
                    x=communes_list,
                    y=prices_list,
                    marker_color=colors_list,
                    text=[f"{p:,.0f} €" for p in prices_list],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Prix médian au m² (€)",
                xaxis_title="Commune",
                yaxis_title="€/m²",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

            # Time-series trend chart
            st.markdown("#### Évolution des prix par année")
            if API_AVAILABLE:
                trend_data = api_get(f"/trends/{commune_query.strip()}")
                trend_rows = trend_data.get("trends", []) if trend_data else []
            else:
                # Read directly from index when API is off
                trend_rows = _load_trends_data(commune_key)

            if trend_rows:
                # Group rows by type_bien for separate lines
                from collections import defaultdict
                by_type: dict = defaultdict(list)
                for row in trend_rows:
                    by_type[row["type_bien"]].append(row)

                fig_trend = go.Figure()
                for type_bien, rows in sorted(by_type.items()):
                    rows_sorted = sorted(rows, key=lambda r: r["annee"])
                    fig_trend.add_trace(go.Scatter(
                        x=[str(r["annee"]) for r in rows_sorted],
                        y=[
                            r.get("prix_median_m2", r.get("prix_moyen_m2", 0))
                            for r in rows_sorted
                        ],
                        mode="lines+markers",
                        name=type_bien,
                        hovertemplate=(
                            "%{x}<br>%{y:,.0f} €/m²<extra>" + type_bien + "</extra>"
                        ),
                    ))

                fig_trend.update_layout(
                    title=f"Tendance des prix médians — {data.get('commune', commune_key)}",
                    xaxis_title="Année",
                    yaxis_title="Prix médian (€/m²)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    legend_title="Type de bien",
                    xaxis={"type": "category"},
                )
                st.plotly_chart(fig_trend, width="stretch")
            else:
                st.info(
                    "Données de tendance non disponibles — construisez l'index d'abord "
                    "(`python 02_vector_indexing.py`)."
                )

    elif search_btn:
        st.warning("Veuillez saisir un nom de commune.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Q&A Immobilier
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("💬 Q&A Immobilier")
    st.markdown("Posez une question sur l'immobilier en France.")

    question = st.text_area(
        "Votre question",
        placeholder="Ex : Quelle est la tendance des prix à Bordeaux ces 5 dernières années ?",
        height=120,
    )

    with st.expander("📎 Contexte supplémentaire (optionnel)"):
        context_text = st.text_area(
            "Contexte",
            placeholder="Collez ici un extrait de rapport, d'annonce ou tout contexte utile…",
            height=150,
        )

    qa_btn = st.button("📨 Soumettre la question", width="content")

    if qa_btn:
        if not question.strip():
            st.warning("Veuillez saisir une question.")
        else:
            with st.spinner("Recherche de la réponse…"):
                if API_AVAILABLE:
                    payload = {"question": question}
                    if context_text.strip():
                        payload["context"] = context_text
                    result = api_post("/qa", payload)
                else:
                    # Demo mock answer
                    result = {
                        "answer": (
                            "En mode démonstration : les prix immobiliers en France ont connu "
                            "une hausse significative dans les grandes métropoles sur les 5 dernières années, "
                            "avec des disparités régionales importantes. Paris reste la ville la plus chère "
                            "avec environ 10 500 €/m², suivie de Nice (5 800 €/m²) et Lyon (5 200 €/m²)."
                        ),
                        "confidence": 0.82,
                        "context_used": [
                            "Base de données DVF 2018-2023",
                            "Indices INSEE des prix immobiliers",
                        ],
                    }

            if result:
                st.markdown("---")
                st.markdown("#### Réponse")
                st.markdown(f"**{result.get('answer', 'Aucune réponse disponible.')}**")

                confidence = result.get("score") or result.get("confidence", 0.0)
                st.markdown(f"**Score de confiance : {confidence:.0%}**")
                st.progress(float(confidence))

                ctx_used = result.get("context_used", [])
                if ctx_used:
                    with st.expander("📚 Sources / contexte utilisé"):
                        if isinstance(ctx_used, list):
                            for src in ctx_used:
                                st.markdown(f"- {src}")
                        else:
                            st.markdown(str(ctx_used))

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Résumé de Rapport
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📄 Résumé de Rapport")
    st.markdown("Générez un résumé d'un rapport immobilier (texte ou PDF).")

    inner_tab_text, inner_tab_pdf = st.tabs(["📝 Texte", "📎 PDF"])

    # ── Texte sub-tab ──
    with inner_tab_text:
        report_text = st.text_area(
            "Texte du rapport",
            placeholder="Collez ici le contenu du rapport à résumer…",
            height=250,
        )
        max_length = st.slider(
            "Longueur maximale du résumé (mots)",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
        )
        sum_btn_text = st.button("✂️ Générer le résumé (texte)", width="content")

        if sum_btn_text:
            if not report_text.strip():
                st.warning("Veuillez saisir un texte à résumer.")
            else:
                with st.spinner("Génération du résumé…"):
                    if API_AVAILABLE:
                        result = api_post("/summarize", {
                            "text": report_text,
                            "max_length": max_length,
                        })
                    else:
                        # Simple demo truncation
                        words = report_text.split()
                        summary_words = words[:max_length]
                        summary = " ".join(summary_words)
                        if len(words) > max_length:
                            summary += "…"
                        result = {
                            "summary": summary,
                            "original_length": len(words),
                            "summary_length": len(summary_words),
                        }

                if result:
                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div style="background:#f0f4f8;padding:1.2rem;border-radius:8px;border-left:4px solid #457b9d;">
                        <b>Résumé :</b><br><br>{result.get('summary', '')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                    mc1, mc2, mc3 = st.columns(3)
                    orig = result.get("original_length", len(report_text.split()))
                    summ = result.get("summary_length", len(result.get("summary", "").split()))
                    ratio = round(summ / orig * 100, 1) if orig else 0
                    mc1.metric("Longueur originale (mots)", f"{orig:,}")
                    mc2.metric("Longueur résumé (mots)", f"{summ:,}")
                    mc3.metric("Taux de compression", f"{ratio}%")

    # ── PDF sub-tab ──
    with inner_tab_pdf:
        uploaded_pdf = st.file_uploader(
            "Téléversez un fichier PDF",
            type=["pdf"],
            help="Formats acceptés : PDF uniquement"
        )
        max_length_pdf = st.slider(
            "Longueur maximale du résumé PDF (mots)",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            key="pdf_max_length",
        )
        sum_btn_pdf = st.button("✂️ Générer le résumé (PDF)", width="content")

        if sum_btn_pdf:
            if uploaded_pdf is None:
                st.warning("Veuillez téléverser un fichier PDF.")
            else:
                pdf_bytes = uploaded_pdf.read()
                with st.spinner("Analyse du PDF en cours…"):
                    if API_AVAILABLE:
                        result = api_post(
                            "/summarize-pdf",
                            files={"file": (uploaded_pdf.name, pdf_bytes, "application/pdf")},
                        )
                    else:
                        # Demo: can't truly parse PDF without extra lib — show placeholder
                        pdf_size_kb = len(pdf_bytes) / 1024
                        placeholder_summary = (
                            f"[Mode démonstration] Le document PDF « {uploaded_pdf.name} » "
                            f"({pdf_size_kb:.1f} Ko) a été reçu. "
                            "En mode production, un modèle NLP extrairait et résumerait son contenu. "
                            "Les thèmes principaux seraient : marché immobilier, tendances des prix, "
                            "analyse des transactions, recommandations d'investissement."
                        )
                        result = {
                            "summary": placeholder_summary,
                            "original_length": int(pdf_size_kb * 10),
                            "summary_length": len(placeholder_summary.split()),
                        }

                if result:
                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div style="background:#f0f4f8;padding:1.2rem;border-radius:8px;border-left:4px solid #457b9d;">
                        <b>Résumé :</b><br><br>{result.get('summary', '')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                    mc1, mc2, mc3 = st.columns(3)
                    orig = result.get("original_length", 0)
                    summ = result.get("summary_length", 0)
                    ratio = round(summ / orig * 100, 1) if orig else 0
                    mc1.metric("Longueur originale (mots est.)", f"{orig:,}")
                    mc2.metric("Longueur résumé (mots)", f"{summ:,}")
                    mc3.metric("Taux de compression", f"{ratio}%")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analyse de Bien
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔍 Analyse de Bien Immobilier")
    st.markdown("Analysez une description de bien et obtenez une recommandation d'investissement.")

    property_desc = st.text_area(
        "Description du bien",
        placeholder=(
            "Ex : Appartement T3 de 72m² en plein centre-ville de Lyon, "
            "lumineux, rénové, proche transports. Idéal investissement locatif. "
            "Charges faibles, copropriété bien entretenue…"
        ),
        height=200,
    )

    analyze_btn = st.button("🧠 Analyser le bien", width="content")

    SENTIMENT_EMOJI = {
        "très positif": "⭐⭐⭐⭐⭐",
        "positif": "⭐⭐⭐⭐",
        "neutre": "⭐⭐⭐",
        "négatif": "⭐⭐",
        "très négatif": "⭐",
    }

    SENTIMENT_COLOR = {
        "très positif": "#2ca02c",
        "positif": "#5cb85c",
        "neutre": "#f0ad4e",
        "négatif": "#d9534f",
        "très négatif": "#c0392b",
    }

    RECOMMENDATION_MAP = {
        "très positif": "✅ Investissement fortement recommandé",
        "positif": "✅ Bon potentiel d'investissement",
        "neutre": "⚠️ À étudier selon le prix de marché",
        "négatif": "❌ Prudence recommandée",
        "très négatif": "❌ Déconseillé — risques importants",
    }

    if analyze_btn:
        if not property_desc.strip():
            st.warning("Veuillez saisir une description de bien.")
        else:
            with st.spinner("Analyse en cours…"):
                if API_AVAILABLE:
                    result = api_post("/sentiment", {"text": property_desc})
                else:
                    # Demo: simple heuristic based on positive/negative keywords
                    positive_words = [
                        "lumineux", "rénové", "centre", "excellent", "idéal",
                        "proche", "calme", "moderne", "spacieux", "investissement",
                        "locatif", "qualité", "charme", "prestations", "atout",
                    ]
                    negative_words = [
                        "travaux", "urgent", "humidité", "bruit", "vétuste",
                        "dégradé", "problème", "risque", "ancien", "isolation",
                    ]
                    desc_lower = property_desc.lower()
                    pos_count = sum(1 for w in positive_words if w in desc_lower)
                    neg_count = sum(1 for w in negative_words if w in desc_lower)
                    score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
                    score = max(-1.0, min(1.0, score))

                    if score >= 0.5:
                        label = "très positif"
                    elif score >= 0.1:
                        label = "positif"
                    elif score >= -0.1:
                        label = "neutre"
                    elif score >= -0.5:
                        label = "négatif"
                    else:
                        label = "très négatif"

                    result = {
                        "label": label,
                        "score": round((score + 1) / 2, 2),  # normalize 0-1
                        "raw_score": round(score, 3),
                    }

            if result:
                label = result.get("label", "neutre").lower()
                score = float(result.get("score", 0.5))
                color = SENTIMENT_COLOR.get(label, "#f0ad4e")
                emoji_stars = SENTIMENT_EMOJI.get(label, "⭐⭐⭐")
                recommendation = RECOMMENDATION_MAP.get(label, "À étudier")

                st.markdown("---")
                st.subheader("Résultat de l'analyse")

                col_sent, col_rec = st.columns(2)

                with col_sent:
                    st.markdown(
                        f"""
                        <div style="background:{color}22;border:2px solid {color};
                        border-radius:10px;padding:1rem;text-align:center;">
                        <div style="font-size:2rem;">{emoji_stars}</div>
                        <div style="font-size:1.3rem;font-weight:bold;color:{color};">
                        {label.capitalize()}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_rec:
                    st.markdown(
                        f"""
                        <div style="background:#f8f9fa;border-radius:10px;padding:1rem;">
                        <b>Recommandation :</b><br>
                        <span style="font-size:1.1rem;">{recommendation}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("")
                st.markdown("**Score de sentiment (jauge) :**")

                # Gauge chart using plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 30], "color": "#ffcccc"},
                            {"range": [30, 50], "color": "#fff3cd"},
                            {"range": [50, 70], "color": "#d4edda"},
                            {"range": [70, 100], "color": "#c3e6cb"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": score * 100,
                        },
                    },
                    title={"text": "Score de positivité"},
                ))
                fig_gauge.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
                st.plotly_chart(fig_gauge, width="stretch")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Carte Interactive
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🗺️ Carte Interactive")
    st.markdown("Visualisez les prix immobiliers des principales communes françaises.")

    # Try folium first, fallback to st.map
    try:
        import folium
        from streamlit_folium import st_folium

        map_communes, source_label = _load_map_data()
        st.caption(f"Source : {source_label} — {len(map_communes)} communes")

        # Color scale: green (cheap) → red (expensive)
        prices = [v["prix_moyen_m2"] for v in map_communes.values()]
        min_price = min(prices)
        max_price = max(prices)

        def price_to_color(prix: float) -> str:
            """Map price to a color between green and red."""
            ratio = (prix - min_price) / (max_price - min_price)
            r = int(255 * ratio)
            g = int(255 * (1 - ratio))
            return f"#{r:02x}{g:02x}00"

        def price_to_radius(prix: float) -> float:
            """Map price to circle radius (15-50)."""
            ratio = (prix - min_price) / (max_price - min_price)
            return 15 + ratio * 35

        m = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="CartoDB positron")

        for commune, data in map_communes.items():
            color = price_to_color(data["prix_moyen_m2"])
            radius = price_to_radius(data["prix_moyen_m2"])
            tooltip = (
                f"<b>{commune}</b><br>"
                f"Prix moyen : {data['prix_moyen_m2']:,.0f} €/m²<br>"
                f"Prix médian : {data['prix_median_m2']:,.0f} €/m²<br>"
                f"Transactions : {data['nb_transactions']:,}"
            )
            folium.CircleMarker(
                location=[data["lat"], data["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=folium.Tooltip(tooltip),
                popup=folium.Popup(tooltip, max_width=250),
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
             background:white;padding:10px;border-radius:8px;
             border:1px solid #ccc;font-size:13px;">
        <b>Légende — Prix €/m²</b><br>
        <span style="color:#00ff00;">●</span> Bas (~3 500 €)<br>
        <span style="color:#ffaa00;">●</span> Moyen (~5 000 €)<br>
        <span style="color:#ff0000;">●</span> Élevé (~10 500 €)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width="100%", height=550, returned_objects=[])

    except ImportError:
        st.info(
            "Les librairies `folium` et `streamlit-folium` ne sont pas installées. "
            "Affichage de secours avec `st.map`."
        )
        import pandas as pd

        map_communes, _ = _load_map_data()
        map_df = pd.DataFrame([
            {
                "lat": v["lat"],
                "lon": v["lon"],
                "commune": k,
                "prix_moyen_m2": v["prix_moyen_m2"],
            }
            for k, v in map_communes.items()
        ])
        st.map(map_df[["lat", "lon"]], zoom=5)

        # Show table alongside
        st.markdown("#### Données par commune")
        display_df = map_df[["commune", "prix_moyen_m2"]].copy()
        display_df.columns = ["Commune", "Prix moyen (€/m²)"]
        display_df = display_df.sort_values("Prix moyen (€/m²)", ascending=False)
        st.dataframe(display_df, width="stretch", hide_index=True)

    # Price comparison bar chart at the bottom
    st.markdown("---")
    st.markdown("#### Comparatif des prix par commune")
    _bar_communes, _ = _load_map_data()
    sorted_communes = sorted(_bar_communes.items(), key=lambda x: x[1]["prix_moyen_m2"], reverse=True)
    commune_names = [c for c, _ in sorted_communes]
    commune_prices = [v["prix_moyen_m2"] for _, v in sorted_communes]
    bar_colors = [price_to_color(p) if "price_to_color" in dir() else "#457b9d" for p in commune_prices]

    fig_bar = go.Figure(
        go.Bar(
            x=commune_names,
            y=commune_prices,
            marker_color=bar_colors,
            text=[f"{p:,.0f} €" for p in commune_prices],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        xaxis_title="Commune",
        yaxis_title="Prix moyen €/m²",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    st.plotly_chart(fig_bar, width="stretch")
