"""
French Real Estate Q&A System using CamemBERT-based extractive QA.
Supports direct Q&A and RAG mode (retrieving context from commune metadata).
"""

import argparse
import json
import os
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Globals / caching
# ---------------------------------------------------------------------------

_qa_pipeline = None

META_PATH = os.path.join(os.path.dirname(__file__), "indexes", "real_estate_meta.json")

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_COMMUNES = [
    {
        "commune": "Paris",
        "code_postal": "75000",
        "prix_moyen_m2": 10500,
        "nb_transactions": 42000,
        "annee": 2023,
        "description": (
            "Paris est la capitale de la France. Le prix moyen au m² à Paris est de "
            "10 500 euros en 2023. Il y a eu 42 000 transactions immobilières à Paris "
            "en 2023. Paris reste la ville la plus chère de France en immobilier."
        ),
    },
    {
        "commune": "Lyon",
        "code_postal": "69000",
        "prix_moyen_m2": 5200,
        "nb_transactions": 18500,
        "annee": 2023,
        "description": (
            "Lyon est la deuxième ville de France par son dynamisme économique. "
            "Le prix moyen au m² à Lyon est de 5 200 euros en 2023. "
            "Il y a eu 18 500 transactions immobilières à Lyon en 2023."
        ),
    },
    {
        "commune": "Marseille",
        "code_postal": "13000",
        "prix_moyen_m2": 3800,
        "nb_transactions": 15000,
        "annee": 2023,
        "description": (
            "Marseille est la troisième plus grande ville de France. "
            "Le prix moyen au m² à Marseille est de 3 800 euros en 2023. "
            "Il y a eu 15 000 transactions immobilières à Marseille en 2023."
        ),
    },
]

MOCK_QA_RESPONSES = {
    "paris": {
        "answer": "10 500 euros",
        "score": 0.92,
    },
    "lyon": {
        "answer": "18 500 transactions immobilières",
        "score": 0.88,
    },
    "ville": {
        "answer": "Paris",
        "score": 0.85,
    },
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def get_qa_pipeline():
    """Lazy-load and return the QA pipeline (cached after first call)."""
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline

    try:
        from transformers import pipeline  # type: ignore

        print("Chargement du modèle CamemBERT QA (peut prendre quelques instants)...")
        _qa_pipeline = pipeline(
            "question-answering",
            model="etalab-ia/camembert-base-squadFR-fquad-piaf",
        )
        print("Modèle chargé avec succès.")
    except Exception as exc:  # pragma: no cover
        print(f"Impossible de charger le modèle: {exc}")
        _qa_pipeline = None

    return _qa_pipeline


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_commune_data() -> list[dict]:
    """Load commune metadata. Returns mock data if file not found."""
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            # Support {"communes": [...]} wrapper
            if isinstance(data, dict) and "communes" in data:
                return data["communes"]
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Erreur lors du chargement des métadonnées: {exc}. Utilisation des données mock.")

    return MOCK_COMMUNES


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def get_commune_context(commune_name: str) -> str:
    """Build a rich text context about a commune from the metadata JSON."""
    communes = load_commune_data()
    name_lower = commune_name.lower()

    matched = [
        c for c in communes if name_lower in c.get("commune", "").lower()
    ]

    if not matched:
        return ""

    parts: list[str] = []
    for c in matched:
        if "description" in c and c["description"]:
            parts.append(c["description"])
        else:
            # Build a synthetic description from structured fields
            lines = [f"Commune : {c.get('commune', 'Inconnue')}"]
            if "code_postal" in c:
                lines.append(f"Code postal : {c['code_postal']}")
            if "prix_moyen_m2" in c:
                lines.append(
                    f"Le prix moyen au m² est de {c['prix_moyen_m2']} euros en {c.get('annee', 'N/A')}."
                )
            if "nb_transactions" in c:
                lines.append(
                    f"Nombre de transactions immobilières en {c.get('annee', 'N/A')} : {c['nb_transactions']}."
                )
            parts.append(" ".join(lines))

    return " ".join(parts)


def _build_rag_context(question: str) -> str:
    """Find relevant passages from commune data based on question keywords."""
    communes = load_commune_data()
    question_lower = question.lower()

    # Remove common French stop words before matching
    stop_words = {
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
        "au", "aux", "est", "sont", "il", "elle", "ils", "elles",
        "quel", "quelle", "quels", "quelles", "combien", "avec", "pour",
        "que", "qui", "dans", "sur", "par", "plus", "prix", "moyen",
        "transactions", "immobilières", "immobilier",
    }
    words = re.findall(r"\b\w+\b", question_lower)
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    scored: list[tuple[int, dict]] = []
    for c in communes:
        commune_text = " ".join(str(v) for v in c.values()).lower()
        score = sum(1 for kw in keywords if kw in commune_text)
        if score > 0:
            scored.append((score, c))

    # Sort by relevance (descending) and take top 3
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]

    if not top:
        # Fallback: return all descriptions
        top = [(0, c) for c in communes]

    parts: list[str] = []
    for _, c in top:
        if "description" in c and c["description"]:
            parts.append(c["description"])
        else:
            lines = []
            if "commune" in c:
                lines.append(f"Commune : {c['commune']}.")
            if "prix_moyen_m2" in c:
                lines.append(
                    f"Prix moyen au m² : {c['prix_moyen_m2']} euros ({c.get('annee', '')})."
                )
            if "nb_transactions" in c:
                lines.append(
                    f"Transactions en {c.get('annee', '')} : {c['nb_transactions']}."
                )
            parts.append(" ".join(lines))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Core Q&A function
# ---------------------------------------------------------------------------


def answer_question(question: str, context: Optional[str] = None) -> dict:
    """
    Answer a question.
    - If context provided: use it directly
    - If not: build context from commune data matching the question keywords

    Returns: {"answer": str, "score": float, "context_used": str, "mode": str}
    """
    if context:
        context_used = context
        mode = "direct"
    else:
        context_used = _build_rag_context(question)
        mode = "rag"

    pipe = get_qa_pipeline()

    if pipe is None:
        return {
            "answer": "[Modèle non disponible]",
            "score": 0.0,
            "context_used": context_used,
            "mode": mode,
        }

    try:
        result = pipe(question=question, context=context_used)
        return {
            "answer": result["answer"],
            "score": float(result["score"]),
            "context_used": context_used,
            "mode": mode,
        }
    except Exception as exc:
        return {
            "answer": f"[Erreur lors de la réponse: {exc}]",
            "score": 0.0,
            "context_used": context_used,
            "mode": mode,
        }


# ---------------------------------------------------------------------------
# Mock answer (for --mock flag)
# ---------------------------------------------------------------------------


def _mock_answer(question: str, context: Optional[str] = None) -> dict:
    """Return a deterministic mock answer without downloading any model."""
    question_lower = question.lower()
    context_used = context or _build_rag_context(question)
    mode = "direct" if context else "rag"

    for key, resp in MOCK_QA_RESPONSES.items():
        if key in question_lower:
            return {
                "answer": resp["answer"],
                "score": resp["score"],
                "context_used": context_used,
                "mode": f"{mode} (mock)",
            }

    return {
        "answer": "Information non disponible dans les données mock.",
        "score": 0.0,
        "context_used": context_used,
        "mode": f"{mode} (mock)",
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _print_result(idx: int, question: str, result: dict) -> None:
    separator = "-" * 60
    print(separator)
    print(f"Question {idx}: {question}")
    print(f"  Mode      : {result['mode']}")
    print(f"  Réponse   : {result['answer']}")
    print(f"  Score     : {result['score']:.4f}")
    ctx_preview = result["context_used"][:120].replace("\n", " ")
    print(f"  Contexte  : {ctx_preview}...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Système de Q&R immobilier français")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Utiliser des réponses mock sans télécharger le modèle",
    )
    args = parser.parse_args()

    answer_fn = _mock_answer if args.mock else answer_question

    examples = [
        ("Quel est le prix moyen au m² à Paris ?", None),
        ("Combien de transactions immobilières à Lyon en 2023 ?", None),
        (
            "Quelle est la ville avec le prix le plus élevé ?",
            (
                "Paris est la ville la plus chère de France avec un prix moyen au m² de "
                "10 500 euros en 2023. Lyon affiche 5 200 euros au m² et Marseille 3 800 euros."
            ),
        ),
    ]

    print("=" * 60)
    print("  Système de Q&R Immobilier Français (CamemBERT)")
    print("=" * 60)
    if args.mock:
        print("  Mode : MOCK (aucun téléchargement de modèle)")
    print()

    for idx, (question, context) in enumerate(examples, start=1):
        result = answer_fn(question, context)
        _print_result(idx, question, result)


if __name__ == "__main__":
    main()
