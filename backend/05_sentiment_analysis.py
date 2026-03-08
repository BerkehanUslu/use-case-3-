"""
05_sentiment_analysis.py

Sentiment analysis for French real estate property descriptions.
Uses nlptown/bert-base-multilingual-uncased-sentiment (1-5 star ratings).
"""

import argparse
import sys
from typing import Optional

# Global cached pipeline
_sentiment_pipeline = None


def get_sentiment_pipeline():
    """Lazy-load and cache the sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
        )
    return _sentiment_pipeline


# Label mappings
STAR_TO_LABEL = {
    5: "Très positif",
    4: "Positif",
    3: "Neutre",
    2: "Négatif",
    1: "Très négatif",
}

STAR_TO_RECOMMENDATION_BASE = {
    5: "Forte recommandation d'achat",
    4: "Bien noté, investissement intéressant",
    3: "À étudier davantage",
    2: "Prudence recommandée",
    1: "Déconseillé",
}


def _parse_stars(raw_label: str) -> int:
    """Extract star count from model label like '4 stars' or '1 star'."""
    try:
        return int(raw_label.split()[0])
    except (ValueError, IndexError):
        return 3


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a property description.

    Returns:
        {
            "label": str,           # "Très positif", "Positif", "Neutre", "Négatif", "Très négatif"
            "score": float,         # confidence 0-1
            "stars": int,           # 1-5
            "raw_label": str,       # original model label e.g. "4 stars"
            "recommendation": str   # investment recommendation based on sentiment
        }
    """
    pipe = get_sentiment_pipeline()
    result = pipe(text, truncation=True, max_length=512)[0]

    raw_label = result["label"]
    score = float(result["score"])
    stars = _parse_stars(raw_label)
    label = STAR_TO_LABEL.get(stars, "Neutre")
    recommendation = get_investment_recommendation(stars)

    return {
        "label": label,
        "score": score,
        "stars": stars,
        "raw_label": raw_label,
        "recommendation": recommendation,
    }


def analyze_batch(texts: list[str]) -> list[dict]:
    """Batch sentiment analysis. Returns list of analyze_sentiment results."""
    pipe = get_sentiment_pipeline()
    raw_results = pipe(texts, truncation=True, max_length=512, batch_size=8)

    output = []
    for result in raw_results:
        raw_label = result["label"]
        score = float(result["score"])
        stars = _parse_stars(raw_label)
        label = STAR_TO_LABEL.get(stars, "Neutre")
        recommendation = get_investment_recommendation(stars)

        output.append({
            "label": label,
            "score": score,
            "stars": stars,
            "raw_label": raw_label,
            "recommendation": recommendation,
        })

    return output


def get_investment_recommendation(stars: int, prix_m2: Optional[float] = None) -> str:
    """
    Generate a simple investment recommendation based on sentiment stars and price.
    Returns a short French text recommendation.
    """
    base = STAR_TO_RECOMMENDATION_BASE.get(stars, "À étudier davantage")

    if prix_m2 is None:
        return base

    # Enrich recommendation with price context
    if stars >= 4:
        if prix_m2 < 5000:
            return f"{base} — prix attractif à {prix_m2:.0f} €/m²."
        elif prix_m2 < 10000:
            return f"{base} — prix dans la moyenne à {prix_m2:.0f} €/m²."
        else:
            return f"{base} — prix élevé à {prix_m2:.0f} €/m², vérifier la rentabilité."
    elif stars == 3:
        if prix_m2 < 5000:
            return f"{base} — le prix bas ({prix_m2:.0f} €/m²) peut compenser les incertitudes."
        else:
            return f"{base} — à {prix_m2:.0f} €/m², négocier avant d'investir."
    else:  # stars <= 2
        if prix_m2 < 3000:
            return f"{base} — prix très bas ({prix_m2:.0f} €/m²), travaux à prévoir."
        else:
            return f"{base} — à {prix_m2:.0f} €/m², risque trop élevé."


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_analyze_sentiment(text: str, index: int = 0) -> dict:
    """Return a deterministic fake sentiment without loading the model."""
    mock_data = [
        {
            "label": "Très positif",
            "score": 0.92,
            "stars": 5,
            "raw_label": "5 stars",
            "recommendation": "Forte recommandation d'achat",
        },
        {
            "label": "Neutre",
            "score": 0.71,
            "stars": 3,
            "raw_label": "3 stars",
            "recommendation": "À étudier davantage",
        },
        {
            "label": "Très négatif",
            "score": 0.85,
            "stars": 1,
            "raw_label": "1 star",
            "recommendation": "Déconseillé",
        },
    ]
    return mock_data[index % len(mock_data)]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_DESCRIPTIONS = [
    (
        "Positif",
        "Magnifique appartement lumineux, vue imprenable sur la Tour Eiffel, "
        "entièrement rénové avec des matériaux de qualité supérieure, "
        "cuisine équipée haut de gamme, parquet massif, double vitrage, "
        "ascenseur, gardien, cave et parking inclus. Quartier prisé, "
        "commerces et transports à deux pas.",
    ),
    (
        "Neutre",
        "Appartement fonctionnel en bon état, quartier calme avec toutes "
        "les commodités à proximité. Bien entretenu, quelques travaux de "
        "rafraîchissement souhaitables mais non urgents. Charges raisonnables.",
    ),
    (
        "Négatif",
        "Bien vétuste nécessitant de lourds travaux de rénovation complète, "
        "charges élevées, copropriété en difficulté financière, nuisances "
        "sonores importantes, humidité présente dans plusieurs pièces, "
        "DPE très mauvais (classe G), quartier peu attractif.",
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment analysis for French real estate descriptions."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without loading the model.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Analyse de sentiment — Descriptions immobilières")
    print("=" * 60)

    if args.mock:
        print("[MODE MOCK] Pas de chargement du modèle.\n")

    for idx, (expected_tone, description) in enumerate(DEMO_DESCRIPTIONS):
        print(f"\nDescription {idx + 1} (attendu : {expected_tone})")
        print(f"Texte : {description[:80]}...")

        if args.mock:
            result = _mock_analyze_sentiment(description, index=idx)
        else:
            result = analyze_sentiment(description)

        print(f"  Sentiment   : {result['label']} ({result['stars']} etoile(s))")
        print(f"  Confiance   : {result['score']:.2%}")
        print(f"  Label brut  : {result['raw_label']}")
        print(f"  Conseil     : {result['recommendation']}")

    print("\n" + "=" * 60)
    print("Recommandation enrichie avec prix/m2 :")
    for stars, prix in [(5, 4500.0), (3, 7200.0), (1, 2800.0)]:
        rec = get_investment_recommendation(stars, prix_m2=prix)
        label = STAR_TO_LABEL[stars]
        print(f"  {label} | {prix:.0f} euro/m2 -> {rec}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
