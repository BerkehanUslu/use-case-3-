"""
DVF (Demandes de Valeurs Foncieres) Data Aggregation Pipeline.

This script aggregates real estate transaction data by commune, year,
and property type. When no real DVF data is available, it generates
realistic mock data for the 10 largest French cities.
"""

import os
import sys
import random
import numpy as np
import pandas as pd

# Mock data constants
MOCK_COMMUNES = [
    ("Paris", "75056"),
    ("Lyon", "69123"),
    ("Marseille", "13055"),
    ("Bordeaux", "33063"),
    ("Toulouse", "31555"),
    ("Nantes", "44109"),
    ("Strasbourg", "67482"),
    ("Montpellier", "34172"),
    ("Nice", "06088"),
    ("Rennes", "35238"),
]

TYPES_BIEN = ["Appartement", "Maison"]
ANNEES = [2020, 2021, 2022, 2023]

# Realistic base price per m2 by city (EUR/m2)
BASE_PRICES = {
    "Paris": 10500,
    "Lyon": 4800,
    "Marseille": 3200,
    "Bordeaux": 4500,
    "Toulouse": 3600,
    "Nantes": 3900,
    "Strasbourg": 3400,
    "Montpellier": 3300,
    "Nice": 5200,
    "Rennes": 3700,
}

# Maison prices are typically ~20% higher per m2 than appartements
TYPE_MULTIPLIER = {"Appartement": 1.0, "Maison": 1.2}

# Annual price growth factor (approximate market trend)
YEAR_GROWTH = {2020: 1.00, 2021: 1.06, 2022: 1.10, 2023: 1.08}


def _generate_mock_data(n_rows: int = 100) -> pd.DataFrame:
    """Generate realistic mock DVF rows (individual transactions)."""
    random.seed(42)
    np.random.seed(42)

    rows = []
    for _ in range(n_rows):
        commune, code = random.choice(MOCK_COMMUNES)
        type_bien = random.choice(TYPES_BIEN)
        annee = random.choice(ANNEES)

        base = BASE_PRICES[commune]
        multiplier = TYPE_MULTIPLIER[type_bien]
        growth = YEAR_GROWTH[annee]

        # Add Gaussian noise (~10%)
        prix_m2 = base * multiplier * growth * np.random.normal(1.0, 0.10)
        prix_m2 = max(prix_m2, 500)  # floor

        rows.append(
            {
                "commune": commune,
                "code_commune": code,
                "annee": annee,
                "prix_m2": round(prix_m2, 2),
                "type_bien": type_bien,
            }
        )

    return pd.DataFrame(rows)


def aggregate_dvf(input_path: str | None = None) -> pd.DataFrame:
    """
    Aggregate DVF transaction data by commune, year, and property type.

    Parameters
    ----------
    input_path : str | None
        Path to an existing CSV or Parquet file with raw DVF data.
        If None or the file does not exist, mock data is generated.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns:
        commune, code_commune, annee, prix_moyen_m2, nb_transactions,
        prix_median_m2, type_bien
    """
    if input_path is not None and os.path.exists(input_path):
        print(f"[INFO] Loading real DVF data from: {input_path}")
        if input_path.endswith(".parquet"):
            raw = pd.read_parquet(input_path)
        else:
            raw = pd.read_csv(input_path)

        # Minimal column normalisation — adapt to actual DVF schema as needed
        required = {"commune", "code_commune", "annee", "prix_m2", "type_bien"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(
                f"Input file is missing required columns: {missing}. "
                "Expected columns: commune, code_commune, annee, prix_m2, type_bien"
            )
    else:
        if input_path is not None:
            print(f"[WARN] File not found: {input_path}. Generating mock data.")
        else:
            print("[INFO] No input path provided. Generating mock DVF data.")
        raw = _generate_mock_data(n_rows=100)

    # Aggregate
    agg = (
        raw.groupby(["commune", "code_commune", "annee", "type_bien"], as_index=False)
        .agg(
            prix_moyen_m2=("prix_m2", "mean"),
            nb_transactions=("prix_m2", "count"),
            prix_median_m2=("prix_m2", "median"),
        )
    )

    # Round price columns for readability
    agg["prix_moyen_m2"] = agg["prix_moyen_m2"].round(2)
    agg["prix_median_m2"] = agg["prix_median_m2"].round(2)

    # Reorder columns
    agg = agg[
        [
            "commune",
            "code_commune",
            "annee",
            "prix_moyen_m2",
            "nb_transactions",
            "prix_median_m2",
            "type_bien",
        ]
    ]

    return agg.sort_values(["commune", "annee", "type_bien"]).reset_index(drop=True)


def _generate_sample(n: int = 50) -> pd.DataFrame:
    """Generate a sample subset of mock DVF transactions."""
    raw = _generate_mock_data(n_rows=n)
    return raw


def main():
    """Run the DVF data preparation pipeline."""
    print("=" * 60)
    print("DVF Data Preparation Pipeline")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)

    # --- Step 1: Aggregate DVF data ---
    print("\n[STEP 1] Aggregating DVF data...")
    df_agg = aggregate_dvf(input_path=None)  # Uses mock data by default

    parquet_path = "data/dvf_communes.parquet"
    df_agg.to_parquet(parquet_path, index=False)
    print(f"[OK] Saved aggregated data -> {parquet_path}")
    print(f"     Shape: {df_agg.shape}")

    # --- Step 2: Save sample CSV ---
    print("\n[STEP 2] Saving 50-row sample CSV...")
    df_sample = _generate_sample(n=50)
    sample_path = "data/sample_dvf.csv"
    df_sample.to_csv(sample_path, index=False)
    print(f"[OK] Saved sample data -> {sample_path}")
    print(f"     Shape: {df_sample.shape}")

    # --- Step 3: Print summary ---
    print("\n[SUMMARY]")
    print(f"  Communes:           {df_agg['commune'].nunique()}")
    print(f"  Years:              {sorted(df_agg['annee'].unique().tolist())}")
    print(f"  Property types:     {sorted(df_agg['type_bien'].unique().tolist())}")
    print(f"  Total agg rows:     {len(df_agg)}")
    print(f"  Avg price (€/m²):   {df_agg['prix_moyen_m2'].mean():.0f}")
    print(f"  Max price (€/m²):   {df_agg['prix_moyen_m2'].max():.0f}")
    print(f"  Min price (€/m²):   {df_agg['prix_moyen_m2'].min():.0f}")

    print("\nTop 5 communes by average price per m²:")
    top5 = (
        df_agg.groupby("commune")["prix_moyen_m2"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    for commune, prix in top5.items():
        print(f"  {commune:<15} {prix:>8.0f} €/m²")

    print("\n[DONE] Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
