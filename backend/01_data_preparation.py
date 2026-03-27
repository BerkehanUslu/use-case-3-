"""
DVF (Demandes de Valeurs Foncieres) Data Aggregation Pipeline.

This script aggregates real estate transaction data by commune, year,
and property type from raw DVF pipe-delimited text files.
"""

import glob
import os
import pandas as pd


def aggregate_dvf(input_paths: list[str] | str | None = None) -> pd.DataFrame:
    """
    Aggregate DVF transaction data by commune, year, and property type.

    Parameters
    ----------
    input_paths : list[str] | str
        Path(s) to raw DVF pipe-delimited text file(s).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns:
        commune, code_commune, annee, prix_moyen_m2, nb_transactions,
        prix_median_m2, type_bien
    """
    if input_paths is None:
        raise ValueError("input_paths must be provided. No mock data fallback available.")

    paths = [input_paths] if isinstance(input_paths, str) else input_paths
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep="|")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # Validate actual DVF schema columns
    required = {"Commune", "Code commune", "Date mutation", "Valeur fonciere", "Surface reelle bati", "Type local"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}."
        )

    # Normalize to internal schema
    raw = raw.dropna(subset=["Valeur fonciere", "Surface reelle bati"])
    raw["Valeur fonciere"] = raw["Valeur fonciere"].astype(str).str.replace(",", ".").astype(float)

    raw = raw[(raw["Surface reelle bati"] > 0) & (raw["Valeur fonciere"] > 0)]

    raw["prix_m2"] = raw["Valeur fonciere"] / raw["Surface reelle bati"]
    raw["annee"] = pd.to_datetime(raw["Date mutation"], dayfirst=True).dt.year
    raw = raw.rename(columns={
        "Commune": "commune",
        "Code commune": "code_commune",
        "Type local": "type_bien",
    })

    # Aggregate
    agg = raw.groupby(
        ["commune", "code_commune", "annee", "type_bien"], as_index=False
    ).agg(
        prix_moyen_m2=("prix_m2", "mean"),
        nb_transactions=("prix_m2", "count"),
        prix_median_m2=("prix_m2", "median"),
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
    dvf_files = glob.glob("data/*.txt") + glob.glob("data/*.csv")
    if not dvf_files:
        print("[WARNING] No DVF files found in data/ (*.txt or *.csv). Skipping aggregation.")
        return
    print(f"  Found {len(dvf_files)} file(s): {', '.join(dvf_files)}")
    df_agg = aggregate_dvf(input_paths=dvf_files)

    parquet_path = "data/dvf_communes.parquet"
    df_agg.to_parquet(parquet_path, index=False)
    print(f"[OK] Saved aggregated data -> {parquet_path}")
    print(f"     Shape: {df_agg.shape}")

    # --- Step 2: Print summary ---
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
