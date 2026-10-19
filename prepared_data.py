"""
prepare_data.py

Reproducible preprocessing for Assignment 3:
Topic 1 — Compute and algorithms.

Principles:
1. Keep model-scaling data and benchmark data as separate analytical tables.
2. Use ONLY measured benchmark observations from Epoch AI.
3. Never estimate/fill missing benchmark scores from compute, year, parameters, or model name.
4. Keep missing benchmark/compute values as NaN and filter them only for charts that require them.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = "data"

RAW_MODELS_PATH = os.path.join(DATA_DIR, "raw_ai_models.csv")
RAW_BENCHMARKS_PATH = os.path.join(DATA_DIR, "raw_benchmarks.csv")

PROCESSED_MODELS_PATH = os.path.join(DATA_DIR, "ai_models.csv")
PROCESSED_BENCHMARKS_PATH = os.path.join(DATA_DIR, "benchmarks.csv")
PROCESSED_GPQA_PATH = os.path.join(DATA_DIR, "gpqa.csv")

# The current Epoch Capabilities & Benchmarking export contains GPQA Diamond
# but does NOT contain MMLU. GPQA is therefore used as the primary measured
# capability benchmark for the current app.
PRIMARY_BENCHMARK = "GPQA diamond"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_source_files():
    """Fail clearly if the raw source samples are not present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    missing = [
        path
        for path in [RAW_MODELS_PATH, RAW_BENCHMARKS_PATH]
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required source file(s):\n- "
            + "\n- ".join(missing)
            + "\n\nPlace the original Epoch AI CSV files in data/ before running "
              "this script. The script intentionally does not create substitute "
              "or synthetic source data."
        )


def classify_accessibility(series: pd.Series) -> pd.Series:
    """
    Collapse raw Epoch accessibility labels into two broad categories.

    This is a convenience category for visual comparison only; the original
    accessibility field is retained in the processed data.
    """
    text = series.fillna("").astype(str).str.lower()

    return np.where(
        text.str.contains("open|unrestricted|weights", regex=True, na=False),
        "Open-Weight",
        "Closed / API-Only",
    )


def safe_log10(series: pd.Series) -> pd.Series:
    """Return log10 only for strictly positive numeric values."""
    numeric = pd.to_numeric(series, errors="coerce")
    return np.where(numeric > 0, np.log10(numeric), np.nan)


# ---------------------------------------------------------------------
# 1. Notable-model data
# ---------------------------------------------------------------------

def transform_models(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Epoch AI Notable Models for scaling / compute analysis.

    This table is NOT given synthetic capability scores.
    """

    col_map = {
        "Model": "model_name",
        "Organization": "organization",
        "Publication date": "publication_date",
        "Domain": "domain",
        "Parameters": "parameters",
        "Training compute (FLOP)": "training_compute_flop",
        "Training dataset size (total)": "dataset_size_datapoints",
        "Training compute cost (2023 USD)": "training_cost_usd",
        "Model accessibility": "accessibility",
        "Base model": "base_model",
    }

    missing_required = [
        col
        for col in [
            "Model",
            "Publication date",
            "Parameters",
            "Training compute (FLOP)",
        ]
        if col not in df_raw.columns
    ]

    if missing_required:
        raise ValueError(
            "source_epoch_models.csv is missing required column(s): "
            + ", ".join(missing_required)
        )

    available_cols = [col for col in col_map if col in df_raw.columns]

    df = (
        df_raw[available_cols]
        .copy()
        .rename(columns=col_map)
    )

    # Dates and scope
    df["publication_date"] = pd.to_datetime(
        df["publication_date"],
        errors="coerce",
    )
    df["release_year"] = df["publication_date"].dt.year

    # Assignment narrative currently focuses on models released from 2018–2026.
    df = df[df["release_year"].between(2018, 2026)].copy()

    # Numeric fields
    numeric_cols = [
        "parameters",
        "training_compute_flop",
        "dataset_size_datapoints",
        "training_cost_usd",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # The scaling charts require both compute and parameter count.
    df = df.dropna(
        subset=["model_name", "training_compute_flop", "parameters"]
    ).copy()

    df = df[
        (df["training_compute_flop"] > 0)
        & (df["parameters"] > 0)
    ].copy()

    # Derived scale variables
    df["log10_compute_flop"] = np.log10(df["training_compute_flop"])
    df["log10_parameters"] = np.log10(df["parameters"])
    df["parameters_billions"] = df["parameters"] / 1e9

    # Training-intensity proxy.
    # IMPORTANT: this is compute per parameter, NOT a direct measure of
    # algorithmic efficiency.
    df["flop_per_param"] = (
        df["training_compute_flop"] / df["parameters"]
    )
    df["log10_flop_per_param"] = np.log10(df["flop_per_param"])

    # Accessibility / licence grouping
    if "accessibility" in df.columns:
        df["licence_category"] = classify_accessibility(
            df["accessibility"]
        )
    else:
        df["accessibility"] = np.nan
        df["licence_category"] = "Unspecified"

    # Primary domain
    if "domain" in df.columns:
        df["primary_domain"] = (
            df["domain"]
            .fillna("Unspecified")
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
        )
    else:
        df["domain"] = "Unspecified"
        df["primary_domain"] = "Unspecified"

    # Primary key documented for this processed table
    df = (
        df.drop_duplicates(
            subset=["model_name", "publication_date"]
        )
        .sort_values("publication_date", ascending=False)
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# 2. Benchmark-run data
# ---------------------------------------------------------------------

def transform_benchmarks(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Epoch AI Capabilities & Benchmarking export.

    Each output row remains a REAL benchmark observation. Scores are never
    inferred from compute, model size, release year, or any scaling formula.
    """

    required = [
        "id_runs",
        "id_model_version",
        "Model",
        "Best score (across scorers)",
    ]

    missing_required = [
        col for col in required if col not in df_raw.columns
    ]

    if missing_required:
        raise ValueError(
            "source_epoch_benchmarks.csv is missing required column(s): "
            + ", ".join(missing_required)
        )

    df = pd.DataFrame(index=df_raw.index)

    # Identifiers
    df["benchmark_run_id"] = df_raw["id_runs"]
    df["model_version_id"] = df_raw["id_model_version"]
    df["model_name"] = df_raw["Model"]

    # Benchmark name: original_task_name is the clearest human-readable field
    # in the current Epoch export; fall back to task if needed.
    if "original_task_name" in df_raw.columns:
        benchmark_name = df_raw["original_task_name"].copy()
    else:
        benchmark_name = pd.Series(np.nan, index=df_raw.index)

    if "task" in df_raw.columns:
        benchmark_name = benchmark_name.fillna(df_raw["task"])

    df["benchmark_name"] = benchmark_name

    # Measured score only
    score = pd.to_numeric(
        df_raw["Best score (across scorers)"],
        errors="coerce",
    )

    # Defensive fallback to the raw best_score field if the display field is
    # unexpectedly missing for a row. This is still a measured source value.
    if "best_score" in df_raw.columns:
        score = score.fillna(
            pd.to_numeric(df_raw["best_score"], errors="coerce")
        )

    # Current Epoch benchmark export uses a 0–1 score scale.
    # Keep the original proportion AND a percentage-point version.
    df["benchmark_score"] = score
    df["benchmark_score_pct"] = score * 100.0
    df["benchmark_score_is_measured"] = True

    # Dates: prefer model-version release date, then model publication date.
    if "Version release date" in df_raw.columns:
        release_date = pd.to_datetime(
            df_raw["Version release date"],
            errors="coerce",
        )
    else:
        release_date = pd.Series(pd.NaT, index=df_raw.index)

    if "Publication date" in df_raw.columns:
        release_date = release_date.fillna(
            pd.to_datetime(
                df_raw["Publication date"],
                errors="coerce",
            )
        )

    df["publication_date"] = release_date
    df["release_year"] = df["publication_date"].dt.year

    # Model metadata embedded in Epoch benchmark export
    mapping = {
        "Organization": "organization",
        "Domain": "domain",
        "Parameters": "parameters",
        "Training compute (FLOP)": "training_compute_flop",
        "Training compute cost (2023 USD)": "training_cost_usd",
        "Model accessibility": "accessibility",
        "Country (of organization)": "country",
        "Reference": "reference",
        "Link_model": "model_link",
    }

    for source_col, target_col in mapping.items():
        if source_col in df_raw.columns:
            df[target_col] = df_raw[source_col]
        else:
            df[target_col] = np.nan

    # Numeric conversion
    for col in [
        "parameters",
        "training_compute_flop",
        "training_cost_usd",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived numeric variables are created ONLY where the raw value exists.
    df["log10_compute_flop"] = safe_log10(
        df["training_compute_flop"]
    )
    df["log10_parameters"] = safe_log10(df["parameters"])
    df["parameters_billions"] = df["parameters"] / 1e9

    df["flop_per_param"] = np.where(
        (df["training_compute_flop"] > 0)
        & (df["parameters"] > 0),
        df["training_compute_flop"] / df["parameters"],
        np.nan,
    )

    df["log10_flop_per_param"] = safe_log10(
        df["flop_per_param"]
    )

    # Availability flags make chart filtering explicit and auditable.
    df["has_training_compute"] = (
        df["training_compute_flop"].notna()
        & (df["training_compute_flop"] > 0)
    )
    df["has_parameters"] = (
        df["parameters"].notna()
        & (df["parameters"] > 0)
    )

    # Accessibility grouping
    df["licence_category"] = classify_accessibility(
        df["accessibility"]
    )

    # Domain
    df["primary_domain"] = (
        df["domain"]
        .fillna("Unspecified")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )

    # Keep only rows that truly contain a measured score.
    df = df.dropna(
        subset=[
            "benchmark_run_id",
            "model_version_id",
            "model_name",
            "benchmark_name",
            "benchmark_score",
        ]
    ).copy()

    # Sanity check: source score is expected to be a proportion.
    invalid_score = ~df["benchmark_score"].between(0, 1)

    if invalid_score.any():
        examples = df.loc[
            invalid_score,
            ["model_name", "benchmark_name", "benchmark_score"],
        ].head()

        raise ValueError(
            "Found benchmark scores outside the expected 0–1 range.\n"
            + examples.to_string(index=False)
        )

    # Current Epoch export should contain one row per
    # model-version × benchmark pair.
    df = (
        df.drop_duplicates(
            subset=["model_version_id", "benchmark_name"],
            keep="last",
        )
        .sort_values(
            ["benchmark_name", "publication_date", "model_name"]
        )
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# 3. Primary GPQA analytical subset
# ---------------------------------------------------------------------

def build_gpqa_subset(
    benchmark_df: pd.DataFrame,
    benchmark_name: str = PRIMARY_BENCHMARK,
) -> pd.DataFrame:
    """
    Create the measured GPQA subset used for capability analysis.

    IMPORTANT:
    - Rows without compute are retained.
    - Capability-vs-compute charts must explicitly filter
      has_training_compute == True.
    - No missing benchmark score is filled or estimated.
    """
    gpqa = benchmark_df[
        benchmark_df["benchmark_name"].str.casefold()
        == benchmark_name.casefold()
    ].copy()

    gpqa = gpqa.sort_values(
        ["publication_date", "model_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return gpqa


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------

def main():
    require_source_files()

    raw_models = pd.read_csv(RAW_MODELS_PATH)
    raw_benchmarks = pd.read_csv(RAW_BENCHMARKS_PATH)

    processed_models = transform_models(raw_models)
    processed_benchmarks = transform_benchmarks(raw_benchmarks)
    processed_gpqa = build_gpqa_subset(processed_benchmarks)

    processed_models.to_csv(
        PROCESSED_MODELS_PATH,
        index=False,
    )
    processed_benchmarks.to_csv(
        PROCESSED_BENCHMARKS_PATH,
        index=False,
    )
    processed_gpqa.to_csv(
        PROCESSED_GPQA_PATH,
        index=False,
    )

    print("\n=== DATA PREPARATION SUMMARY ===")
    print(
        f"Notable models for scaling analysis: "
        f"{len(processed_models):,}"
    )
    print(
        f"Measured benchmark observations: "
        f"{len(processed_benchmarks):,}"
    )
    print(
        f"Distinct benchmark tasks: "
        f"{processed_benchmarks['benchmark_name'].nunique():,}"
    )
    print(
        f"{PRIMARY_BENCHMARK} observations: "
        f"{len(processed_gpqa):,}"
    )
    print(
        f"{PRIMARY_BENCHMARK} with measured training compute: "
        f"{processed_gpqa['has_training_compute'].sum():,}"
    )
    print(
        f"{PRIMARY_BENCHMARK} with measured compute + parameters: "
        f"{(
            processed_gpqa['has_training_compute']
            & processed_gpqa['has_parameters']
        ).sum():,}"
    )

    if not processed_gpqa.empty:
        years = processed_gpqa["release_year"].dropna()
        if not years.empty:
            print(
                f"{PRIMARY_BENCHMARK} release-year coverage: "
                f"{int(years.min())}–{int(years.max())}"
            )

    print("\nSaved:")
    print(f"- {PROCESSED_MODELS_PATH}")
    print(f"- {PROCESSED_BENCHMARKS_PATH}")
    print(f"- {PROCESSED_GPQA_PATH}")

    print(
        "\nNOTE: No synthetic benchmark scores were created. "
        "Missing measured values remain NaN."
    )


if __name__ == "__main__":
    main()
