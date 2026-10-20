"""DVN36104 Assignment 3 — User Story 3B: Resource-efficient choices.

Drop this file into the group's ``tabs/`` directory. The main app can call
``tabs.tab5.render()`` (or ``tabs.tab5.render(df=gpqa_dataframe)``).

For an independent preview, run from the repository root:
    streamlit run tabs/tab5.py

Data are always read from the group's real ``data/gpqa.csv``; no demonstration
scores or missing-data imputations are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REQUIRED_COLUMNS = {
    "model_name", "benchmark_name", "benchmark_run_id", "benchmark_score_pct",
    "training_compute_flop", "publication_date", "organization",
}


def load_data() -> pd.DataFrame:
    """Read the processed GPQA Diamond subset (do not fabricate scores)."""
    path = DATA_DIR / "gpqa.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place tab5.py under tabs/ and use the "
            "group's original data/ folder."
        )
    return pd.read_csv(path)


def prepare_data(source: pd.DataFrame) -> pd.DataFrame:
    """Keep measured GPQA Diamond observations with known positive compute."""
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"The input CSV is missing required fields: {', '.join(missing)}")

    df = source.copy()
    df = df[df["benchmark_name"].astype(str).str.strip().str.casefold() == "gpqa diamond"]
    df["training_compute_flop"] = pd.to_numeric(df["training_compute_flop"], errors="coerce")
    df["benchmark_score_pct"] = pd.to_numeric(df["benchmark_score_pct"], errors="coerce")
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")

    # No numeric estimates, imputation or zero substitutions: use only source measurements.
    df = df[
        df["training_compute_flop"].gt(0)
        & df["benchmark_score_pct"].between(0, 100)
        & df["model_name"].notna()
        & df["benchmark_run_id"].notna()
    ].copy()

    if "benchmark_score_is_measured" in df.columns:
        measured = df["benchmark_score_is_measured"].astype(str).str.casefold().eq("true")
        df = df[measured].copy()
    if "has_training_compute" in df.columns:
        df = df[df["has_training_compute"].astype(str).str.casefold().eq("true")].copy()

    df["release_year"] = df["publication_date"].dt.year
    df["organization"] = df["organization"].fillna("Unspecified").astype(str)
    df["model_name"] = df["model_name"].astype(str)
    df = df.drop_duplicates(subset=["benchmark_run_id"])
    return df.reset_index(drop=True)


def add_pareto_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Tag observed non-dominated models: no other row has <= compute AND >= score.

    Computed within the current year/organisation selection, *before* applying
    the user-selected minimum score. Not a causal or monetary efficiency metric.
    """
    result = df.copy()
    compute = result["training_compute_flop"].to_numpy(dtype=float)
    score = result["benchmark_score_pct"].to_numpy(dtype=float)
    frontier = []
    for i in range(len(result)):
        dominates = (
            (compute <= compute[i])
            & (score >= score[i])
            & ((compute < compute[i]) | (score > score[i]))
        )
        frontier.append(not bool(dominates.any()))
    result["is_pareto"] = frontier
    return result


def _sci(value: float) -> str:
    return f"{value:.2e} FLOP"


def _source_check(row: pd.Series) -> None:
    """Provide a reproducible raw-record lookup for the video verification."""
    raw_path = DATA_DIR / "raw_benchmarks.csv"
    with st.expander("Trace this model's displayed score back to source", expanded=False):
        st.markdown(
            "**Data lineage:** Epoch AI benchmark export → `raw_benchmarks.csv` → "
            "`prepared_data.py` → `benchmarks.csv` → GPQA Diamond filter → `gpqa.csv`."
        )
        st.write(f"Processed benchmark run ID: `{row['benchmark_run_id']}`")
        st.write(
            f"Processed GPQA Diamond score: **{row['benchmark_score_pct']:.2f}%**; "
            f"training compute: **{_sci(row['training_compute_flop'])}**."
        )
        if not raw_path.exists():
            st.warning("Original `data/raw_benchmarks.csv` is not present here; verify against the group repository.")
            return

        raw = pd.read_csv(raw_path, low_memory=False)
        if "id_runs" not in raw.columns:
            st.warning("The raw file has no `id_runs` column. Check the preprocessing script.")
            return
        matches = raw[raw["id_runs"].astype(str) == str(row["benchmark_run_id"])].copy()
        if matches.empty:
            st.warning("No raw record matched the benchmark run ID. Ask the data owner to check the export.")
            return
        src = matches.iloc[0]
        raw_score = pd.to_numeric(pd.Series([src.get("Best score (across scorers)")]), errors="coerce").iloc[0]
        score_field = "Best score (across scorers)"
        if pd.isna(raw_score):
            raw_score = pd.to_numeric(pd.Series([src.get("best_score")]), errors="coerce").iloc[0]
            score_field = "best_score (measured-source fallback)"
        raw_compute = pd.to_numeric(pd.Series([src.get("Training compute (FLOP)")]), errors="coerce").iloc[0]
        summary = pd.DataFrame(
            {
                "Field": ["Raw record: Model", "Raw benchmark task", f"Raw {score_field}",
                          "Raw training compute (FLOP)", "Processed score (%)", "Processed compute (FLOP)"],
                "Value": [str(src.get("Model", "")), str(src.get("original_task_name", "")),
                          "Missing" if pd.isna(raw_score) else f"{raw_score:.8g}",
                          "Missing" if pd.isna(raw_compute) else f"{raw_compute:.6e}",
                          f"{row['benchmark_score_pct']:.4f}", f"{row['training_compute_flop']:.6e}"],
            }
        )
        st.dataframe(summary, use_container_width=True)
        if pd.notna(raw_score):
            if np.isclose(raw_score * 100.0, float(row["benchmark_score_pct"]), atol=1e-4):
                st.success("Verified: measured raw benchmark score × 100 equals the processed percentage.")
            else:
                st.error("Raw and processed scores differ; check `prepared_data.py` before presenting this model.")
        if pd.notna(raw_compute):
            if np.isclose(raw_compute, float(row["training_compute_flop"]), rtol=1e-9):
                st.success("Verified: training FLOP matches the raw benchmark export.")
            else:
                st.error("Training FLOP differs; check source transformation before making claims.")
        st.caption(
            "Source: Epoch AI, Capabilities & Benchmarking — "
            "https://epoch.ai/benchmarks/use-this-data (CC BY attribution)."
        )


def render(df: Optional[pd.DataFrame] = None) -> None:
    """Render US3B within the team app. Optionally pass the team's GPQA dataframe."""
    st.header("US3B · Resource-efficient model choices")
    st.write(
        "**Decision question:** Which measured models meet a minimum GPQA Diamond "
        "capability requirement with comparatively less *training* compute?"
    )
    st.caption("Audience: AI Infrastructure Lead · Benchmark: GPQA Diamond · Compute: pre-training FLOP")

    try:
        source = load_data() if df is None else df
        valid = prepare_data(source)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Unable to load US3B data: {exc}")
        st.info("Required file: `data/gpqa.csv`. Confirm the paths and data dictionary before proceeding.")
        return

    if valid.empty:
        st.warning("No measured GPQA Diamond observations contain positive training compute.")
        return

    st.caption(
        f"Source coverage: {len(source):,} GPQA source rows; "
        f"{len(valid):,} rows have both a measured GPQA score and positive training FLOP. "
        "Missing compute values are excluded, not treated as zero."
    )

    years = sorted(int(y) for y in valid["release_year"].dropna().unique())
    orgs = sorted(valid["organization"].unique().tolist())

    st.subheader("1 · Set your decision requirements")
    left, middle, right = st.columns(3)
    with left:
        min_score = st.slider("Minimum GPQA Diamond score (%)", min_value=0, max_value=100, value=60, step=5)
    with middle:
        if years and min(years) != max(years):
            selected_years = st.slider("Model release years", min(years), max(years), (min(years), max(years)))
        elif years:
            st.caption(f"Release year: {years[0]}")
            selected_years = (years[0], years[0])
        else:
            st.caption("Release date unavailable")
            selected_years = None
    with right:
        org = st.selectbox("Organisation", ["All organisations"] + orgs)

    comparison = valid.copy()
    if selected_years is not None:
        comparison = comparison[comparison["release_year"].between(*selected_years)]
    if org != "All organisations":
        comparison = comparison[comparison["organization"] == org]

    # Relative-efficiency flags are based on the visible comparison cohort.
    comparison = add_pareto_labels(comparison)
    eligible = comparison[comparison["benchmark_score_pct"] >= min_score].copy()
    eligible = eligible.sort_values(["training_compute_flop", "benchmark_score_pct"], ascending=[True, False])

    a, b, c = st.columns(3)
    a.metric("Models with compute + GPQA", f"{len(comparison):,}")
    b.metric("Meet your score threshold", f"{len(eligible):,}")
    c.metric("Pareto-efficient (in cohort)", f"{comparison['is_pareto'].sum():,}")

    if eligible.empty:
        st.warning("No recorded model meets these filters. Lower the score threshold or broaden the period.")
        return

    leader = eligible.iloc[0]
    st.info(
        f"**Decision insight (for selected filters):** {leader['model_name']} "
        f"has the lowest *reported training compute* among the {len(eligible)} GPQA-qualified "
        f"observations: {_sci(leader['training_compute_flop'])}, "
        f"with a measured GPQA Diamond score of {leader['benchmark_score_pct']:.1f}%. "
        "This is a comparison within the observed sample, not evidence that it has the "
        "lowest inference cost or best overall efficiency."
    )

    st.subheader("2 · Rank qualifying models by reported training compute")
    top_n = st.slider("Number of models to display", min_value=1, max_value=min(20, len(eligible)),
                      value=min(10, len(eligible)))
    ranked = eligible.head(top_n).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["display_name"] = ranked.apply(
        lambda r: f"{r['rank']:02d}. {r['model_name']} ({r['publication_date']:%Y-%m-%d})"
        if pd.notna(r["publication_date"])
        else f"{r['rank']:02d}. {r['model_name']}", axis=1
    )
    ranked["Status"] = np.where(ranked["is_pareto"], "Pareto-efficient", "Other qualifying model")

    fig = px.scatter(
        ranked,
        x="training_compute_flop",
        y="display_name",
        log_x=True,
        color="Status",
        color_discrete_map={"Pareto-efficient": "#087e8b", "Other qualifying model": "#ef9a38"},
        hover_name="model_name",
        hover_data={
            "training_compute_flop": ":.3e", "benchmark_score_pct": ":.2f",
            "organization": True, "display_name": False, "Status": False,
        },
        labels={
            "training_compute_flop": "Training compute (FLOP; log scale)",
            "display_name": "Model ranked by compute", "benchmark_score_pct": "GPQA Diamond (%)",
        },
        title="Lower training FLOP is further left (same minimum capability requirement)",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=ranked["display_name"].iloc[::-1].tolist(), title=None)
    fig.update_traces(marker={"size": 12, "line": {"width": 1, "color": "white"}})
    fig.update_layout(height=max(400, 48 * len(ranked) + 160), margin={"l": 10, "r": 15, "t": 70, "b": 30}, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "A ranked *dot plot* is used instead of a raw-FLOP bar chart: training FLOP spans "
        "orders of magnitude, so a logarithmic axis makes the comparison readable. "
        "Pareto-efficient means no other observation in the chosen comparison cohort has "
        "both at least as high a score and no more training compute (with one strict improvement)."
    )

    shown = ranked[["rank", "model_name", "organization", "benchmark_score_pct",
                    "training_compute_flop", "publication_date", "Status", "benchmark_run_id"]].copy()
    shown.columns = ["Rank", "Model", "Organisation", "GPQA Diamond (%)", "Training FLOP",
                     "Publication date", "Comparison status", "Benchmark run ID"]
    st.dataframe(shown, use_container_width=True)
    st.subheader("3 · Verify a displayed model")
    selected_rank = st.selectbox(
        "Select a model from the ranking",
        list(range(len(ranked))),
        format_func=lambda i: ranked.iloc[i]["display_name"],
    )
    _source_check(ranked.iloc[selected_rank])

    with st.expander("Interpretation, limitations and sources"):
        st.markdown(
            "- **What this shows:** observed training-resource trade-offs among models "
            "with measured GPQA Diamond scores.\n"
            "- **What this cannot show:** real-world inference cost, latency, model suitability "
            "for every task, causal effects of compute, or direct algorithmic efficiency.\n"
            "- **Missingness matters:** the GPQA dataset includes models without published "
            "training FLOP; they are excluded, which may bias the ranking.\n"
            "- **Measured-only rule:** no benchmark scores or compute values are imputed.\n"
            "- **Source:** [Epoch AI — Capabilities & Benchmarking]"
            "(https://epoch.ai/benchmarks/use-this-data).\n"
            "- **Reproducibility:** see `data/DATA_DICTIONARY.md` and `prepared_data.py`."
        )


# Compatibility with different possible conventions of the team's main app.
def show(df: Optional[pd.DataFrame] = None) -> None:
    render(df)


def main() -> None:
    render()


if __name__ == "__main__":
    st.set_page_config(page_title="US3B · Resource-efficient models", layout="wide")
    render()
