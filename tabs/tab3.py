from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


GPQA_BENCHMARK_NAME = "GPQA diamond"
MIN_MODELS_PER_YEAR = 3

# Comparable training-compute bands with enough observations to support
# meaningful year-to-year comparisons in the current GPQA dataset.
COMPUTE_BANDS = {
    "10²³–10²⁴ FLOP": (23.0, 24.0),
    "10²⁴–10²⁵ FLOP": (24.0, 25.0),
    "10²⁵–10²⁶ FLOP": (25.0, 26.0),
}

DEFAULT_BAND = "10²⁴–10²⁵ FLOP"
SOURCE_URL = "https://epoch.ai/benchmarks/use-this-data"


def _find_gpqa_path() -> Path:
    """
    Locate data/gpqa.csv whether this module is run from the repo root,
    imported from a tabs/ directory, or run directly with Streamlit.
    """
    candidates = [
        Path("data/gpqa.csv"),
        Path(__file__).resolve().parent / "data" / "gpqa.csv",
        Path(__file__).resolve().parent.parent / "data" / "gpqa.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find data/gpqa.csv. "
        "Make sure the processed GPQA file is inside the repository's data/ folder."
    )


@st.cache_data(show_spinner=False)
def load_gpqa_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _prepare_gpqa(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "benchmark_run_id",
        "model_name",
        "benchmark_name",
        "benchmark_score_pct",
        "publication_date",
        "release_year",
        "training_compute_flop",
        "log10_compute_flop",
        "has_training_compute",
        "organization",
        "accessibility",
        "licence_category",
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "gpqa.csv is missing required column(s): " + ", ".join(missing)
        )

    out = df.copy()

    out["publication_date"] = pd.to_datetime(
        out["publication_date"], errors="coerce"
    )
    out["release_year"] = pd.to_numeric(out["release_year"], errors="coerce")
    out["benchmark_score_pct"] = pd.to_numeric(
        out["benchmark_score_pct"], errors="coerce"
    )
    out["training_compute_flop"] = pd.to_numeric(
        out["training_compute_flop"], errors="coerce"
    )
    out["log10_compute_flop"] = pd.to_numeric(
        out["log10_compute_flop"], errors="coerce"
    )

    # Defensive conversion because CSV readers may occasionally parse booleans
    # as strings depending on the environment.
    if out["has_training_compute"].dtype != bool:
        out["has_training_compute"] = (
            out["has_training_compute"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

    # Story 2 uses measured GPQA Diamond results with reported training compute.
    out = out[
        out["benchmark_name"]
        .astype(str)
        .str.casefold()
        .eq(GPQA_BENCHMARK_NAME.casefold())
        & out["has_training_compute"].eq(True)
        & out["benchmark_score_pct"].notna()
        & out["log10_compute_flop"].notna()
        & out["release_year"].notna()
    ].copy()

    return out


def _summarise_by_year(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("release_year", as_index=False)
        .agg(
            n_models=("benchmark_run_id", "size"),
            median_gpqa_pct=("benchmark_score_pct", "median"),
            q1_gpqa_pct=("benchmark_score_pct", lambda x: x.quantile(0.25)),
            q3_gpqa_pct=("benchmark_score_pct", lambda x: x.quantile(0.75)),
            median_log10_compute=("log10_compute_flop", "median"),
        )
        .sort_values("release_year")
    )

    summary["release_year"] = summary["release_year"].astype(int)
    return summary


def _build_chart(summary: pd.DataFrame, band_label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=summary["release_year"],
            y=summary["median_gpqa_pct"],
            mode="lines+markers+text",
            name="Median GPQA",
            text=[
                f"{score:.1f}%<br>n={n}"
                for score, n in zip(
                    summary["median_gpqa_pct"],
                    summary["n_models"],
                )
            ],
            textposition="top center",
            line=dict(width=3, color="#0072B2"),
            marker=dict(size=10, color="#0072B2"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=summary["q3_gpqa_pct"] - summary["median_gpqa_pct"],
                arrayminus=summary["median_gpqa_pct"] - summary["q1_gpqa_pct"],
                thickness=1.5,
                width=5,
                color="#0072B2",
            ),
            customdata=np.column_stack(
                [
                    summary["n_models"],
                    summary["q1_gpqa_pct"],
                    summary["q3_gpqa_pct"],
                    summary["median_log10_compute"],
                ]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Median GPQA: %{y:.1f}%<br>"
                "Models: %{customdata[0]:.0f}<br>"
                "IQR: %{customdata[1]:.1f}%–%{customdata[2]:.1f}%<br>"
                "Median log10 compute: %{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"GPQA capability over time within {band_label}",
            x=0.0,
            xanchor="left",
        ),
        xaxis_title="Model release year",
        yaxis_title="Median GPQA Diamond score (%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        xaxis=dict(tickmode="linear", dtick=1),
        margin=dict(l=20, r=20, t=70, b=20),
        height=470,
        showlegend=False,
        hovermode="closest",
    )

    return fig


def _format_verification_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation-ready table of the observations behind a year median."""
    verification_cols = [
        "model_name",
        "organization",
        "publication_date",
        "release_year",
        "training_compute_flop",
        "log10_compute_flop",
        "benchmark_score_pct",
        "licence_category",
        "benchmark_run_id",
    ]

    if "reference" in df.columns:
        verification_cols.append("reference")

    verification = (
        df[verification_cols]
        .sort_values("benchmark_score_pct", ascending=False)
        .copy()
    )

    verification["publication_date"] = verification["publication_date"].dt.strftime(
        "%Y-%m-%d"
    )
    verification["training_compute_flop"] = verification[
        "training_compute_flop"
    ].map(lambda x: f"{x:.3e}" if pd.notna(x) else "")
    verification["log10_compute_flop"] = verification["log10_compute_flop"].round(3)
    verification["benchmark_score_pct"] = verification[
        "benchmark_score_pct"
    ].round(2)

    return verification.rename(
        columns={
            "model_name": "Model",
            "organization": "Organisation",
            "publication_date": "Publication date",
            "release_year": "Release year",
            "training_compute_flop": "Training compute (FLOP)",
            "log10_compute_flop": "log10(FLOP)",
            "benchmark_score_pct": "GPQA Diamond score (%)",
            "licence_category": "Licence category",
            "benchmark_run_id": "Benchmark run ID",
            "reference": "Reference",
        }
    )


def display_tab3(gpqa_df: pd.DataFrame | None = None) -> None:
    """
    Render Tab 3 / User Story 2.

    Optional:
        Pass an already-loaded GPQA dataframe from app.py to avoid re-reading
        the file. If omitted, this function loads data/gpqa.csv itself.
    """

    # ------------------------------------------------------------------
    # Header / user story
    # ------------------------------------------------------------------
    st.header("User Story 2 · Capability at comparable compute")

    st.markdown(
        """
**As an AI Infrastructure Lead, I want to compare the capability of models
released at different times under similar training-compute levels, so that I
can assess whether newer models are achieving better performance without
proportionally increasing compute.**
        """
    )

    st.caption(
        "Audience: AI Infrastructure Lead · Benchmark: GPQA Diamond · "
        "Compute: reported pre-training FLOP · Missing compute is excluded, not estimated."
    )

    try:
        if gpqa_df is None:
            gpqa_path = _find_gpqa_path()
            gpqa_df = load_gpqa_data(str(gpqa_path))

        gpqa = _prepare_gpqa(gpqa_df)

    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    # ------------------------------------------------------------------
    # Decision control
    # ------------------------------------------------------------------
    control_col, _ = st.columns([2, 1])

    with control_col:
        band_label = st.selectbox(
            "Compare models within a similar training-compute range",
            options=list(COMPUTE_BANDS.keys()),
            index=list(COMPUTE_BANDS.keys()).index(DEFAULT_BAND),
            help=(
                "Keeping models within the same order-of-magnitude compute band "
                "reduces the extent to which a simple increase in training compute "
                "explains differences in benchmark performance."
            ),
            key="tab3_compute_band",
        )

    low, high = COMPUTE_BANDS[band_label]

    selected = gpqa[
        (gpqa["log10_compute_flop"] >= low)
        & (gpqa["log10_compute_flop"] < high)
    ].copy()

    if selected.empty:
        st.warning(
            "No GPQA observations with reported training compute fall inside "
            "the selected range. Choose another compute band."
        )
        return

    raw_summary = _summarise_by_year(selected)
    summary = raw_summary[raw_summary["n_models"] >= MIN_MODELS_PER_YEAR].copy()

    if summary.empty:
        st.warning(
            f"The selected range has no release year with at least "
            f"{MIN_MODELS_PER_YEAR} measured models. Choose another compute range."
        )
        return

    # ------------------------------------------------------------------
    # KPI row
    # ------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Measured models in band", f"{len(selected):,}")
    c2.metric(
        "Years shown",
        f"{len(summary):,}",
        help=f"Only years with at least {MIN_MODELS_PER_YEAR} observations are plotted.",
    )
    c3.metric("Overall median GPQA", f"{selected['benchmark_score_pct'].median():.1f}%")

    # ------------------------------------------------------------------
    # Key insight
    # ------------------------------------------------------------------
    if len(summary) >= 2:
        first = summary.iloc[0]
        last = summary.iloc[-1]
        change_pp = last["median_gpqa_pct"] - first["median_gpqa_pct"]
        direction = "higher" if change_pp >= 0 else "lower"

        st.info(
            f"**What the selected view shows:** among years with at least "
            f"{MIN_MODELS_PER_YEAR} measured models, median GPQA was "
            f"**{first['median_gpqa_pct']:.1f}% in {int(first['release_year'])}** "
            f"and **{last['median_gpqa_pct']:.1f}% in {int(last['release_year'])}** "
            f"— **{abs(change_pp):.1f} percentage points {direction}**. "
            "This is consistent with capability improvement at comparable reported "
            "training-compute levels, but it does not establish a causal effect of "
            "algorithmic progress."
        )
    else:
        st.info(
            "Only one year in this compute range currently has enough "
            "observations for a year-level comparison."
        )

    # ------------------------------------------------------------------
    # 1 · Main evidence
    # ------------------------------------------------------------------
    st.subheader("1 · Capability over time at comparable compute")

    st.plotly_chart(
        _build_chart(summary, band_label),
        use_container_width=True,
        config={"displayModeBar": False},
        key="tab3_capability_chart",
    )

    st.caption(
        f"Figure: Median measured GPQA Diamond score by release year for models "
        f"trained within {band_label}. Error bars show the interquartile range. "
        f"Years with fewer than {MIN_MODELS_PER_YEAR} observations are not plotted. "
        "Scores from different benchmark tasks are not pooled."
    )

    with st.expander("View the underlying yearly values"):
        display_summary = summary.rename(
            columns={
                "release_year": "Release year",
                "n_models": "Models (n)",
                "median_gpqa_pct": "Median GPQA (%)",
                "q1_gpqa_pct": "25th percentile (%)",
                "q3_gpqa_pct": "75th percentile (%)",
                "median_log10_compute": "Median log10(FLOP)",
            }
        ).copy()

        for col in [
            "Median GPQA (%)",
            "25th percentile (%)",
            "75th percentile (%)",
            "Median log10(FLOP)",
        ]:
            display_summary[col] = display_summary[col].round(2)

        st.dataframe(display_summary, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 2 · Verification
    # ------------------------------------------------------------------
    st.subheader("2 · Verify a displayed value")

    st.markdown(
        "**Data lineage:** Epoch AI benchmark export → `raw_benchmarks.csv` "
        "→ `prepared_data.py` → `benchmarks.csv` → GPQA Diamond filter "
        "→ `gpqa.csv` → compute-band filter → yearly median."
    )

    trace_year = st.selectbox(
        "Choose a displayed year to verify",
        options=summary["release_year"].tolist(),
        index=len(summary) - 1,
        key="tab3_trace_year",
    )

    displayed_row = summary.loc[summary["release_year"].eq(trace_year)].iloc[0]
    year_rows = selected.loc[selected["release_year"].eq(trace_year)].copy()
    recalculated_median = year_rows["benchmark_score_pct"].median()

    st.markdown(
        f"**Displayed result:** {int(trace_year)} median GPQA = "
        f"**{displayed_row['median_gpqa_pct']:.2f}%**, based on "
        f"**{int(displayed_row['n_models'])} measured models** in "
        f"**{band_label}**."
    )

    v1, v2, v3 = st.columns(3)
    v1.metric("Displayed median", f"{displayed_row['median_gpqa_pct']:.2f}%")
    v2.metric("Recalculated median", f"{recalculated_median:.2f}%")
    v3.metric("Underlying models", f"{len(year_rows):,}")

    median_matches = np.isclose(
        displayed_row["median_gpqa_pct"],
        recalculated_median,
        rtol=0,
        atol=1e-9,
    )
    band_matches = (
        year_rows["log10_compute_flop"].ge(low)
        & year_rows["log10_compute_flop"].lt(high)
    ).all()
    count_matches = len(year_rows) == int(displayed_row["n_models"])

    if median_matches:
        st.success(
            "Verified: the displayed yearly median equals the median of the "
            "underlying measured GPQA scores."
        )
    else:
        st.error(
            "Verification failed: the displayed yearly median does not match "
            "the median recalculated from the underlying observations."
        )

    if band_matches and count_matches:
        st.success(
            "Verified: every observation used for this displayed year is inside "
            "the selected compute band, and the displayed model count matches "
            "the underlying rows."
        )
    else:
        st.error(
            "Verification failed: at least one underlying row falls outside the "
            "selected compute band or the displayed count does not match."
        )

    with st.expander("View the observations behind this median"):
        verification = _format_verification_rows(year_rows)
        st.dataframe(verification, use_container_width=True, hide_index=True)

    st.markdown(
        f"Source: [Epoch AI — Capabilities & Benchmarking]({SOURCE_URL}) "
        "(CC BY attribution)."
    )

    # ------------------------------------------------------------------
    # Interpretation / limits
    # ------------------------------------------------------------------
    with st.expander("Interpretation, limitations and sources"):
        st.markdown(
            f"""
- **What this shows:** observed GPQA Diamond capability over time among models whose reported pre-training compute falls within the selected **{band_label}** range.
- **What this cannot show:** a causal estimate of algorithmic efficiency. Differences can still reflect architecture, training data, post-training methods, model selection and other factors.
- **Comparable-compute rule:** compute is controlled only at a broad order-of-magnitude level, not at exactly equal FLOP.
- **Minimum-sample rule:** a release year is displayed only when it has at least **{MIN_MODELS_PER_YEAR} measured models**, avoiding a yearly median based on only one or two observations.
- **Missingness matters:** GPQA observations without published training compute are excluded, which may affect the pattern shown.
- **Measured-only rule:** benchmark scores and compute values are not imputed.
- **Source:** [Epoch AI — Capabilities & Benchmarking]({SOURCE_URL}).
- **Reproducibility:** see `data/DATA_DICTIONARY.md` and `prepared_data.py` for the documented source fields and transforms.
            """
        )


def render(gpqa_df: pd.DataFrame | None = None) -> None:
    display_tab3(gpqa_df)


def show(gpqa_df: pd.DataFrame | None = None) -> None:
    display_tab3(gpqa_df)


# Allows this tab to be tested directly with:
#     streamlit run tabs/tab3.py
# while still working normally when imported into the main app.
if __name__ == "__main__":
    # This only applies when tab3.py is launched directly. When imported by the
    # main app, the root app should own st.set_page_config().
    st.set_page_config(
        page_title="User Story 2 · Capability at Comparable Compute",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    display_tab3()
