"""DVN36104 Assignment 3 — Tab 2 (User Story 1): Compute and capability.

User story: As an AI Infrastructure Lead, I want to compare measured AI capability
with training compute, so that I can assess whether higher-compute models are
consistently associated with stronger performance.

Data: the group's processed ``data/gpqa.csv`` (Epoch AI — Capabilities & Benchmarking,
GPQA Diamond only), restricted to rows with disclosed training compute.
Only measured source values are used; nothing is imputed.

Layout follows the shared tab structure (Tab 3 reference):
header → user story → scope caption → controls → KPI row → key insight →
main charts → supporting table (collapsed) → visible verification →
collapsed interpretation / limitations / sources.

Use inside the main app:   tab2.display_tab2()   (aliases: render(), show())
Preview on its own:        streamlit run tabs/tab2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

SOURCE_URL = "https://epoch.ai/benchmarks/use-this-data"
MIN_PER_BAND = 5  # bands with fewer observations are not used for spread statistics

# Same access colours and trend grey as Tab 1, so the app reads as one system.
ACCESS_COLOURS = {"Open-Weight": "#E69F00", "Closed / API-Only": "#0072B2"}
ACCESS_SYMBOLS = {"Open-Weight": "circle", "Closed / API-Only": "diamond"}
TREND_COLOUR = "#9A9A9A"
# Release year: mid-luminance colour-blind-safe hues (Okabe-Ito) that stay readable
# on both light and dark themes, and avoid Tab 1's orange/blue access colours.
YEAR_COLOURS = {2023: "#999999", 2024: "#56B4E9", 2025: "#009E73", 2026: "#CC79A7"}

# One order of magnitude per band; the sparse tails are merged.
BAND_EDGES = [0, 23, 24, 25, 26, 30]
BAND_LABELS = ["< 10²³", "10²³–10²⁴", "10²⁴–10²⁵", "10²⁵–10²⁶", "≥ 10²⁶"]

REQUIRED_COLUMNS = {
    "benchmark_run_id",
    "model_version_id",
    "model_name",
    "benchmark_name",
    "benchmark_score_pct",
    "publication_date",
    "organization",
    "training_compute_flop",
    "log10_compute_flop",
    "has_training_compute",
    "licence_category",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _find_data_path(filename: str) -> Optional[Path]:
    """Locate data/<filename> from the repo root, the tabs/ folder, or cwd."""
    here = Path(__file__).resolve().parent
    for path in (Path("data") / filename, here / "data" / filename, here.parent / "data" / filename):
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def prepare_gpqa(df: pd.DataFrame) -> pd.DataFrame:
    """Type-convert and keep GPQA Diamond rows with positive measured compute."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("gpqa.csv is missing required column(s): " + ", ".join(missing))

    out = df.copy()
    out["publication_date"] = pd.to_datetime(out["publication_date"], errors="coerce")
    for col in ("training_compute_flop", "log10_compute_flop", "benchmark_score_pct"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    has_compute = out["has_training_compute"].astype(str).str.lower().eq("true")

    out = out[
        out["benchmark_name"].eq("GPQA diamond")
        & has_compute
        & out["training_compute_flop"].gt(0)
        & out["benchmark_score_pct"].notna()
        & out["publication_date"].notna()
    ].copy()

    out["release_year"] = out["publication_date"].dt.year.astype(int)
    out["licence_category"] = out["licence_category"].fillna("Unspecified").astype(str)
    out["organization"] = out["organization"].fillna("Unspecified").astype(str)
    out["compute_band"] = pd.cut(
        out["log10_compute_flop"], BAND_EDGES, labels=BAND_LABELS, right=False
    ).astype(str)
    return out.sort_values("training_compute_flop").reset_index(drop=True)


def best_version_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per model: the best-scoring evaluated version (they share one compute value)."""
    return (
        df.sort_values("benchmark_score_pct")
        .groupby(["model_name", "training_compute_flop"], as_index=False)
        .last()
        .sort_values("training_compute_flop")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def fit_trend(df: pd.DataFrame) -> Optional[dict]:
    """OLS fit of GPQA score (pp) on log10(compute), plus Spearman rank correlation."""
    if len(df) < 5 or df["log10_compute_flop"].nunique() < 2:
        return None
    x, y = df["log10_compute_flop"], df["benchmark_score_pct"]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    rho, p_value = stats.spearmanr(x, y)
    return {
        "slope": float(slope),          # pp per 10x compute
        "intercept": float(intercept),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rho": float(rho),
        "p_value": float(p_value),
        "n": int(len(df)),
    }


def summarise_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Score distribution within each compute band (bands with >= MIN_PER_BAND rows)."""
    summary = (
        df.groupby("compute_band")["benchmark_score_pct"]
        .agg(n="size", min="min", median="median", max="max")
        .reindex([b for b in BAND_LABELS if b in set(df["compute_band"])])
    )
    summary["range"] = summary["max"] - summary["min"]
    return summary[summary["n"] >= MIN_PER_BAND]


def _sci(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2e} FLOP"


def _sup(exponent: int) -> str:
    return str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def _name_date(name: str, date: pd.Timestamp) -> str:
    """'Model (Mon YYYY)', unless the model name already carries a date (same rule as Tab 4)."""
    return name if "(" in name else f"{name} ({date:%b %Y})"


def _deviation_labels(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n models furthest above and below trend (one label per model)."""
    if n <= 0:
        return df.iloc[0:0]
    above = df.sort_values("residual").drop_duplicates("model_name", keep="last").nlargest(n, "residual")
    below = df.sort_values("residual").drop_duplicates("model_name", keep="first").nsmallest(n, "residual")
    return pd.concat([above, below])


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _log_x_axis(fig: go.Figure, x_min: float, x_max: float) -> None:
    """Plot on log10(FLOP) and label ticks as readable powers of ten (10²², 10²⁴ ...)."""
    lo, hi = int(np.floor(x_min)), int(np.ceil(x_max))
    ticks = list(range(lo, hi + 1))
    fig.update_xaxes(
        tickvals=ticks,
        ticktext=[f"10{_sup(t)}" for t in ticks],
        range=[lo - 0.2, hi + 0.2],
        title="Training compute (FLOP, log scale)",
        gridcolor="rgba(128,128,128,0.2)",
    )


def _score_y_axis(fig: go.Figure) -> None:
    fig.update_yaxes(title="GPQA Diamond score (%)", range=[0, 105], automargin=True,
                     gridcolor="rgba(128,128,128,0.2)")


def build_scatter(df: pd.DataFrame, trend: Optional[dict], colour_by: str, label_n: int) -> go.Figure:
    fig = go.Figure()

    if colour_by == "Release year":
        groups = [
            (f"{year} (n={len(part)})", part, YEAR_COLOURS.get(year, "#999999"))
            for year, part in df.groupby("release_year")
        ]
    else:
        groups = [
            (f"{access} (n={len(df[df['licence_category'] == access])})",
             df[df["licence_category"] == access], colour)
            for access, colour in {**ACCESS_COLOURS, "Unspecified": "#999999"}.items()
        ]

    for name, part, colour in groups:
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["log10_compute_flop"],
                y=part["benchmark_score_pct"],
                mode="markers",
                name=name,
                marker=dict(
                    size=10,
                    color=colour,
                    symbol=[ACCESS_SYMBOLS.get(a, "circle") for a in part["licence_category"]],
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                customdata=np.column_stack(
                    [part["model_name"], part["organization"].str.slice(0, 45),
                     part["training_compute_flop"], part["residual"], part["licence_category"]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[4]}<br>"
                    "Compute: %{customdata[2]:.2e} FLOP<br>GPQA Diamond: %{y:.1f}%<br>"
                    "vs trend: %{customdata[3]:+.1f} pp<extra></extra>"
                ),
            )
        )

    if trend is not None:
        xs = np.linspace(df["log10_compute_flop"].min(), df["log10_compute_flop"].max(), 50)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=trend["slope"] * xs + trend["intercept"],
                mode="lines",
                name=f"Trend: {trend['slope']:+.1f} pp per 10× compute",
                line=dict(color=TREND_COLOUR, width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    # Fan labels out sideways so neighbouring points don't stack their text; labels near
    # the top/bottom of the axis point inwards so they stay inside the plot.
    labels = _deviation_labels(df, label_n).sort_values("log10_compute_flop")
    for i, (_, r) in enumerate(labels.iterrows()):
        y = r["benchmark_score_pct"]
        going_up = r["residual"] > 0
        if y > 85:
            going_up = False
        elif y < 15:
            going_up = True
        fig.add_annotation(
            x=r["log10_compute_flop"], y=y,
            text=f"<b>{r['model_name']}</b> · {y:.0f}%",
            showarrow=True, arrowhead=0, arrowwidth=1,
            ax=(-70 if i % 2 == 0 else 70), ay=(-30 - 14 * (i % 2)) if going_up else (30 + 14 * (i % 2)),
            font=dict(size=11), arrowcolor=TREND_COLOUR,
            bgcolor="rgba(128,128,128,0.15)", borderpad=2,
        )

    _log_x_axis(fig, df["log10_compute_flop"].min(), df["log10_compute_flop"].max())
    _score_y_axis(fig)
    fig.update_layout(
        title=dict(text="Measured GPQA Diamond score vs reported training compute", x=0.0, xanchor="left"),
        height=560,
        margin=dict(l=60, r=10, t=90, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_bands(df: pd.DataFrame, bands: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for band in [b for b in BAND_LABELS if b in set(df["compute_band"])]:
        part = df[df["compute_band"] == band]
        fig.add_trace(
            go.Box(
                y=part["benchmark_score_pct"],
                name=f"{band}<br>n={len(part)}",
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker=dict(size=6, color="#6A47A8", opacity=0.6),
                line=dict(color="#3B1F78"),
                fillcolor="rgba(155,124,203,0.18)",
                customdata=part[["model_name"]],
                hovertemplate="<b>%{customdata[0]}</b><br>%{y:.1f}%<extra></extra>",
                showlegend=False,
            )
        )
    # Print each band's median on the chart so the key value is readable without hovering.
    for band, row in bands.iterrows():
        n = int((df["compute_band"] == band).sum())
        fig.add_annotation(
            x=f"{band}<br>n={n}", y=row["max"], text=f"median {row['median']:.0f}%",
            showarrow=False, yshift=14, font=dict(size=11),
        )
    _score_y_axis(fig)
    fig.update_xaxes(title="Training compute band (FLOP)")
    fig.update_layout(
        title=dict(text="Spread of GPQA Diamond scores within each order of magnitude of compute",
                   x=0.0, xanchor="left"),
        height=440, margin=dict(l=60, r=10, t=60, b=10), plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------- #
# Verification (for the video walkthrough)
# --------------------------------------------------------------------------- #
def _verify_band(view: pd.DataFrame, bands: pd.DataFrame) -> None:
    if bands.empty:
        st.info(f"No compute band has at least {MIN_PER_BAND} observations with the current filters.")
        return
    band = st.selectbox("Pick a box on the compute-band chart", bands.index.tolist(), key="tab2_verify_band")
    rows = view[view["compute_band"] == band]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(rows)}")
    c2.metric("Lowest score", f"{rows['benchmark_score_pct'].min():.1f}%")
    c3.metric("Median score", f"{rows['benchmark_score_pct'].median():.1f}%")
    c4.metric("Highest score", f"{rows['benchmark_score_pct'].max():.1f}%")

    table = rows[["model_name", "organization", "publication_date", "training_compute_flop",
                  "benchmark_score_pct"]].copy()
    table["publication_date"] = table["publication_date"].dt.date
    # Streamlit number columns cannot hold values above 2^53, so FLOP is shown as text.
    table["training_compute_flop"] = table["training_compute_flop"].map(lambda x: f"{x:.3e}")
    table.columns = ["Model", "Organisation", "Released", "Training compute (FLOP)", "GPQA Diamond (%)"]
    st.dataframe(
        table.sort_values("GPQA Diamond (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "GPQA Diamond (%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    if np.isclose(rows["benchmark_score_pct"].max() - rows["benchmark_score_pct"].min(),
                  bands.loc[band, "range"]):
        st.success("Verified: the box spans exactly the lowest and highest measured scores listed above.")
    else:
        st.error("Verification failed: the box range does not match the listed scores.")


def _trace_run(view: pd.DataFrame) -> None:
    labels = (view["model_name"] + " · " + view["model_version_id"].astype(str)).tolist()
    choice = st.selectbox("Pick a point to trace back to the raw Epoch AI benchmark file", labels,
                          key="tab2_trace")
    row = view.iloc[labels.index(choice)]

    st.markdown(
        "**Data lineage:** Epoch AI benchmark export → `raw_benchmarks.csv` → `prepared_data.py` → "
        "`benchmarks.csv` → GPQA Diamond filter → `gpqa.csv` → this tab "
        "(rows with reported compute)."
    )

    raw_path = _find_data_path("raw_benchmarks.csv")
    if raw_path is None:
        st.warning("`data/raw_benchmarks.csv` is not present; verify against the group repository.")
        return
    raw = _read_csv(str(raw_path))
    if "id_runs" not in raw.columns:
        st.warning("The raw file has no `id_runs` column — check the preprocessing script.")
        return
    match = raw[raw["id_runs"].astype(str) == str(row["benchmark_run_id"])]
    if match.empty:
        st.warning("No raw benchmark run matched this run ID.")
        return
    src = match.iloc[0]
    raw_score = pd.to_numeric(pd.Series([src.get("Best score (across scorers)")]), errors="coerce").iloc[0]
    if pd.isna(raw_score):
        raw_score = pd.to_numeric(pd.Series([src.get("best_score")]), errors="coerce").iloc[0]
    raw_compute = pd.to_numeric(pd.Series([src.get("Training compute (FLOP)")]), errors="coerce").iloc[0]

    st.dataframe(
        pd.DataFrame(
            {
                "Field": ["Run ID", "Model version", "Task", "Score", "Training compute (FLOP)"],
                "Raw file (raw_benchmarks.csv)": [
                    str(src.get("id_runs")), str(src.get("id_model_version")),
                    str(src.get("original_task_name", src.get("task"))),
                    "—" if pd.isna(raw_score) else f"{raw_score:.6f} (proportion)",
                    _sci(raw_compute),
                ],
                "Processed (gpqa.csv)": [
                    str(row["benchmark_run_id"]), str(row["model_version_id"]), row["benchmark_name"],
                    f"{row['benchmark_score_pct']:.4f}%", _sci(row["training_compute_flop"]),
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    ok_score = pd.notna(raw_score) and np.isclose(raw_score * 100, row["benchmark_score_pct"], rtol=1e-9)
    ok_compute = pd.notna(raw_compute) and np.isclose(raw_compute, row["training_compute_flop"], rtol=1e-9)
    if ok_score and ok_compute:
        st.success("Verified: score (× 100) and training FLOP match the raw Epoch AI benchmark run exactly.")
    else:
        st.error("Raw and processed values differ — check prepared_data.py before presenting this point.")


# --------------------------------------------------------------------------- #
# Tab
# --------------------------------------------------------------------------- #
def display_tab2(df: Optional[pd.DataFrame] = None) -> None:
    # ---- Header and user story ---------------------------------------------- #
    st.header("User Story 1 · Compute vs capability")
    st.markdown(
        "**As an AI Infrastructure Lead, I want to compare measured AI capability with training "
        "compute, so that I can assess whether higher-compute models are consistently associated "
        "with stronger performance.**"
    )
    st.caption(
        "Audience: AI Infrastructure Lead · Benchmark: GPQA Diamond · "
        "Compute: reported pre-training FLOP · Missing compute is excluded, not estimated."
    )

    # ---- Data --------------------------------------------------------------- #
    if df is None or not REQUIRED_COLUMNS.issubset(df.columns):
        path = _find_data_path("gpqa.csv")
        if path is None:
            st.error("Could not find `data/gpqa.csv`. Run `python3 prepared_data.py` first.")
            return
        df = _read_csv(str(path))
    gpqa_total = int(df["benchmark_name"].eq("GPQA diamond").sum()) if "benchmark_name" in df else len(df)
    try:
        runs = prepare_gpqa(df)
    except ValueError as err:
        st.error(str(err))
        return
    if runs.empty:
        st.warning("No GPQA Diamond results with measured training compute are available.")
        return

    # ---- Controls ----------------------------------------------------------- #
    y_min, y_max = int(runs["release_year"].min()), int(runs["release_year"].max())
    f1, f2, f3 = st.columns([2, 2, 2])
    years = f1.slider("Release years", y_min, y_max, (y_min, y_max), key="tab2_years")
    groups_available = sorted(runs["licence_category"].unique())
    groups = f2.multiselect("Model access", groups_available, default=groups_available, key="tab2_access")
    unit = f3.radio(
        "Unit of analysis",
        ["All evaluated versions", "Best version per model"],
        key="tab2_unit",
        help=(
            "Some models were evaluated in several versions (e.g. different reasoning settings) "
            "that share one training-compute estimate. 'Best version' keeps one point per model, "
            "to check the result is not driven by repeated models."
        ),
    )

    view = runs[runs["release_year"].between(*years) & runs["licence_category"].isin(groups)]
    if unit == "Best version per model":
        view = best_version_per_model(view)
    view = view.copy()
    trend = fit_trend(view)
    if trend is None:
        st.warning("Too few observations for this filter combination. Widen the year range or add an access group.")
        return
    view["residual"] = view["benchmark_score_pct"] - (
        trend["slope"] * view["log10_compute_flop"] + trend["intercept"]
    )
    bands = summarise_bands(view)

    # ---- KPI row ------------------------------------------------------------ #
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Results shown", f"{trend['n']}",
              help=f"Out of {gpqa_total} GPQA Diamond results; only those with disclosed training compute are used.")
    k2.metric("Rank correlation (ρ)", f"{trend['rho']:.2f}",
              help=f"Spearman correlation between compute and score; p = {trend['p_value']:.1e}. 1 = perfectly monotonic.")
    k3.metric("Per 10× compute", f"{trend['slope']:+.1f} pp",
              help="Slope of the log-linear trend line, in GPQA percentage points.")
    k4.metric("Widest gap in one band", "—" if bands.empty else f"{bands['range'].max():.0f} pp",
              help=f"Highest minus lowest score within one order of magnitude of compute "
                   f"(bands with ≥ {MIN_PER_BAND} results).")

    # ---- Key insight -------------------------------------------------------- #
    above = view.loc[view["residual"].idxmax()]
    below = view.loc[view["residual"].idxmin()]
    gap_text = (
        "" if bands.empty
        else f" Models with comparable compute can differ by up to **{bands['range'].max():.0f} points**."
    )
    st.info(
        f"**What the selected view shows:** across {trend['n']} results, higher training compute goes with "
        f"higher GPQA Diamond scores (**{trend['slope']:+.1f} pp per 10× compute**, Spearman ρ = "
        f"{trend['rho']:.2f}), but the trend explains only **{trend['r2']:.0%}** of the variation (R²)."
        f"{gap_text} {_name_date(above['model_name'], above['publication_date'])} sits "
        f"{above['residual']:+.0f} pp above the trend and "
        f"{_name_date(below['model_name'], below['publication_date'])} {below['residual']:+.0f} pp below it. "
        "Compute and capability are associated, but not consistently enough to choose a model on scale "
        "alone. This is an association, not a causal effect."
    )

    # ---- Chart 1: capability vs compute ------------------------------------ #
    st.subheader("1 · Capability vs training compute")
    c1, c2 = st.columns([1, 2])
    colour_by = c1.radio("Colour points by", ["Release year", "Model access"], horizontal=True,
                         key="tab2_colour")
    label_n = c2.slider("Label the largest deviations from trend (above and below)", 0, 5, 3,
                        key="tab2_label_n")
    st.plotly_chart(build_scatter(view, trend, colour_by, label_n), use_container_width=True,
                    config={"displayModeBar": False}, key="tab2_scatter")
    st.caption(
        "Figure: each point is one measured GPQA Diamond result. Trend = ordinary least-squares fit of "
        "GPQA score on log₁₀(training FLOP). Diamonds are closed / API-only models, circles are "
        "open-weight. Labelled models show their score on the chart; exact values for every point are "
        "listed in the tables below and in *Verify a displayed value*."
    )

    # ---- Chart 2: consistency within compute bands ------------------------- #
    st.subheader("2 · How consistent is it at comparable compute?")
    st.plotly_chart(build_bands(view, bands), use_container_width=True,
                    config={"displayModeBar": False}, key="tab2_bands")
    if not bands.empty:
        widest = bands["range"].idxmax()
        st.caption(
            f"Figure: each box is one order of magnitude of training compute; every result is shown as a "
            f"point and each band's median is printed above it. In the **{widest} FLOP** band, scores range "
            f"from **{bands.loc[widest, 'min']:.0f}%** to **{bands.loc[widest, 'max']:.0f}%**, a "
            f"{bands.loc[widest, 'range']:.0f}-point gap between models trained with comparable compute. "
            f"Bands with fewer than {MIN_PER_BAND} results are excluded from the spread statistics."
        )

    # ---- Supporting table (collapsed) -------------------------------------- #
    cols = {"model_name": "Model", "organization": "Organisation", "publication_date": "Released",
            "training_compute_flop": "Training compute (FLOP)", "benchmark_score_pct": "GPQA Diamond (%)",
            "residual": "vs trend (pp)"}
    col_cfg = {
        "GPQA Diamond (%)": st.column_config.NumberColumn(format="%.1f"),
        "vs trend (pp)": st.column_config.NumberColumn(format="%+.1f"),
    }

    def _dev_table(rows: pd.DataFrame) -> pd.DataFrame:
        out = rows[list(cols)].rename(columns=cols)
        out["Released"] = out["Released"].dt.date
        # Streamlit number columns cannot hold values above 2^53, so FLOP is shown as text.
        out["Training compute (FLOP)"] = out["Training compute (FLOP)"].map(lambda x: f"{x:.2e}")
        return out

    with st.expander("View models furthest from the trend"):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Above trend** — more capability than compute alone would suggest")
            st.dataframe(_dev_table(view.nlargest(5, "residual")), use_container_width=True,
                         hide_index=True, column_config=col_cfg)
        with t2:
            st.markdown("**Below trend** — less capability than compute alone would suggest")
            st.dataframe(_dev_table(view.nsmallest(5, "residual")), use_container_width=True,
                         hide_index=True, column_config=col_cfg)

    # ---- Verification (visible) -------------------------------------------- #
    st.subheader("3 · Verify a displayed value")
    v1, v2 = st.tabs(["Check a compute band", "Trace a point to source"])
    with v1:
        _verify_band(view, bands)
    with v2:
        _trace_run(view)

    st.markdown(f"Source: [Epoch AI — Capabilities & Benchmarking]({SOURCE_URL}) (CC BY attribution).")

    # ---- Interpretation, limitations and sources (collapsed) --------------- #
    with st.expander("Interpretation, limitations and sources"):
        st.markdown(
            f"""
- **What this shows:** how measured GPQA Diamond scores vary with training compute for models released from {years[0]} to {years[1]}, and how widely scores spread at comparable compute.
- **What this cannot show:** that more compute *causes* higher scores — this is an association. Release date, post-training and reasoning settings also differ between models.
- **Sample:** {len(runs)} of {gpqa_total} GPQA Diamond results have disclosed training compute; the rest are excluded, not imputed. Closed frontier models rarely disclose compute, so they are under-represented ({int((runs['licence_category'] == 'Closed / API-Only').sum())} closed results).
- **Repeated models:** some models appear several times (different evaluated versions sharing one compute estimate). Switch to *Best version per model* to check the conclusion holds.
- **Pre-training compute only:** reinforcement-learning / post-training compute and inference-time reasoning are not counted, which can place newer reasoning models above the trend.
- **Estimates:** many FLOP values are Epoch AI estimates; treat single points as approximate and read the overall pattern.
- **Ceiling effect:** GPQA Diamond tops out near 100%, so gains among the strongest models are compressed.
- **Access groups:** "Open-Weight" vs "Closed / API-Only" is the project's grouping, not a legal licence.
- **Measured-only rule:** benchmark scores and compute values are not imputed.
- **Source:** [Epoch AI — Capabilities & Benchmarking]({SOURCE_URL}) (CC BY attribution).
- **Reproducibility:** see `data/DATA_DICTIONARY.md` and `prepared_data.py`.
"""
        )


# Aliases for different main-app conventions.
def render(df: Optional[pd.DataFrame] = None) -> None:
    display_tab2(df)


def show(df: Optional[pd.DataFrame] = None) -> None:
    display_tab2(df)


if __name__ == "__main__":
    st.set_page_config(page_title="User Story 1 · Compute vs capability", layout="wide")
    display_tab2()
