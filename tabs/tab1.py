"""DVN36104 Assignment 3 — Tab 1: Macro overview of AI training compute.

Question answered: how quickly has the training compute behind notable AI models
grown since 2018, and does the trend differ between open-weight and closed / API
models?

Data: the group's processed ``data/ai_models.csv`` (Epoch AI — Notable AI Models).
Only measured source values are used; nothing is imputed.

Use inside the main app:   tab1.display_tab1()   (aliases: render(), show())
Preview on its own:        streamlit run tabs/tab1.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SOURCE_URL = "https://epoch.ai/data/ai-models"
MIN_MODELS_PER_YEAR = 3
ACCESS_COLOURS = {"Open-Weight": "#E69F00", "Closed / API-Only": "#0072B2"}
TREND_COLOUR = "#9A9A9A"  # mid-grey: visible on light and dark themes
REQUIRED_COLUMNS = {
    "model_name",
    "organization",
    "publication_date",
    "release_year",
    "training_compute_flop",
    "log10_compute_flop",
    "parameters",
    "licence_category",
    "primary_domain",
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


def prepare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Type-convert and keep rows with positive measured compute and a valid date."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("ai_models.csv is missing required column(s): " + ", ".join(missing))

    out = df.copy()
    out["publication_date"] = pd.to_datetime(out["publication_date"], errors="coerce")
    for col in ("training_compute_flop", "log10_compute_flop", "parameters", "release_year"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[
        out["publication_date"].notna()
        & out["training_compute_flop"].gt(0)
        & out["log10_compute_flop"].notna()
    ].copy()

    out["release_year"] = out["publication_date"].dt.year.astype(int)
    out["licence_category"] = out["licence_category"].fillna("Unspecified").astype(str)
    out["primary_domain"] = out["primary_domain"].fillna("Unspecified").astype(str)
    out["organization"] = out["organization"].fillna("Unspecified").astype(str)
    # Decimal year, used as the x-variable of the log-linear trend.
    out["decimal_year"] = (
        out["publication_date"].dt.year
        + (out["publication_date"].dt.dayofyear - 1) / 365.25
    )
    return out.sort_values("publication_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def fit_trend(df: pd.DataFrame) -> Optional[dict]:
    """OLS fit of log10(compute) on decimal year. Returns None if too few points."""
    if len(df) < 5 or df["decimal_year"].nunique() < 2:
        return None
    slope, intercept = np.polyfit(df["decimal_year"], df["log10_compute_flop"], 1)
    fitted = slope * df["decimal_year"] + intercept
    ss_res = float(((df["log10_compute_flop"] - fitted) ** 2).sum())
    ss_tot = float(((df["log10_compute_flop"] - df["log10_compute_flop"].mean()) ** 2).sum())
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "growth_per_year": float(10 ** slope),
        "doubling_months": float(12 * np.log10(2) / slope) if slope > 0 else np.nan,
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": int(len(df)),
    }


def summarise_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Median compute per release year and access group (years with >= MIN models)."""
    summary = (
        df.groupby(["release_year", "licence_category"], as_index=False)
        .agg(
            n_models=("model_name", "size"),
            median_log10_compute=("log10_compute_flop", "median"),
            max_log10_compute=("log10_compute_flop", "max"),
        )
        .sort_values(["licence_category", "release_year"])
    )
    summary["median_compute_flop"] = 10 ** summary["median_log10_compute"]
    return summary[summary["n_models"] >= MIN_MODELS_PER_YEAR].reset_index(drop=True)


def _sci(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2e} FLOP"


def _big_factor(x: float) -> str:
    """Readable growth multiple, e.g. 3.3 million×, instead of false precision."""
    for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if x >= size:
            return f"{x / size:.1f} {word}×"
    return f"{x:.0f}×"


def _sup(exponent: int) -> str:
    return str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _log_axis(fig: go.Figure, y_min: float, y_max: float) -> None:
    """Show the log10 axis as readable powers of ten (10¹⁸, 10²⁰ ...)."""
    lo, hi = int(np.floor(y_min)), int(np.ceil(y_max))
    step = 2 if hi - lo > 6 else 1
    ticks = list(range(lo, hi + 1, step))
    fig.update_yaxes(
        tickvals=ticks,
        ticktext=[f"10{_sup(t)}" for t in ticks],
        range=[lo - 0.3, hi + 0.3],
        title="Training compute (FLOP, log scale)",
        gridcolor="rgba(128,128,128,0.2)",
    )


def build_scatter(df: pd.DataFrame, trend: Optional[dict], highlight: list[str]) -> go.Figure:
    fig = go.Figure()
    for group, colour in {**ACCESS_COLOURS, "Unspecified": "#999999"}.items():
        part = df[df["licence_category"] == group]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["publication_date"],
                y=part["log10_compute_flop"],
                mode="markers",
                name=f"{group} (n={len(part)})",
                marker=dict(size=8, color=colour, opacity=0.75, line=dict(width=0.5, color="white")),
                customdata=np.column_stack(
                    [part["model_name"], part["organization"], part["training_compute_flop"],
                     part["primary_domain"]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                    "Released: %{x|%d %b %Y}<br>Compute: %{customdata[2]:.2e} FLOP<br>"
                    "Domain: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    if trend is not None:
        xs = np.linspace(df["decimal_year"].min(), df["decimal_year"].max(), 50)
        dates = pd.to_datetime((xs - 1970) * 365.25, unit="D", origin="unix")
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=trend["slope"] * xs + trend["intercept"],
                mode="lines",
                name=f"Trend: ×{trend['growth_per_year']:.1f} per year",
                line=dict(color=TREND_COLOUR, width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    # Label the selected models so the story can point at concrete examples.
    for name in highlight:
        row = df[df["model_name"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        fig.add_annotation(
            x=r["publication_date"], y=r["log10_compute_flop"], text=f"<b>{name}</b>",
            showarrow=True, arrowhead=0, ax=-40, ay=-28, font=dict(size=11),
            arrowcolor="#9A9A9A",
        )

    _log_axis(fig, df["log10_compute_flop"].min(), df["log10_compute_flop"].max())
    fig.update_xaxes(title="Publication date", gridcolor="rgba(128,128,128,0.2)")
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_yearly(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for group, colour in ACCESS_COLOURS.items():
        part = summary[summary["licence_category"] == group]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["release_year"],
                y=part["median_log10_compute"],
                mode="lines+markers",
                name=group,
                line=dict(color=colour, width=3),
                marker=dict(size=9, color=colour),
                customdata=np.column_stack([part["n_models"], part["median_compute_flop"]]),
                hovertemplate=(
                    f"<b>{group}</b><br>Year: %{{x}}<br>Median compute: %{{customdata[1]:.2e}} FLOP"
                    "<br>Models: %{customdata[0]}<extra></extra>"
                ),
            )
        )
    if not summary.empty:
        _log_axis(fig, summary["median_log10_compute"].min(), summary["median_log10_compute"].max())
    fig.update_yaxes(title="Median training compute (FLOP, log scale)")
    fig.update_xaxes(title="Release year", dtick=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------- #
# Verification (for the video walkthrough)
# --------------------------------------------------------------------------- #
def _verify_year(view: pd.DataFrame, summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("No year/access group has enough models to verify with the current filters.")
        return
    options = [f"{int(r.release_year)} · {r.licence_category}" for r in summary.itertuples()]
    choice = st.selectbox("Pick a point on the yearly-median chart", options, index=len(options) - 1)
    year_text, group = choice.split(" · ", 1)
    displayed = summary[
        (summary["release_year"] == int(year_text)) & (summary["licence_category"] == group)
    ].iloc[0]
    rows = view[(view["release_year"] == int(year_text)) & (view["licence_category"] == group)]
    recalculated = float(rows["training_compute_flop"].median())

    c1, c2, c3 = st.columns(3)
    c1.metric("Displayed median", _sci(displayed["median_compute_flop"]))
    c2.metric("Recalculated from raw FLOP", _sci(recalculated))
    c3.metric("Underlying models", f"{len(rows):,}")

    table = rows[["model_name", "organization", "publication_date", "training_compute_flop"]].copy()
    table["publication_date"] = table["publication_date"].dt.date
    table.columns = ["Model", "Organisation", "Publication date", "Training compute (FLOP)"]
    st.dataframe(
        table.sort_values("Training compute (FLOP)"),
        use_container_width=True,
        hide_index=True,
        column_config={"Training compute (FLOP)": st.column_config.NumberColumn(format="%.3e")},
    )
    # Median of log10 values equals log10 of the median for odd n; for even n the
    # chart's geometric midpoint can differ slightly from the arithmetic median.
    if np.isclose(displayed["median_compute_flop"], recalculated, rtol=1e-6):
        st.success("Verified: the plotted median equals the median of the underlying models' FLOP.")
    elif len(rows) % 2 == 0:
        st.info(
            "Even number of models: the chart plots the median on the log scale (geometric mean of the "
            "two middle models), so it differs slightly from the arithmetic median above. "
            "Both middle models are listed in the table."
        )
    else:
        st.error("Verification failed: plotted and recalculated medians differ — check prepared_data.py.")


def _trace_model(view: pd.DataFrame) -> None:
    names = view.sort_values("training_compute_flop", ascending=False)["model_name"].unique().tolist()
    name = st.selectbox("Pick a model to trace back to the raw Epoch AI file", names)
    row = view[view["model_name"] == name].iloc[0]

    raw_path = _find_data_path("raw_ai_models.csv")
    if raw_path is None:
        st.warning("`data/raw_ai_models.csv` is not present; verify against the group repository.")
        return
    raw = _read_csv(str(raw_path))
    if "Model" not in raw.columns:
        st.warning("The raw file has no `Model` column — check the preprocessing script.")
        return
    raw_dates = pd.to_datetime(raw.get("Publication date"), errors="coerce")
    match = raw[(raw["Model"].astype(str) == name) & (raw_dates == row["publication_date"])]
    if match.empty:
        match = raw[raw["Model"].astype(str) == name]
    if match.empty:
        st.warning("No raw record matched this model name.")
        return
    src = match.iloc[0]
    raw_compute = pd.to_numeric(pd.Series([src.get("Training compute (FLOP)")]), errors="coerce").iloc[0]
    raw_params = pd.to_numeric(pd.Series([src.get("Parameters")]), errors="coerce").iloc[0]

    st.dataframe(
        pd.DataFrame(
            {
                "Field": ["Model", "Publication date", "Training compute (FLOP)", "Parameters",
                          "Accessibility"],
                "Raw file (raw_ai_models.csv)": [
                    str(src.get("Model")), str(src.get("Publication date")),
                    _sci(raw_compute), "—" if pd.isna(raw_params) else f"{raw_params:.4g}",
                    str(src.get("Model accessibility")),
                ],
                "Processed (ai_models.csv)": [
                    row["model_name"], str(row["publication_date"].date()),
                    _sci(row["training_compute_flop"]), f"{row['parameters']:.4g}",
                    f"{row.get('accessibility', '—')} → {row['licence_category']}",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    ok_compute = pd.notna(raw_compute) and np.isclose(raw_compute, row["training_compute_flop"], rtol=1e-9)
    ok_params = pd.notna(raw_params) and np.isclose(raw_params, row["parameters"], rtol=1e-9)
    if ok_compute and ok_params:
        st.success("Verified: training FLOP and parameter count match the raw Epoch AI record exactly.")
    else:
        st.error("Raw and processed values differ — check prepared_data.py before presenting this model.")


# --------------------------------------------------------------------------- #
# Tab
# --------------------------------------------------------------------------- #
def display_tab1(df: Optional[pd.DataFrame] = None) -> None:
    st.header("Macro overview · How fast is AI training compute growing?")
    st.markdown(
        "Every dot is a notable AI model released since 2018. The vertical axis is the "
        "compute used to train it, on a **log scale** — each gridline is a 10× or 100× jump. "
        "This tab sets the scene for the user stories that follow: before asking *what* "
        "compute buys, we show *how much* more of it the field is using each year."
    )

    if df is None:
        path = _find_data_path("ai_models.csv")
        if path is None:
            st.error("Could not find `data/ai_models.csv`. Run `python3 prepared_data.py` first.")
            return
        df = _read_csv(str(path))
    try:
        models = prepare_models(df)
    except ValueError as err:
        st.error(str(err))
        return
    if models.empty:
        st.warning("No models with measured training compute are available.")
        return

    # ---- Filters ----------------------------------------------------------- #
    y_min, y_max = int(models["release_year"].min()), int(models["release_year"].max())
    f1, f2, f3 = st.columns([2, 2, 2])
    years = f1.slider("Release years", y_min, y_max, (y_min, y_max), key="tab1_years")
    groups_available = sorted(models["licence_category"].unique())
    groups = f2.multiselect("Model access", groups_available, default=groups_available, key="tab1_access")
    top_domains = models["primary_domain"].value_counts().index.tolist()
    domain = f3.selectbox("Primary domain", ["All domains"] + top_domains, key="tab1_domain")

    view = models[
        models["release_year"].between(*years) & models["licence_category"].isin(groups)
    ]
    if domain != "All domains":
        view = view[view["primary_domain"] == domain]
    view = view.copy()
    if view.empty:
        st.warning("No models match these filters. Widen the year range or add an access group.")
        return

    trend = fit_trend(view)
    top = view.loc[view["training_compute_flop"].idxmax()]

    # ---- Headline numbers --------------------------------------------------- #
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Models shown", f"{len(view):,}")
    k2.metric("Compute growth", "—" if trend is None else f"×{trend['growth_per_year']:.1f} / year")
    k3.metric(
        "Doubling time",
        "—" if trend is None or np.isnan(trend["doubling_months"]) else f"{trend['doubling_months']:.1f} months",
    )
    k4.metric("Largest training run", f"10{_sup(int(np.floor(top['log10_compute_flop'])))} FLOP",
              help=f"{top['model_name']} ({top['organization']}): {_sci(top['training_compute_flop'])}")

    # ---- Chart 1: every model ------------------------------------------------ #
    st.subheader("1 · Training compute of every notable model")
    default_labels = (
        view.sort_values("training_compute_flop").groupby("release_year").tail(1)["model_name"].tolist()
    )
    highlight = st.multiselect(
        "Label models on the chart (default: the largest run each year)",
        view["model_name"].sort_values().unique().tolist(),
        default=default_labels[-4:],
        key="tab1_labels",
    )
    st.plotly_chart(build_scatter(view, trend, highlight), use_container_width=True)
    if trend is not None:
        first, last = view["publication_date"].min(), view["publication_date"].max()
        span_years = (last - first).days / 365.25
        st.markdown(
            f"**Reading the chart:** across the {trend['n']} models shown, training compute grew by "
            f"roughly **×{trend['growth_per_year']:.1f} per year** — doubling about every "
            f"**{trend['doubling_months']:.1f} months**. Over the {span_years:.1f} years shown, that "
            f"compounds to about **{_big_factor(trend['growth_per_year'] ** span_years)}**. "
            f"The fitted line explains {trend['r2']:.0%} of the variation in log-compute (R²)."
        )
    st.caption(
        "Trend = ordinary least-squares fit of log₁₀(training FLOP) on release date. "
        "A log scale is used because compute spans more than ten orders of magnitude; on a linear "
        "axis every model before the last two years would sit on the floor."
    )

    # ---- Chart 2: open vs closed ------------------------------------------- #
    st.subheader("2 · Open-weight vs closed / API models")
    summary = summarise_by_year(view)
    st.plotly_chart(build_yearly(summary), use_container_width=True)

    gap_rows = []
    for group in ACCESS_COLOURS:
        part = view[view["licence_category"] == group]
        t = fit_trend(part)
        if t is not None:
            gap_rows.append(
                {"Access group": group, "Models": t["n"],
                 "Growth per year": f"×{t['growth_per_year']:.1f}",
                 "Doubling time (months)": round(t["doubling_months"], 1),
                 "Median compute": _sci(part["training_compute_flop"].median())}
            )
    if gap_rows:
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Yearly medians are shown only for year/group combinations with at least "
        f"{MIN_MODELS_PER_YEAR} models, so a single model cannot drive the line. "
        "Medians are taken on the log scale, which is robust to the very largest runs."
    )

    # ---- Verification ------------------------------------------------------ #
    st.subheader("3 · Verify the numbers")
    v1, v2 = st.tabs(["Check a yearly median", "Trace a model to source"])
    with v1:
        _verify_year(view, summary)
    with v2:
        _trace_model(view)

    st.markdown(f"Source: [Epoch AI — Notable AI Models]({SOURCE_URL}) (CC BY attribution).")

    with st.expander("Interpretation, limitations and sources"):
        st.markdown(
            f"""
- **What this shows:** the training compute reported for notable AI models released from {years[0]} to {years[1]}, and the average exponential growth rate across them.
- **What this cannot show:** why compute grew, or whether more compute *caused* better models — later tabs look at capability (GPQA Diamond) against compute.
- **Selection bias:** Epoch AI lists *notable* models, and only models with published training compute and parameter counts are kept. Frontier closed models often withhold these figures, so the largest runs may be under-represented.
- **Estimates:** many FLOP values are Epoch AI estimates from reported hardware and training time; treat single points as approximate to within a few× and read the trend, not individual dots.
- **Access groups:** "Open-Weight" vs "Closed / API-Only" is the project's grouping of Epoch's accessibility field, not a legal licence.
- **Measured-only rule:** no compute or parameter values are imputed.
- **Source:** [Epoch AI — Notable AI Models]({SOURCE_URL}).
- **Reproducibility:** see `data/DATA_DICTIONARY.md` and `prepared_data.py`.
"""
        )


# Aliases for different main-app conventions.
def render(df: Optional[pd.DataFrame] = None) -> None:
    display_tab1(df)


def show(df: Optional[pd.DataFrame] = None) -> None:
    display_tab1(df)


if __name__ == "__main__":
    st.set_page_config(page_title="Tab 1 · Macro overview", layout="wide")
    display_tab1()
