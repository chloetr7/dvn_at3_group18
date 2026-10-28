"""DVN36104 Assignment 3 — Tab 4 (User Story 3A): How much compute does a capability level need?

User story 3: As an AI Infrastructure Lead, I want to identify models that achieve
strong measured capability with comparatively lower training compute, so that I can
evaluate more resource-efficient alternatives instead of selecting models based on
scale alone.

3A (this tab) asks *when* efficient alternatives appear: for a chosen GPQA Diamond
level, which model reached it first, which model has reached it with the least
reported training compute, and how quickly that minimum has fallen.
3B (tab5.py) then ranks every model that meets a requirement today.

Data: data/gpqa.csv (measured GPQA Diamond runs from Epoch AI). Only rows with a
measured score and positive reported training compute are used; nothing is imputed.

Run on its own from the repository root:
    streamlit run tabs/tab4.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SOURCE_URL = "https://epoch.ai/benchmarks/use-this-data"

# Same colour-blind-safe (Okabe-Ito) access colours as Tabs 1 and 2.
ACCESS_COLOURS = {"Open-Weight": "#E69F00", "Closed / API-Only": "#0072B2"}
CONTEXT_GREY = "#BDBDBD"
RECORD_COLOUR = "#009E73"  # bluish green: the "lowest compute so far" line
FIRST_COLOUR = "#7F7F7F"

ACCESS_OPTIONS = {
    "All models": None,
    "Open-weight only": "Open-Weight",
    "Closed / API-only": "Closed / API-Only",
}
SUMMARY_LEVELS = [40, 50, 60, 70, 80, 90, 95]  # GPQA Diamond levels compared in chart 2
ACCESS_NOUNS = {None: "model", "Open-Weight": "open-weight model", "Closed / API-Only": "closed / API-only model"}

REQUIRED_COLUMNS = {
    "benchmark_run_id", "model_version_id", "model_name", "benchmark_name",
    "benchmark_score_pct", "publication_date", "organization",
    "training_compute_flop", "has_training_compute", "licence_category",
}



# Data

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
    """Keep measured GPQA Diamond rows with positive reported training compute."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("gpqa.csv is missing required column(s): " + ", ".join(missing))

    out = df.copy()
    out["publication_date"] = pd.to_datetime(out["publication_date"], errors="coerce")
    out["training_compute_flop"] = pd.to_numeric(out["training_compute_flop"], errors="coerce")
    out["benchmark_score_pct"] = pd.to_numeric(out["benchmark_score_pct"], errors="coerce")
    has_compute = out["has_training_compute"].astype(str).str.lower().eq("true")
    out = out[
        out["benchmark_name"].eq("GPQA diamond")
        & has_compute
        & out["training_compute_flop"].gt(0)
        & out["benchmark_score_pct"].between(0, 100)
        & out["publication_date"].notna()
    ].copy()
    out["licence_category"] = out["licence_category"].fillna("Unspecified").astype(str)
    out["organization"] = out["organization"].fillna("Unspecified").astype(str)
    return out


def best_version_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: its best-scoring evaluated version (same rule as Tab 2).

    Versions of one model (e.g. different reasoning settings) share one training
    compute value, so keeping the best run asks "can this model reach the level?".
    """
    return (
        df.sort_values("benchmark_score_pct")
        .groupby(["model_name", "training_compute_flop"], as_index=False)
        .last()
        .reset_index(drop=True)
    )



# Analysis
def compute_records(models: pd.DataFrame, level: float) -> pd.DataFrame:
    """Models that reach `level`, in release order, flagged when they set a new
    lowest-training-compute record for that level."""
    qualifying = models[models["benchmark_score_pct"] >= level].copy()
    if qualifying.empty:
        return qualifying.assign(is_record=pd.Series(dtype=bool), record_so_far=pd.Series(dtype=float))
    # On a shared release date, the lower-compute model is counted first.
    qualifying = qualifying.sort_values(["publication_date", "training_compute_flop"]).reset_index(drop=True)
    qualifying["record_so_far"] = qualifying["training_compute_flop"].cummin()
    previous = qualifying["record_so_far"].shift(1, fill_value=np.inf)
    qualifying["is_record"] = qualifying["training_compute_flop"] < previous
    return qualifying


def summarise_level(qualifying: pd.DataFrame) -> Optional[dict]:
    """First model to reach the level vs the lowest-compute model to reach it."""
    if qualifying.empty:
        return None
    first = qualifying.iloc[0]
    cheapest = qualifying.loc[qualifying["training_compute_flop"].idxmin()]
    return {
        "n": len(qualifying),
        "first": first,
        "cheapest": cheapest,
        "factor": float(first["training_compute_flop"] / cheapest["training_compute_flop"]),
        "months": (cheapest["publication_date"] - first["publication_date"]).days / 30.44,
    }


def summary_table(models: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    rows = []
    for level in levels:
        s = summarise_level(compute_records(models, level))
        if s is None:
            continue
        rows.append({
            "level": level,
            "first_model": s["first"]["model_name"],
            "first_date": s["first"]["publication_date"],
            "first_flop": s["first"]["training_compute_flop"],
            "cheapest_model": s["cheapest"]["model_name"],
            "cheapest_date": s["cheapest"]["publication_date"],
            "cheapest_flop": s["cheapest"]["training_compute_flop"],
            "factor": s["factor"],
            "months": s["months"],
            "n": s["n"],
        })
    return pd.DataFrame(rows)


# Formatting

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sci(value: float) -> str:
    """2.5e24 -> '2.5 × 10²⁴'."""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10 ** exponent
    return f"{mantissa:.1f} × 10{str(exponent).translate(_SUP)}"


def _factor(x: float) -> str:
    return f"{x:,.0f}×" if x >= 10 else f"{x:.1f}×"


def _name_date(name: str, date: pd.Timestamp) -> str:
    """'Model (Mon YYYY)', unless the model name already carries a date."""
    return name if "(" in name else f"{name} ({date:%b %Y})"


def _months(m: float) -> str:
    m = round(m)
    return "same month" if m <= 0 else f"{m} month{'s' if m != 1 else ''} later"


def _log_x_axis(fig: go.Figure, lo: float, hi: float, title: str) -> None:
    exps = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    fig.update_xaxes(
        type="log", title=title,
        tickvals=[10 ** e for e in exps], ticktext=[f"10{str(e).translate(_SUP)}" for e in exps],
        gridcolor="rgba(128,128,128,0.2)",
    )


def _log_y_axis(fig: go.Figure, lo: float, hi: float, title: str) -> None:
    exps = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    fig.update_yaxes(
        type="log", title=title,
        tickvals=[10 ** e for e in exps], ticktext=[f"10{str(e).translate(_SUP)}" for e in exps],
        gridcolor="rgba(128,128,128,0.2)",
    )


# Charts

def build_record_chart(models: pd.DataFrame, qualifying: pd.DataFrame, level: int) -> go.Figure:
    """Change over time: every model's compute by release date, with the falling
    'lowest compute to reach the level' line."""
    fig = go.Figure()
    below = models[models["benchmark_score_pct"] < level]
    fig.add_trace(go.Scatter(
        x=below["publication_date"], y=below["training_compute_flop"], mode="markers",
        name=f"Below {level}% (context)",
        marker=dict(color=CONTEXT_GREY, size=7, opacity=0.6),
        customdata=below[["model_name", "benchmark_score_pct"]],
        hovertemplate="%{customdata[0]}<br>GPQA %{customdata[1]:.1f}%<br>%{y:.2e} FLOP<extra></extra>",
    ))
    for access, colour in ACCESS_COLOURS.items():
        part = qualifying[qualifying["licence_category"] == access]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["publication_date"], y=part["training_compute_flop"], mode="markers",
            name=f"Reaches {level}% · {access.lower()}",
            marker=dict(color=colour, size=10, line=dict(width=1, color="white"),
                        symbol="circle" if access == "Open-Weight" else "diamond"),
            customdata=part[["model_name", "benchmark_score_pct"]],
            hovertemplate="%{customdata[0]}<br>GPQA %{customdata[1]:.1f}%<br>%{y:.2e} FLOP<extra></extra>",
        ))

    # Step line: the lowest compute that has reached the level, as of each date.
    records = qualifying[qualifying["is_record"]]
    end = max(models["publication_date"].max(), records["publication_date"].max())
    step_x = list(records["publication_date"]) + [end]
    step_y = list(records["training_compute_flop"]) + [records["training_compute_flop"].iloc[-1]]
    fig.add_trace(go.Scatter(
        x=step_x, y=step_y, mode="lines", line=dict(color=RECORD_COLOUR, width=3, shape="hv"),
        name=f"Lowest compute to reach {level}% so far", hoverinfo="skip",
    ))
    # Label each record-setter so values are readable without hovering.
    positions = ["top right", "bottom right", "top right", "bottom right"]
    fig.add_trace(go.Scatter(
        x=records["publication_date"], y=records["training_compute_flop"], mode="markers+text",
        marker=dict(color=RECORD_COLOUR, size=13, symbol="star", line=dict(width=1, color="white")),
        text=[f"{r.model_name}<br>{_sci(r.training_compute_flop)} FLOP · {r.benchmark_score_pct:.0f}%"
              for r in records.itertuples()],
        # Alternate above/below; the latest record is labelled to its left so it stays inside the plot.
        textposition=[positions[i % len(positions)] if i < len(records) - 1 else "bottom left"
                      for i in range(len(records))],
        textfont=dict(size=11),
        name="New lowest-compute record", hoverinfo="skip",
    ))

    lo = models["training_compute_flop"].min()
    hi = models["training_compute_flop"].max()
    _log_y_axis(fig, lo, hi, "Training compute (FLOP, log scale)")
    fig.update_xaxes(title="Model release date", gridcolor="rgba(128,128,128,0.2)")
    fig.update_layout(
        title=dict(text=f"Lowest training compute that has reached ≥ {level}% on GPQA Diamond, over time"),
        height=470, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    return fig


def build_dumbbell(summary: pd.DataFrame) -> go.Figure:
    """Magnitude: compute of the first model to reach each level vs the lowest
    compute that has reached it since. Line length = the compute reduction."""
    s = summary.sort_values("level")
    labels = [f"≥ {lv}%" for lv in s["level"]]
    fig = go.Figure()
    for y, row in zip(labels, s.itertuples()):
        fig.add_trace(go.Scatter(
            x=[row.cheapest_flop, row.first_flop], y=[y, y], mode="lines",
            line=dict(color="rgba(128,128,128,0.6)", width=4), showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=s["first_flop"], y=labels, mode="markers+text", name="First model to reach the level",
        marker=dict(color=FIRST_COLOUR, size=13, symbol="square"),
        text=[_name_date(m, d) + ("" if f > 1 else " · no cheaper model yet")
              for m, d, f in zip(s["first_model"], s["first_date"], s["factor"])],
        textposition="middle right", textfont=dict(size=11), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=s["cheapest_flop"], y=labels, mode="markers+text", name="Lowest-compute model to reach it",
        marker=dict(color=RECORD_COLOUR, size=14, symbol="star"),
        text=[f"{_name_date(m, d)} · {_factor(f)} less" if f > 1 else ""
              for m, d, f in zip(s["cheapest_model"], s["cheapest_date"], s["factor"])],
        textposition="middle left", textfont=dict(size=11), hoverinfo="skip",
    ))
    lo = min(s["cheapest_flop"].min(), s["first_flop"].min())
    hi = max(s["cheapest_flop"].max(), s["first_flop"].max())
    # Extra room on both sides for the text labels.
    _log_x_axis(fig, lo / 3000, hi * 3000, "Training compute (FLOP, log scale)")
    fig.update_xaxes(range=[np.log10(lo) - 2.9, np.log10(hi) + 2.1])
    fig.update_yaxes(title="Minimum GPQA Diamond score", categoryorder="array", categoryarray=labels)
    fig.update_layout(
        title=dict(text="Compute of the first model to reach each level vs the least compute that has reached it since"),
        height=max(300, 60 * len(s) + 150), margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    return fig



# Verification

def _trace_to_source(records: pd.DataFrame) -> None:
    choice = st.selectbox(
        "Pick a record-setting model to trace",
        list(range(len(records))),
        format_func=lambda i: f"{records.iloc[i]['model_name']} ({records.iloc[i]['publication_date']:%Y-%m-%d})",
        key="tab4_trace",
    )
    row = records.iloc[choice]
    st.markdown(
        "**Lineage:** Epoch AI benchmark export → `data/raw_benchmarks.csv` → `prepared_data.py` → "
        "`data/benchmarks.csv` → GPQA Diamond filter → `data/gpqa.csv` → this tab "
        "(rows with reported compute; best run per model)."
    )
    st.write(f"Benchmark run ID (`benchmark_run_id` in gpqa.csv, `id_runs` in raw_benchmarks.csv): `{row['benchmark_run_id']}`")

    raw_path = _find_data_path("raw_benchmarks.csv")
    if raw_path is None:
        st.warning("`data/raw_benchmarks.csv` is not in this copy of the repository, so the raw lookup is skipped.")
        return
    raw = _read_csv(str(raw_path))
    match = raw[raw["id_runs"].astype(str) == str(row["benchmark_run_id"])]
    if match.empty:
        st.error("No raw row has this run ID. Check prepared_data.py before presenting this number.")
        return
    src = match.iloc[0]
    raw_score = pd.to_numeric(src.get("Best score (across scorers)"), errors="coerce")
    if pd.isna(raw_score):
        raw_score = pd.to_numeric(src.get("best_score"), errors="coerce")
    raw_flop = pd.to_numeric(src.get("Training compute (FLOP)"), errors="coerce")
    st.dataframe(pd.DataFrame({
        "Field": ["Model", "Benchmark", "Score", "Training compute (FLOP)"],
        "raw_benchmarks.csv": [str(src.get("Model")), str(src.get("original_task_name")),
                               f"{raw_score:.6f} (proportion)", f"{raw_flop:.4e}"],
        "Shown in this tab": [row["model_name"], row["benchmark_name"],
                              f"{row['benchmark_score_pct']:.4f}%", f"{row['training_compute_flop']:.4e}"],
    }), hide_index=True)
    score_ok = pd.notna(raw_score) and np.isclose(raw_score * 100, row["benchmark_score_pct"], atol=1e-6)
    flop_ok = pd.notna(raw_flop) and np.isclose(raw_flop, row["training_compute_flop"], rtol=1e-9)
    if score_ok and flop_ok:
        st.success("Match: raw score × 100 equals the score shown, and the training FLOP is identical.")
    else:
        st.error("Mismatch between raw and displayed values. Do not present this number until it is checked.")



# Page

def display_tab4(df: Optional[pd.DataFrame] = None) -> None:
    """Render Tab 4 / User Story 3A. Optionally pass the GPQA dataframe from app.py."""

    # Header / user story
    st.header("User Story 3A · Compute needed to reach a capability level")
    st.markdown(
        "**As an AI Infrastructure Lead, I want to identify models that achieve strong measured capability "
        "with comparatively lower training compute, so that I can evaluate more resource-efficient "
        "alternatives instead of selecting models based on scale alone.**"
    )
    st.markdown(
        "**Decision this tab supports:** pick the capability level you need. The tab shows which model reached "
        "it first, which model has since reached it with the least reported training compute, and how long that "
        "took, so you can judge whether a cheaper-to-train alternative is likely to exist or to appear soon."
    )

    try:
        if df is None:
            path = _find_data_path("gpqa.csv")
            if path is None:
                st.error("Could not find `data/gpqa.csv`. Run the app from the repository root.")
                return
            df = _read_csv(str(path))
        rows = prepare_gpqa(df)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Unable to load the GPQA data: {exc}")
        return

    st.caption(
        "Audience: AI Infrastructure Lead · Benchmark: GPQA Diamond (best run per model) · "
        "Compute: reported pre-training FLOP · "
        f"{rows['model_name'].nunique()} models with reported compute, released "
        f"{rows['publication_date'].min():%b %Y} – {rows['publication_date'].max():%b %Y} · "
        "Missing compute is excluded, not estimated."
    )

    # Decision controls
    c1, c2 = st.columns([3, 2])
    level = c1.slider("Capability level you need: minimum GPQA Diamond score (%)",
                      min_value=40, max_value=95, value=80, step=5, key="tab4_level",
                      help="GPQA Diamond: 198 graduate-level science questions. Random guessing scores 25%, "
                           "so levels start at 40% to stay clear of chance.")
    access_label = c2.radio("Which models can you use?", list(ACCESS_OPTIONS), horizontal=True, key="tab4_access")
    access = ACCESS_OPTIONS[access_label]

    models = best_version_per_model(rows)
    if access is not None:
        models = models[models["licence_category"] == access]
    if models.empty:
        st.warning("No models with reported training compute match this access filter. Choose another option.")
        return

    qualifying = compute_records(models, level)
    s = summarise_level(qualifying)

    # KPI row and key insight for the selected level
    if s is None:
        best = models.loc[models["benchmark_score_pct"].idxmax()]
        st.warning(
            f"No {ACCESS_NOUNS[access]} with reported training compute reaches "
            f"{level}% on GPQA Diamond. The highest here is {best['model_name']} at "
            f"{best['benchmark_score_pct']:.1f}%. Lower the level or widen the access filter."
        )
    else:
        k1, k2, k3, k4 = st.columns([1.1, 1.5, 0.8, 0.8])
        k1.metric(f"First to reach it: {_name_date(s['first']['model_name'], s['first']['publication_date'])}",
                  f"{_sci(s['first']['training_compute_flop'])} FLOP")
        k2.metric(f"Least compute to reach it: "
                  f"{_name_date(s['cheapest']['model_name'], s['cheapest']['publication_date'])}",
                  f"{_sci(s['cheapest']['training_compute_flop'])} FLOP")
        if s["factor"] > 1:
            k3.metric("Compute reduction", _factor(s["factor"]), _months(s["months"]), delta_color="off")
        else:
            k3.metric("Compute reduction", "None yet", "no cheaper model has reached it", delta_color="off")
        k4.metric("Models reaching the level", f"{s['n']} of {len(models)}")

        if s["factor"] > 1:
            st.info(
                f"**What the selected level shows:** {s['first']['model_name']} first reached {level}% in "
                f"{s['first']['publication_date']:%B %Y} with {_sci(s['first']['training_compute_flop'])} FLOP. "
                f"{s['cheapest']['model_name']} reached it {_months(s['months'])} with "
                f"{_sci(s['cheapest']['training_compute_flop'])} FLOP: **{_factor(s['factor'])} less training "
                "compute** for the same measured level. This is an observed association, not proof of why "
                "the later model needed less compute."
            )
        else:
            st.info(
                f"**What the selected level shows:** no later model has reached {level}% with less training "
                f"compute than {s['first']['model_name']} ({_sci(s['first']['training_compute_flop'])} FLOP). "
                "At this level, the data show no cheaper-to-train alternative yet."
            )

    # 1 · Main evidence: record over time
    if s is not None:
        st.subheader(f"1 · The least compute that has reached ≥ {level}%, over time")
        st.plotly_chart(build_record_chart(models, qualifying, level), key="tab4_chart1",
                        config={"displayModeBar": False})
        st.caption(
            f"Figure: training compute (log scale) of every model by release date. Coloured points reach "
            f"≥ {level}% on GPQA Diamond; grey points score below it and are shown for context. Green stars mark "
            "each time a model reached the level with less training compute than any earlier model, and the green "
            "step line is that lowest value as of each date. Record-setters are labelled with their values."
        )

    # 2 · Main evidence: reduction at each level
    st.subheader("2 · How much less compute, at each capability level?")
    summary = summary_table(models, SUMMARY_LEVELS)
    if summary.empty:
        st.warning("None of the comparison levels (40–95%) is reached under this access filter.")
    else:
        st.plotly_chart(build_dumbbell(summary), key="tab4_chart2", config={"displayModeBar": False})
        st.caption(
            "Figure: grey square = training compute of the first model to reach each GPQA Diamond level; green "
            "star = the least compute any model has needed to reach it since. The longer the bar, the more the "
            f"compute needed for that level has fallen. Access filter: {access_label.lower()}. "
            "Log scale: each gridline is 10× more compute."
        )
        table = summary.rename(columns={
            "level": "GPQA level (%)", "first_model": "First to reach", "first_date": "First date",
            "first_flop": "First: training FLOP", "cheapest_model": "Least compute to reach",
            "cheapest_date": "Least-compute date", "cheapest_flop": "Least: training FLOP",
            "factor": "Reduction (×)", "months": "Months between", "n": "Models reaching level",
        })
        table["First date"] = table["First date"].dt.date
        table["Least-compute date"] = table["Least-compute date"].dt.date
        with st.expander("View the underlying values in chart 2"):
            st.dataframe(table, hide_index=True, column_config={
                "First: training FLOP": st.column_config.NumberColumn(format="%.2e"),
                "Least: training FLOP": st.column_config.NumberColumn(format="%.2e"),
                "Reduction (×)": st.column_config.NumberColumn(format="%.1f"),
                "Months between": st.column_config.NumberColumn(format="%.0f"),
            })

        # Answer across all levels
        big = summary[summary["factor"] > 1]
        top = summary.loc[summary["level"].idxmax()]
        top_share = (models["training_compute_flop"] < top["first_flop"]).mean()
        if top_share >= 0.75:
            top_text = (
                f"But the highest level here (≥ {top['level']}%) was first reached by {top['first_model']} with "
                f"{_sci(top['first_flop'])} FLOP, more than {top_share:.0%} of the models in this view. "
                "New top capability still arrived with large compute first: scale appears to lead and "
                "efficiency to follow."
            )
        else:
            top_text = (
                f"The highest level here (≥ {top['level']}%) was first reached by {top['first_model']} with "
                f"{_sci(top['first_flop'])} FLOP, more than only {top_share:.0%} of the models in this view, "
                "so in this view the top level was not bought with the largest compute."
            )
        if not big.empty:
            st.info(
                f"**Answer from this tab:** at {len(big)} of the {len(summary)} levels compared, a later model "
                f"reached the same GPQA Diamond score with less training compute than the first model to get there, "
                f"between {_factor(big['factor'].min())} and {_factor(big['factor'].max())} less, with a median gap "
                f"of {big['months'].median():.0f} months. Efficiency gains follow once a level is reached. " + top_text
            )
        else:
            st.info("**Answer from this tab:** under this filter, no level has been reached more cheaply after "
                    "it was first reached.")

    # 3 · Verification
    st.subheader("3 · Verify a displayed value")
    if s is None or qualifying[qualifying["is_record"]].empty:
        st.info("Choose a level that at least one model reaches to trace a record-setting model back to source.")
    else:
        _trace_to_source(qualifying[qualifying["is_record"]].reset_index(drop=True))
    st.markdown(f"Source: [Epoch AI — Capabilities & Benchmarking]({SOURCE_URL}) (CC BY attribution).")

    # Interpretation / limits
    with st.expander("Interpretation, limitations and sources"):
        st.markdown(
            f"""
- **What this shows:** for each GPQA Diamond level, the first model to reach it and the least reported pre-training compute any model has needed to reach it since, among models with published compute.
- **What this cannot show:** *why* a later model needed less compute. Better data, distillation from a larger model, architecture and post-training may all contribute; these are associations, not causal effects.
- **Distilled models hide their teacher's compute:** a small model trained on a larger model's outputs reports only its own training FLOP.
- **Inference and total cost:** training FLOP says nothing about serving cost, latency or price.
- **Missingness matters:** many GPQA results (often from closed labs) have no published compute and are excluded, so some levels may have been reached earlier or more cheaply than shown.
- **One benchmark:** GPQA Diamond measures graduate-level science questions, not all capabilities.
- **Best-run rule:** each model is represented by its best-scoring GPQA run (same rule as User Story 1), since versions of one model share one training-compute value.
- **Measured-only rule:** benchmark scores and compute values are not imputed.
- **Source:** [Epoch AI — Capabilities & Benchmarking]({SOURCE_URL}).
- **Reproducibility:** see `data/DATA_DICTIONARY.md` and `prepared_data.py` for the documented source fields and transforms.
            """
        )


def render(df: Optional[pd.DataFrame] = None) -> None:
    display_tab4(df)


def show(df: Optional[pd.DataFrame] = None) -> None:
    display_tab4(df)


if __name__ == "__main__":
    st.set_page_config(page_title="User Story 3A · Compute needed to reach a capability level", page_icon="📊",
                       layout="wide", initial_sidebar_state="collapsed")
    display_tab4()
