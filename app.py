"""DVN36104 Assignment 3 — Group 18 main app.

Run from the repository root:
    streamlit run app.py

Each tab lives in tabs/tabN.py and loads its own data from data/. This file only
sets up the page, shows the question / persona / headline on the first screen,
and calls each tab. A tab that raises an error shows a message inside its own
tab instead of breaking the whole app.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI capability: compute or efficiency?",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from tabs import tab1, tab2, tab3, tab4, tab5  # noqa: E402  (after set_page_config)

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_URL = "https://epoch.ai/data/ai-models"
BENCH_URL = "https://epoch.ai/benchmarks/use-this-data"

QUESTION = (
    "To what extent have recent frontier AI capability gains depended on increasing "
    "training compute versus improvements in compute efficiency?"
)
PERSONA = (
    "An **AI Infrastructure Lead** at a mid-sized technology company, deciding how much to invest "
    "in increasingly compute-intensive models versus more compute-efficient alternatives."
)

# (tab label, module, entry-point function name)
TABS = [
    ("Overview · Compute growth", tab1, "display_tab1"),
    ("US1 · Compute vs capability", tab2, "display_tab2"),
    ("US2 · Capability at similar compute", tab3, "display_tab3"),
    ("US3A · Compute to reach a level", tab4, "display_tab4"),
    ("US3B · Efficient model choices", tab5, "render"),
]


# --------------------------------------------------------------------------- #
# Headline numbers for the first screen (computed from the data, not typed in)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def headline() -> dict | None:
    try:
        models = pd.read_csv(DATA_DIR / "ai_models.csv", low_memory=False)
        gpqa = pd.read_csv(DATA_DIR / "gpqa.csv", low_memory=False)
    except OSError:
        return None

    models["publication_date"] = pd.to_datetime(models["publication_date"], errors="coerce")
    models = models[models["training_compute_flop"].gt(0) & models["publication_date"].notna()]
    year = models["publication_date"].dt.year + (models["publication_date"].dt.dayofyear - 1) / 365.25
    slope, _ = np.polyfit(year, np.log10(models["training_compute_flop"]), 1)

    gpqa["publication_date"] = pd.to_datetime(gpqa["publication_date"], errors="coerce")
    has_compute = gpqa["has_training_compute"].astype(str).str.lower().eq("true")
    g = gpqa[gpqa["benchmark_name"].eq("GPQA diamond") & has_compute & gpqa["training_compute_flop"].gt(0)]
    in_band = g[g["log10_compute_flop"].between(24, 25, inclusive="left")]["benchmark_score_pct"]

    return {
        "n_models": len(models),
        "first_year": int(models["publication_date"].dt.year.min()),
        "growth": 10 ** slope,
        "n_gpqa": len(g),
        "band_min": in_band.min(),
        "band_max": in_band.max(),
        "latest": max(models["publication_date"].max(), g["publication_date"].max()),
    }


# --------------------------------------------------------------------------- #
# First screen
# --------------------------------------------------------------------------- #
st.title(QUESTION)
st.markdown(f"**Who this is for:** {PERSONA}")

h = headline()
if h is not None:
    st.info(
        f"**Headline:** across {h['n_models']} notable AI models released since {h['first_year']}, "
        f"training compute grew about **×{h['growth']:.1f} per year**. Yet among {h['n_gpqa']} "
        f"GPQA Diamond results with disclosed compute, models trained with similar compute "
        f"(10²⁴–10²⁵ FLOP) scored anywhere from **{h['band_min']:.0f}% to {h['band_max']:.0f}%**. "
        "More compute goes with stronger models, but it is not the whole story: the tabs below "
        "show where efficiency gains appear."
    )
    st.caption(
        f"Data: Epoch AI — [Notable AI Models]({MODELS_URL}) and "
        f"[Capabilities & Benchmarking]({BENCH_URL}) (CC BY). "
        f"Latest model release in the data: {h['latest']:%d %b %Y}. "
        "See `data/DATA_DICTIONARY.md` for sources and processing."
    )
else:
    st.warning("Data files not found in `data/`. Run the app from the repository root.")

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
for container, (label, module, func_name) in zip(st.tabs([t[0] for t in TABS]), TABS):
    with container:
        func = getattr(module, func_name, None) or getattr(module, "render", None)
        if func is None:
            st.info(f"{label}: coming soon.")
            continue
        try:
            func()
        except Exception as exc:  # keep the rest of the app usable if one tab fails
            st.error(f"This tab could not be displayed: {type(exc).__name__}: {exc}")
