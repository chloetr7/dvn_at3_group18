import os
import streamlit as st
import pandas as pd

# Import 5 module từ thư mục tabs/
from tabs import (
    tab1_macro_overview,
    tab2_scaling_correlation,
    tab3_algorithmic_shift,
    tab4_licence_distribution,
    tab5_cost_ranking,
)

st.set_page_config(
    page_title="AI Capability Gains: Compute vs Algorithms",
    layout="wide",
    initial_sidebar_state="expanded"
)

CVD_PALETTE = {
    "Open-Weight": "#0072B2",
    "Closed / API-Only": "#D55E00",
    "Unspecified": "#7F7F7F"
}

@st.cache_data
def load_data():
    path = "data/processed_ai_models.csv"
    if not os.path.exists(path):
        st.error("Missing 'data/processed_ai_models.csv'. Please run 'python prepare_data.py' first.")
        st.stop()
    return pd.read_csv(path)

df_full = load_data()

# ---- HEADER CHUNG LUÔN HIỂN THỊ TRÊN CÙNG (Checks 2, 20, 24) ----
st.title("The Rise of AI: Compute Scaling vs. Algorithmic Efficiency")
st.markdown(
    "### **Arguable Question:** *Have frontier AI capability gains since 2018 been driven primarily by brute-force compute scaling rather than algorithmic and parameter efficiency?*"
)

# 4 KPIs đúng trên TOÀN BỘ tập dữ liệu (Check 20: opens on something true about the whole dataset)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Whole-Dataset Models", f"{len(df_full):,} models")
k2.metric("Baseline Time Coverage", f"{int(df_full['release_year'].min())} – {int(df_full['release_year'].max())}")
k3.metric("Whole-Dataset Median Compute", f"10^{df_full['log10_compute_flop'].median():.1f} FLOP")
k4.metric("Whole-Dataset Median Size", f"{df_full['parameters_billions'].median():.2f} B params")

st.caption(
    "📌 **Sources & Refresh (Check 24):** Epoch AI Notable Models Database (CC BY 4.0) | "
    "Units: FLOP, Billions of Parameters ($10^9$), 2023 USD | **Last Refresh:** 18 October 2026."
)

# ---- BỘ LỌC CHUNG Ở SIDEBAR (Checks 21 & 22) ----
st.sidebar.header("🔍 Global Filters")
min_yr, max_yr = int(df_full["release_year"].min()), int(df_full["release_year"].max())

year_range = st.sidebar.slider("1. Release Year Range:", min_yr, max_yr, (2018, max_yr))
top_domains = df_full["primary_domain"].value_counts().head(6).index.tolist()
selected_domains = st.sidebar.multiselect("2. Primary Domain:", top_domains, default=top_domains[:3])
licence_opts = df_full["licence_Category"].unique().tolist()
selected_licences = st.sidebar.multiselect("3. Model Licence:", licence_opts, default=licence_opts)

df_filtered = df_full[
    (df_full["release_year"].between(year_range[0], year_range[1])) &
    (df_full["primary_domain"].isin(selected_domains)) &
    (df_full["licence_Category"].isin(selected_licences))
].copy()

# Check 22: Báo lỗi thân thiện nếu bộ lọc không có kết quả nào
if df_filtered.empty:
    st.warning("⚠️ **No models match your current filter criteria.** Please widen the year range or select more domains/licences in the sidebar.")
    st.stop()

# ---- KHỞI TẠO 5 TABS CHO 5 THÀNH VIÊN ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Macro Overview & Answer",
    "2. Story 1: Compute vs. Scale",
    "3. Story 2: Algorithmic Efficiency",
    "4. Story 3A: Open vs. Closed Spread",
    "5. Story 3B: Cost Ranking & Audit"
])

with tab1:
    tab1_macro_overview.render(df_filtered, df_full, CVD_PALETTE)
with tab2:
    tab2_scaling_correlation.render(df_filtered, df_full, CVD_PALETTE)
with tab3:
    tab3_algorithmic_shift.render(df_filtered, df_full, CVD_PALETTE)
with tab4:
    tab4_licence_distribution.render(df_filtered, df_full, CVD_PALETTE)
with tab5:
    tab5_cost_ranking.render(df_filtered, df_full, CVD_PALETTE)