import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EE Analytics Dashboard", layout="wide")

# ------------------------------
# Data loading & preparation
# ------------------------------
@st.cache_data
def load_data(path: str = "dashboard_curated.csv") -> pd.DataFrame:
    """Load curated CSV and perform light, safe parsing for the app."""
    df = pd.read_csv(path)

    # Parse key dates (coerce errors to NaT)
    for c in ["Programme Start Date", "Programme End Date", "Run_Month"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Friendly label for charts (if not already present)
    if "Run_Month" in df.columns:
        df["Run_Month_Label"] = df["Run_Month"].dt.strftime("%Y-%m")

    # Derive Age_Group (with Unknown for NaNs)
    if "Age" in df.columns:
        age_num = pd.to_numeric(df["Age"], errors="coerce")
        bins   = [0, 34, 44, 54, 64, 200]
        labels = ["<35", "35–44", "45–54", "55–64", "65+"]
        df["Age_Group"] = pd.cut(age_num, bins=bins, labels=labels, right=True)
        df["Age_Group"] = df["Age_Group"].astype("string")
        df.loc[age_num.isna(), "Age_Group"] = "Unknown"

    # Light category cleanup for common slicers (as strings to keep Plotly/Streamlit happy)
    cat_cols = [
        "Application Status", "Applicant Type", "Primary Category",
        "Secondary Category", "Seniority", "Gender", "Country Of Residence",
        "Truncated Programme Name", "Domain"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # Normalize Gender Unknown visibility
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].fillna("Unknown").str.capitalize()

    # Organization cleanup (collapse placeholders to "UNKNOWN")
    org_col = "Organisation Name: Organisation Name"
    if org_col in df.columns:
        df[org_col] = (
            df[org_col]
            .fillna("Unknown")
            .astype("string")
            .str.strip()
            .str.upper()
            .replace({
                "N.A.": "UNKNOWN",
                "N. A": "UNKNOWN",
                "NIL": "UNKNOWN",
                "NA": "UNKNOWN"
            })
        )

    return df


# Allow user to upload a newer CSV (optional)
uploaded = st.sidebar.file_uploader("Upload a curated CSV (optional)", type=["csv"])
data_path = uploaded if uploaded is not None else "dashboard_curated.csv"

# Load the full dataset once; df is a working copy for filters
df_full = load_data(data_path)
df = df_full.copy()

if df.empty:
    st.info("No data loaded. Please upload a CSV or place dashboard_curated.csv next to this script.")
    st.stop()

st.title("Executive Education Analytics Dashboard")


# ------------------------------
# Session keys & helpers
# ------------------------------
def _safe_key(label: str, suffix: str) -> str:
    """Make safe, unique Streamlit widget keys from labels."""
    return f"{label}_{suffix}".replace(" ", "_").lower()

# Filter labels used throughout
FILTER_LABELS = [
    "Application Status",
    "Applicant Type",
    "Primary Category",
    "Secondary Category",
    "Country of Residence",
    "Seniority",
    "Domain"
]

def multiselect_with_all_button(label: str, series: pd.Series, default_all: bool = True):
    """
    Multiselect with a single 'Select all' button (no per-filter clear).
    - Persists selection in session_state
    - Handles option changes safely (e.g., after date filter)
    """
    s = series.copy().fillna("Unknown")
    options = sorted(s.unique().tolist())

    ms_key = _safe_key(label, "multi")

    # Initialize once per filter
    if ms_key not in st.session_state:
        st.session_state[ms_key] = options[:] if default_all else []

    # Keep selection only within current options (if options changed)
    current = [v for v in st.session_state[ms_key] if v in options]
    if current != st.session_state[ms_key]:
        st.session_state[ms_key] = current

    # One button to select all current options
    if st.button(f"Select all {label}", key=_safe_key(label, "btn_all")):
        st.session_state[ms_key] = options[:]  # Streamlit auto-reruns

    # Render multiselect bound to state (no default=)
    st.multiselect(label, options, key=ms_key)

    return st.session_state[ms_key]


def apply_filter(series: pd.Series, selected: list[str]) -> pd.Series:
    """
    Smart filter logic:
    - If nothing selected: keep all rows
    - If user selected all current options: keep all rows
    - Else: keep rows whose value is in selected (with NaN -> 'Unknown')
    """
    if selected is None:
        return pd.Series(True, index=series.index)
    s = series.fillna("Unknown")
    all_opts = set(s.unique().tolist())
    sel_set  = set(selected or [])
    if not selected or sel_set == all_opts:
        return pd.Series(True, index=series.index)
    return s.isin(selected)


# ------------------------------
# Global date span (from df_full)
# ------------------------------
if "Run_Month" in df_full.columns:
    full_min = pd.to_datetime(df_full["Run_Month"], errors="coerce").min()
    full_max = pd.to_datetime(df_full["Run_Month"], errors="coerce").max()

    # Persist the true full span across reruns
    if "run_month_full_span" not in st.session_state:
        st.session_state["run_month_full_span"] = (full_min.date(), full_max.date())

    # Initialize current range
    if "run_month_range" not in st.session_state:
        st.session_state["run_month_range"] = st.session_state["run_month_full_span"]


# ------------------------------
# Sidebar: Global buttons & date range
# ------------------------------
with st.sidebar:
    st.header("Filters")

    def select_all_filters():
        """Select all options for each filter and reset date to full span."""
        for label in FILTER_LABELS:
            ms_key = _safe_key(label, "multi")
            # Use the current (date-filtered) df to decide options; change to df_full for absolute all
            series = df[label].fillna("Unknown") if label in df.columns else pd.Series([], dtype="object")
            st.session_state[ms_key] = sorted(series.unique().tolist())
        # Reset date to the stored full span
        if "run_month_full_span" in st.session_state:
            st.session_state["run_month_range"] = st.session_state["run_month_full_span"]

    def clear_all_filters():
        """Clear selections for each filter and reset date to full span."""
        for label in FILTER_LABELS:
            ms_key = _safe_key(label, "multi")
            st.session_state[ms_key] = []
        if "run_month_full_span" in st.session_state:
            st.session_state["run_month_range"] = st.session_state["run_month_full_span"]

    c1, c2 = st.columns(2)
    with c1:
        st.button("✅ Select all filters", key="btn_select_all_filters", on_click=select_all_filters)
    with c2:
        st.button("🧹 Clear all filters", key="btn_clear_all_filters", on_click=clear_all_filters)

    # Date range picker (bound to state), then apply to working df
    if "Run_Month" in df.columns:
        full_min_date, full_max_date = st.session_state["run_month_full_span"]
        st.date_input(
            "Run month range",
            key="run_month_range",
            value=st.session_state["run_month_range"],
            min_value=full_min_date,
            max_value=full_max_date,
        )
        start_d, end_d = map(pd.to_datetime, st.session_state["run_month_range"])
        df = df[(df["Run_Month"] >= start_d) & (df["Run_Month"] <= end_d)]

    # Per-filter multiselects with "Select all" buttons
    sel_status   = multiselect_with_all_button("Application Status", df.get("Application Status", pd.Series([], dtype="object")))
    sel_app_type = multiselect_with_all_button("Applicant Type",     df.get("Applicant Type",     pd.Series([], dtype="object")))
    sel_primcat  = multiselect_with_all_button("Pri Category",   df.get("Primary Category",   pd.Series([], dtype="object")))
    sel_secncat  = multiselect_with_all_button("Sec Category", df.get("Secondary Category", pd.Series([], dtype="object")))
    sel_country  = multiselect_with_all_button("Country",            df.get("Country Of Residence", pd.Series([], dtype="object")))
    sel_senior   = multiselect_with_all_button("Seniority",          df.get("Seniority",          pd.Series([], dtype="object")))
    sel_domain   = multiselect_with_all_button("Domain",          df.get("Domain",          pd.Series([], dtype="object")))
    
    top_k = st.number_input("Top K (for Top-X charts)", min_value=3, max_value=50, value=10, step=1)


# ------------------------------
# Apply all filters to the working df
# ------------------------------
mask = (
    apply_filter(df.get("Application Status",        pd.Series(index=df.index)), sel_status)   &
    apply_filter(df.get("Applicant Type",            pd.Series(index=df.index)), sel_app_type) &
    apply_filter(df.get("Primary Category",          pd.Series(index=df.index)), sel_primcat)  &
    apply_filter(df.get("Secondary Category",        pd.Series(index=df.index)), sel_secncat)  &
    apply_filter(df.get("Country Of Residence",      pd.Series(index=df.index)), sel_country)  &
    apply_filter(df.get("Seniority",                 pd.Series(index=df.index)), sel_senior)   &
    apply_filter(df.get("Domain",                    pd.Series(index=df.index)), sel_domain)
) 
df_f = df[mask].copy()
st.caption(f"Filtered rows: {len(df_f):,} of {len(df):,}")


# ------------------------------
# Tabs & Visualizations
# ------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Time Series", "🗺️ Geography", "🏷️ Programmes × Country",
    "👔 Titles & Orgs", "🧮 Age & Demographics", "ℹ️ Data Preview", "Age Distribution per Category", "Country Distribution per Category"
])

# --- Tab 1: Time Series
with tab1:
    st.subheader("Participants over Time")
    if "Run_Month" in df_f.columns:
        ts = df_f.groupby("Run_Month").size().reset_index(name="Participants")
        ts = ts.sort_values("Run_Month")
        fig = px.line(ts, x="Run_Month", y="Participants", markers=True)
        fig.update_layout(yaxis_title="Participants", xaxis_title="Run Month")
        st.plotly_chart(fig, use_container_width=True)

    # Seasonal bars by Start Month / Quarter
    if "Programme Start Date" in df_f.columns:
        df_f["Start_Month_Num"] = df_f["Programme Start Date"].dt.month
        df_f["Start_Quarter"]   = df_f["Programme Start Date"].dt.quarter

        col_a, col_b = st.columns(2)
        with col_a:
            mon = df_f.groupby("Start_Month_Num").size().reset_index(name="Applications")
            mon["Month"] = mon["Start_Month_Num"].map({
                1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
            })
            mon = mon.sort_values("Start_Month_Num")
            figm = px.bar(mon, x="Month", y="Applications", title="Applications by Start Month")
            st.plotly_chart(figm, use_container_width=True)

        with col_b:
            q = df_f.groupby("Start_Quarter").size().reset_index(name="Applications")
            figq = px.bar(q, x="Start_Quarter", y="Applications", title="Applications by Start Quarter")
            st.plotly_chart(figq, use_container_width=True)

# --- Tab 2: Geography
with tab2:
    st.subheader("Geospatial: Participants by Country")
    if "Country Of Residence" in df_f.columns:
        geo = df_f.groupby("Country Of Residence").size().reset_index(name="Participants")
        if not geo.empty:
            fig = px.choropleth(
                geo, locations="Country Of Residence", locationmode="country names",
                color="Participants", hover_name="Country Of Residence", color_continuous_scale="Viridis"
            )
            fig.update_layout(coloraxis_colorbar_title="Participants")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top Countries over Time (adjustable K)**")
        top_countries = geo.nlargest(top_k, "Participants")["Country Of Residence"].tolist()
        ts_geo = (
            df_f[df_f["Country Of Residence"].isin(top_countries)]
            .groupby(["Run_Month", "Country Of Residence"]).size().reset_index(name="Participants")
            .sort_values("Run_Month")
        )
        if not ts_geo.empty:
            fig = px.line(ts_geo, x="Run_Month", y="Participants", color="Country Of Residence", markers=True)
            fig.update_layout(yaxis_title="Participants", xaxis_title="Run Month")
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Programmes × Country
with tab3:
    st.subheader("Top Programmes & Country Breakdown")
    prog_col = "Truncated Programme Name"
    if prog_col in df_f.columns and "Country Of Residence" in df_f.columns:
        top_progs = df_f[prog_col].value_counts().nlargest(top_k).index.tolist()
        df_top = df_f[df_f[prog_col].isin(top_progs)].copy()

        agg = df_top.groupby([prog_col, "Country Of Residence"]).size().reset_index(name="Participants")
        top_c_in_subset = agg.groupby("Country Of Residence")["Participants"].sum().nlargest(top_k).index.tolist()
        agg = agg[agg["Country Of Residence"].isin(top_c_in_subset)]

        fig = px.bar(
            agg, x=prog_col, y="Participants", color="Country Of Residence",
            title=f"Participants by Programme (Top {top_k}) and Country", barmode="stack"
        )
        fig.update_layout(xaxis_title="Programme (Anon)", yaxis_title="Participants")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Titles & Organisations
with tab4:
    st.subheader("Top Job Titles & Organisations")
    col1, col2 = st.columns(2)

    if "Job Title Clean" in df_f.columns:
        top_titles = df_f["Job Title Clean"].value_counts().nlargest(top_k).reset_index()
        top_titles.columns = ["Job Title", "Participants"]
        fig1 = px.bar(top_titles, x="Participants", y="Job Title", orientation="h",
                      title=f"Top {top_k} Job Titles")
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

    org_col = "Organisation Name: Organisation Name"
    if org_col in df_f.columns:
        top_orgs = df_f[org_col].value_counts().nlargest(top_k).reset_index()
        top_orgs.columns = ["Organisation", "Participants"]
        fig2 = px.bar(top_orgs, x="Participants", y="Organisation", orientation="h",
                      title=f"Top {top_k} Organisations")
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

    if "Seniority" in df_f.columns:
        sen = df_f["Seniority"].value_counts().reset_index()
        sen.columns = ["Seniority", "Participants"]
        fig3 = px.bar(sen, x="Seniority", y="Participants", title="Participants by Seniority")
        st.plotly_chart(fig3, use_container_width=True)
        
    if "Domain" in df_f.columns:
        domain = df_f["Domain"].value_counts().reset_index()
        domain.columns = ["Domain", "Participants"]
        domain = domain[domain["Domain"] != "Unknown"]
        fig4 = px.bar(domain, x="Domain", y="Participants", title="Participants by Job Title Domain")
        st.plotly_chart(fig4, use_container_width=True)
        
# --- Tab 5: Age & Demographics
with tab5:
    st.subheader("Demographics")
    c1, c2 = st.columns(2)

    if "Age_Group" in df_f.columns:
        agec = df_f["Age_Group"].value_counts().reindex(
            ["<35","35–44","45–54","55–64","65+","Unknown"]
        )
        agec = agec.dropna().reset_index()
        agec.columns = ["Age Group", "Participants"]
        fig = px.bar(agec, x="Age Group", y="Participants", title="Participants by Age Group")
        st.plotly_chart(fig, use_container_width=True)

    if "Gender" in df_f.columns:
        gender = df_f["Gender"].value_counts().reset_index()
        gender.columns = ["Gender", "Participants"]
        fig = px.pie(gender, names="Gender", values="Participants", title="Gender Split")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 6: Data Preview
with tab6:
    st.subheader("Filtered Data Preview")
    preview_cols = [c for c in [
        "Application ID", "Contact ID", "Application Status", "Applicant Type",
        "Organisation Name: Organisation Name", "Job Title Clean", "Seniority",
        "Truncated Programme Name", "Truncated Programme Run", "Primary Category", "Secondary Category",
        "Programme Start Date", "Programme End Date", "Run_Month", "Run_Month_Label", "Programme_Duration",
        "Gender", "Age", "Country Of Residence", "Domain"
    ] if c in df_f.columns]

    st.dataframe(
        df_f.sort_values("Run_Month").loc[:, preview_cols].head(500),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_export.csv",
        mime="text/csv"
    )


# --- Tab 7: Test Tab
with tab7:
    st.header("Age Distribution per Category")

    # Choose between Primary or Secondary Category
    category_type = st.radio("Choose category type:", ["Primary Category", "Secondary Category"])

    # Get available categories
    if category_type in df_f.columns and "Age" in df_f.columns:
        categories = df_f[category_type].dropna().unique()
        selected_cat = st.selectbox(f"Select {category_type}:", categories)
        bins = np.arange(10, 80, 5)  # bins: 0-5, 5-10, ..., 85-90

        # Plot for selected category
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.histplot(df_f[df_f[category_type] == selected_cat]["Age"], bins=bins, kde=True, ax=ax)
        ax.set_title(f"Age Distribution - {category_type}: {selected_cat}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_xlim(10, 80)
        plt.tight_layout()
        st.pyplot(fig)


# --- Tab 8: Age & Country Dist. per Category
with tab8:
    st.header("Country Distribution per Category")

    # Choose between Primary or Secondary Category
    country_category_type = st.radio("Choose category type for country distribution:", ["Primary Category", "Secondary Category"], key="country_cat_type")

    if country_category_type in df_f.columns and "Country Of Residence" in df_f.columns:
        categories = df_f[country_category_type].dropna().unique()
        selected_cat = st.selectbox(f"Select {country_category_type}:", categories, key="country_cat_select")

        # Plot for selected category
        subset = df_f[df_f[country_category_type] == selected_cat]
        country_props = subset["Country Of Residence"].value_counts(normalize=True).astype(float)
        major = country_props[country_props >= 0.01]
        others_pct = 1 - major.sum()
        if others_pct > 0:
            major["Others"] = others_pct
        plot_df = major.reset_index()
        plot_df.columns = ["Country", "Percentage"]
        plot_df["Percentage"] = plot_df["Percentage"] * 100
        plot_df = plot_df.sort_values(by="Percentage", ascending=False)
        if "Others" in plot_df["Country"].values:
            others_row = plot_df[plot_df["Country"] == "Others"]
            plot_df = plot_df[plot_df["Country"] != "Others"]
            plot_df = pd.concat([plot_df, others_row], ignore_index=True)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.barplot(x="Country", y="Percentage", data=plot_df, hue="Country", dodge=False, order=plot_df["Country"], ax=ax, legend=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"Country Distribution (%) - {country_category_type}: {selected_cat}")
        ax.set_xlabel("Country")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        # for i, v in enumerate(plot_df["Percentage"]):
        #     ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
        st.pyplot(fig)
    
