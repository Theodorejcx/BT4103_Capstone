import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
# Matplotlib and Seaborn are no longer used in this version but kept in case of other uses
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EE Analytics Dashboard", layout="wide")

# ------------------------------
# Data loading & preparation
# ------------------------------
@st.cache_data
def load_data(path: str = "dashboard_curated_v2.csv") -> pd.DataFrame:
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
        bins    = [0, 34, 44, 54, 64, 200]
        labels  = ["<35", "35–44", "45–54", "55–64", "65+"]
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
data_path = uploaded if uploaded is not None else "dashboard_curated_v2.csv"

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

# ---- UI label ↔ column name mapping ----
COL_MAP = {
    "Pri Category": "Primary Category",
    "Sec Category": "Secondary Category",
    "Country": "Country Of Residence",
    "Application Status": "Application Status",
    "Applicant Type": "Applicant Type",
    "Seniority": "Seniority",
    "Domain": "Domain",
}
UI_FILTER_LABELS = ["Application Status", "Applicant Type", "Pri Category",
                    "Sec Category", "Country", "Seniority", "Domain"]

def _col_from_label(label: str) -> str:
    return COL_MAP.get(label, label)  # fall back if label == col

def multiselect_with_all_button(label: str, df_source: pd.DataFrame, default_all: bool = True):
    """Multiselect that shows a short label but filters on the real dataframe column."""
    col = _col_from_label(label)
    s = df_source.get(col, pd.Series([], dtype="object")).copy().fillna("Unknown")
    options = sorted(s.unique().tolist())

    ms_key = _safe_key(label, "multi")

    if ms_key not in st.session_state:
        st.session_state[ms_key] = options[:] if default_all else []

    current = [v for v in st.session_state[ms_key] if v in options]
    if current != st.session_state[ms_key]:
        st.session_state[ms_key] = current

    if st.button(f"Select all {label}", key=_safe_key(label, "btn_all")):
        st.session_state[ms_key] = options[:]

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
        for label in UI_FILTER_LABELS:
            ms_key = _safe_key(label, "multi")
            col = _col_from_label(label)
            series = df.get(col, pd.Series([], dtype="object")).fillna("Unknown")
            st.session_state[ms_key] = sorted(series.unique().tolist())
        if "run_month_full_span" in st.session_state:
            st.session_state["run_month_range"] = st.session_state["run_month_full_span"]

    def clear_all_filters():
        for label in UI_FILTER_LABELS:
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
    sel_status   = multiselect_with_all_button("Application Status", df)
    sel_app_type = multiselect_with_all_button("Applicant Type", df)
    sel_primcat  = multiselect_with_all_button("Pri Category", df)
    sel_secncat  = multiselect_with_all_button("Sec Category", df)
    sel_country  = multiselect_with_all_button("Country", df)
    sel_senior   = multiselect_with_all_button("Seniority", df)
    sel_domain   = multiselect_with_all_button("Domain", df)
    top_k = st.number_input("Top K (for Top-X charts)", min_value=3, max_value=50, value=10, step=1)


# ------------------------------
# Apply all filters to the working df
# ------------------------------
mask = (
    apply_filter(df.get(_col_from_label("Application Status"), pd.Series(index=df.index)), sel_status) &
    apply_filter(df.get(_col_from_label("Applicant Type"),     pd.Series(index=df.index)), sel_app_type) &
    apply_filter(df.get(_col_from_label("Pri Category"),       pd.Series(index=df.index)), sel_primcat) &
    apply_filter(df.get(_col_from_label("Sec Category"),       pd.Series(index=df.index)), sel_secncat) &
    apply_filter(df.get(_col_from_label("Country"),            pd.Series(index=df.index)), sel_country) &
    apply_filter(df.get(_col_from_label("Seniority"),          pd.Series(index=df.index)), sel_senior) &
    apply_filter(df.get(_col_from_label("Domain"),             pd.Series(index=df.index)), sel_domain)
)
df_f = df[mask].copy()
st.caption(f"Filtered rows: {len(df_f):,} of {len(df):,}")


# ------------------------------
# Tabs & Visualizations
# ------------------------------
tab1, tab2, tab3, tab4, tab5, tab_6, tab_7, tab_8, tab_9 = st.tabs([
    "📈 Time Series",
    "🗺️ Geography",
    "🏷️ Programmes × Country",
    "👔 Titles & Orgs",
    "🧮 Age & Demographics",
    "🧭 Category Insights",
    "💰 Programme Cost", # <-- NEW TAB
    "🎯 Programme Deep Dive",
    "ℹ️ Data Preview",
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
        # Aggregate by country
        geo = (
            df_f.groupby("Country Of Residence")
                .size()
                .reset_index(name="Participants")
        )

        exclude_sg = st.checkbox("Exclude Singapore (reduce skew)", value=False, key="geo_exclude_sg")
        geo_plot = geo[geo["Country Of Residence"] != "Singapore"] if exclude_sg else geo.copy()

        if geo_plot.empty:
            st.info("No data to display for current filters.")
        else:
            # ---- Bubble size: sqrt scale for visual stability
            size = np.sqrt(geo_plot["Participants"].astype(float).clip(lower=0))
            size = 10 + (size / size.max()) * 30 if size.max() > 0 else np.full_like(size, 10, dtype=float)
            geo_plot["BubbleSize"] = size

            # ---- Robust color scaling: cap at 95th percentile
            color_values = geo_plot["Participants"].astype(float)
            cmin = float(color_values.min())
            cmax = float(np.quantile(color_values, 0.95))
            if cmax <= cmin:
                cmax = cmin + 1.0  # avoid zero range

            # ---- Proportional-symbol map 
            fig = px.scatter_geo(
                geo_plot,
                locations="Country Of Residence",
                locationmode="country names",
                size="BubbleSize",  # already scaled
                color="Participants",
                hover_name="Country Of Residence",
                hover_data={"Participants": True, "BubbleSize": False},
                color_continuous_scale="Viridis",
                projection="natural earth",
            )
            fig.update_traces(marker=dict(sizemode="area", line=dict(width=0.5, color="rgba(0,0,0,0.25)")))
            fig.update_layout(
                coloraxis_colorbar_title="Participants",
                coloraxis_cmin=cmin,
                coloraxis_cmax=cmax,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Pareto (Top-K) bar - uses global top_k ----
    st.markdown("**Pareto of Countries (Top K)**")

    if 'geo_plot' in locals():
        k = int(top_k)
        # Total participants in the shown set (respects exclude_sg)
        total_cnt = float(geo_plot["Participants"].sum()) if not geo_plot.empty else 0.0

        if total_cnt == 0:
            st.info("No countries to display for the current filters.")
        else:
            top_countries = geo_plot.nlargest(k, "Participants").copy()
            top_countries["Share_%"] = (top_countries["Participants"] / total_cnt) * 100.0

            fig_bar = px.bar(
                top_countries,
                x="Country Of Residence",
                y="Participants",
                title=f"Top {k} Countries by Participants",
                text=top_countries["Share_%"].round(1).astype(str) + "%",
            )
            fig_bar.update_traces(textposition="outside", cliponaxis=False)
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="Participants",
                xaxis_title="Country"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Friendly caption about scope
            sg_note = " (Singapore excluded)" if exclude_sg else ""
            st.caption(
                f"Total participants shown: {int(total_cnt):,}{sg_note}. "
                f"Top {k} countries account for {top_countries['Share_%'].sum():.1f}% of the shown total."
            )
    else:
        st.info("Internal error: geo_plot not found — ensure the map section defines 'geo_plot'.")

# --- Tab 3: Programmes × Country
with tab3:
    st.subheader("Top Programmes & Country Breakdown (Heatmap: % Participants)")
    prog_col = "Truncated Programme Name"
    if prog_col in df_f.columns and "Country Of Residence" in df_f.columns:
        top_progs = df_f[prog_col].value_counts().nlargest(top_k).index.tolist()
        df_top = df_f[df_f[prog_col].isin(top_progs)].copy()

        agg = df_top.groupby([prog_col, "Country Of Residence"]).size().reset_index(name="Participants")
        top_c_in_subset = agg.groupby("Country Of Residence")["Participants"].sum().nlargest(top_k).index.tolist()
        agg = agg[agg["Country Of Residence"].isin(top_c_in_subset)]

        # Pivot for heatmap
        heatmap_data = agg.pivot(index=prog_col, columns="Country Of Residence", values="Participants").fillna(0)
        total = heatmap_data.values.sum()
        if total > 0:
            heatmap_pct = (heatmap_data / total * 100).round(2)  # percentage per cell
        else:
            heatmap_pct = heatmap_data # Avoid division by zero

        fig = px.imshow(
            heatmap_pct,
            labels=dict(x="Country Of Residence", y="Programme", color="Percentage (%)"),
            x=heatmap_pct.columns,
            y=heatmap_pct.index,
            color_continuous_scale="Viridis",
            aspect="auto",
            title=f"Participants Heatmap (%): Programme × Country (Top {top_k})",
            text_auto=True,  # Show percentage in each cell
        )
        fig.update_layout(xaxis_title="Country Of Residence", yaxis_title="Programme (Anon)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap: Top k Countries × Primary Category ---
    st.subheader(f"Distribution of Top {top_k} Countries Across Primary Categories")
    if ("Primary Category" in df_f.columns) and ("Country Of Residence" in df_f.columns):
        top_countries = df_f["Country Of Residence"].value_counts().nlargest(10).index.tolist()
        df_top_cty = df_f[df_f["Country Of Residence"].isin(top_countries)].copy()
        agg_cat = (
            df_top_cty.groupby(["Country Of Residence", "Primary Category"])
            .size()
            .reset_index(name="Participants")
        )
        heatmap_cat = agg_cat.pivot(index="Country Of Residence", columns="Primary Category", values="Participants").fillna(0)
        # Normalize each row by country total
        heatmap_cat_pct = heatmap_cat.div(heatmap_cat.sum(axis=1), axis=0) * 100
        heatmap_cat_pct = heatmap_cat_pct.round(2)
        fig_cat = px.imshow(
            heatmap_cat_pct,
            labels=dict(x="Primary Category", y="Country Of Residence", color="Row %"),
            x=heatmap_cat_pct.columns,
            y=heatmap_cat_pct.index,
            color_continuous_scale="Viridis",
            aspect="auto",
            title="For each country: % of participants in each Primary Category",
            text_auto=True
        )
        fig_cat.update_layout(xaxis_title="Primary Category", yaxis_title="Country Of Residence")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Required columns not found: ensure ‘Primary Category’ and ‘Country Of Residence’ exist in the dataset.")

# --- Tab 4: Titles & Organisations
with tab4:
    st.subheader("Top Job Titles & Organisations")

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

# --- Tab 6: Age & Demographics: nested tabs ---
with tab_6:
    st.subheader("Category Insights")
    sub_age, sub_country = st.tabs([
        "📊 Age Distribution per Category",
        "🌍 Country Distribution per Category"
    ])

    # -----------------------------
    # Sub-tab A: Age Distribution
    # -----------------------------
    with sub_age:
        st.markdown("##### Age Distribution per Category")
        cat_type = st.radio(
            "Choose category type:",
            ["Primary Category", "Secondary Category"],
            key="age_cat_type",
            horizontal=True
        )
        cat_col = cat_type

        if (cat_col in df_f.columns) and ("Age" in df_f.columns):
            cat_values = (
                df_f[cat_col].fillna("Unknown").astype(str).replace({"": "Unknown"}).unique().tolist()
            )
            # Put 'Unknown' last for UX
            cat_values = [v for v in sorted(cat_values) if v != "Unknown"] + (
                ["Unknown"] if "Unknown" in cat_values else []
            )

            selected_cat = st.selectbox(f"Select {cat_type}:", cat_values, key="age_cat_select")

            subset = df_f[df_f[cat_col].fillna("Unknown").astype(str) == selected_cat].copy()
            if subset.empty:
                st.info("No rows for this selection.")
            else:
                # Ensure Age_Group exists
                if "Age_Group" not in subset.columns:
                    age_num = pd.to_numeric(subset["Age"], errors="coerce")
                    bins    = [0, 34, 44, 54, 64, 200]
                    labels = ["<35", "35–44", "45–54", "55–64", "65+"]
                    subset["Age_Group"] = pd.cut(age_num, bins=bins, labels=labels, right=True).astype("string")
                    subset.loc[age_num.isna(), "Age_Group"] = "Unknown"

                # Build % distribution
                age_series = subset["Age_Group"].fillna("Unknown").astype(str).replace({"Unknown": "Not provided"})
                dist = (age_series.value_counts(normalize=True) * 100.0).reset_index()
                dist.columns = ["Age Group", "Percentage"]

                include_unknown_age = st.checkbox(
                    "Include Unknown ages",
                    value=False,
                    key="include_unknown_age"
                )

                if not include_unknown_age:
                    dist = dist[dist["Age Group"] != "Not provided"].copy()
                    total = float(dist["Percentage"].sum())
                    if total > 0:
                        dist["Percentage"] = dist["Percentage"] * (100.0 / total)

                # Quality note
                unknown_age_pct = float(subset["Age"].isna().mean() * 100.0)
                st.caption(f"Data quality note: Unknown ages = {unknown_age_pct:.1f}% of rows for this selection.")

                # Order buckets
                order_full = ["<35", "35–44", "45–54", "55–64", "65+", "Not provided"]
                order_used = [g for g in order_full if g in dist["Age Group"].values]
                dist = dist.set_index("Age Group").reindex(order_used).reset_index()

                # Plot
                text_labels = dist["Percentage"].round(1).astype(str) + "%"
                fig = px.bar(
                    dist,
                    x="Age Group",
                    y="Percentage",
                    title=f"Age Distribution (%) – {cat_type}: {selected_cat}",
                    text=text_labels,
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                ymax = min(100.0, float(dist["Percentage"].max()) + 10.0)
                fig.update_layout(yaxis_range=[0, ymax])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required columns not found: ensure ‘Age’ and the selected category column exist.")

    # ----------------------------------------
    # Sub-tab B: Country Distribution
    # ----------------------------------------
    with sub_country:
        st.markdown("##### Country Distribution per Category")

        cat_type = st.radio(
            "Choose category type:",
            ["Primary Category", "Secondary Category"],
            key="country_cat_type",
            horizontal=True
        )
        cat_col = cat_type  

        if (cat_col in df_f.columns) and ("Country Of Residence" in df_f.columns):
            cat_values = (
                df_f[cat_col].fillna("Unknown").astype(str).replace({"": "Unknown"}).unique().tolist()
            )
            cat_values = [v for v in sorted(cat_values) if v != "Unknown"] + (
                ["Unknown"] if "Unknown" in cat_values else []
            )

            selected_cat = st.selectbox(f"Select {cat_type}:", cat_values, key="country_cat_select")

            subset = df_f[df_f[cat_col].fillna("Unknown").astype(str) == selected_cat].copy()
            if subset.empty:
                st.info("No rows for this selection.")
            else:
                country_series = subset["Country Of Residence"].fillna("Unknown").astype(str).replace({"": "Unknown"})
                counts = country_series.value_counts()
                total = counts.sum()
                pct = (counts / total) * 100.0
                cdf = pct.reset_index()
                cdf.columns = ["Country", "Percentage"]
                cdf = cdf.sort_values("Percentage", ascending=False)

                top_df  = cdf.head(top_k).copy()
                rest_df = cdf.iloc[top_k:]
                others_pct = float(rest_df["Percentage"].sum()) if not rest_df.empty else 0.0

                rows = [top_df]
                if others_pct > 0:
                    rows.append(pd.DataFrame([{"Country": "All other countries", "Percentage": others_pct}]))

                cdf_final = pd.concat(rows, ignore_index=True) if rows else top_df

                include_unknown = st.checkbox("Include Unknown countries", value=False, key="include_unknown_cty")

                if not include_unknown:
                    keep_mask = cdf_final["Country"] != "Unknown"
                    cdf_final = cdf_final[keep_mask].copy()
                    scale = float(cdf_final["Percentage"].sum())
                    if scale > 0:
                        cdf_final["Percentage"] = cdf_final["Percentage"] * (100.0 / scale)

                unknown_pct = float((country_series == "Unknown").mean() * 100)
                st.caption(f"Data quality note: Unknown countries = {unknown_pct:.1f}% of rows for this selection.")

                text_labels = cdf_final["Percentage"].round(1).astype(str) + "%"
                fig = px.bar(
                    cdf_final,
                    x="Country",
                    y="Percentage",
                    title=f"Country Distribution (%) – {cat_type}: {selected_cat}",
                    text=text_labels,
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                # Keep headroom for labels; clamp max to 100
                ymax = min(100.0, float(cdf_final["Percentage"].max()) + 10.0)
                fig.update_layout(yaxis_range=[0, ymax], xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required columns not found: make sure ‘Country Of Residence’ and category columns exist in the dataset.")       

# --- Tab 7: Programme Cost
with tab_7:
    st.subheader("Programme Cost Analysis")

    # Check if required columns exist in the filtered dataframe
    required_cols = ["Programme Cost", "Truncated Programme Name", "Run_Month"]
    if not all(col in df_f.columns for col in required_cols):
        st.warning("The required columns ('Programme Cost', 'Truncated Programme Name', 'Run_Month') are not available in the data.")
    else:
        # Create a working copy for this tab's analysis
        df_cost = df_f.copy()
        
        # Ensure 'Programme Cost' is a numeric type, coercing errors will turn non-numbers into NaN
        df_cost['Programme Cost'] = pd.to_numeric(df_cost['Programme Cost'], errors='coerce')
        
        # Drop rows where 'Programme Cost' could not be converted (is NaN)
        df_cost.dropna(subset=['Programme Cost'], inplace=True)

        if df_cost.empty:
            st.info("No data with valid programme costs found for the current filters.")
        else:
            # --- 1) Scatter Plot: Enrolment vs. Cost ---
            st.markdown("##### Enrolment Volume vs. Programme Cost")
            
            grouped = df_cost.groupby('Truncated Programme Name').agg(
                enrolment_volume=('Truncated Programme Name', 'size'),
                programme_cost=('Programme Cost', 'first') # 'first' is safe as we group by programme
            ).reset_index()

            fig_scatter = px.scatter(
                grouped,
                x='programme_cost',
                y='enrolment_volume',
                title='Enrolment Volume vs. Programme Cost',
                labels={'programme_cost': 'Programme Cost ($)', 'enrolment_volume': 'Total Enrolments'},
                hover_data=['Truncated Programme Name'] # Show programme name on hover
            )
            fig_scatter.update_traces(marker=dict(size=12, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.divider()

            # --- 2) Time Trend: Monthly Revenue ---
            st.markdown("##### Monthly Revenue Trend")

            # Group by run month and sum the programme cost
            monthly_revenue = (
                df_cost
                .groupby(df_cost['Run_Month'].dt.to_period('M'))['Programme Cost']
                .sum()
                .reset_index()
            )
            monthly_revenue.rename(columns={'Programme Cost': 'Total_Revenue'}, inplace=True)
            
            # Convert the period back to a timestamp for Plotly
            monthly_revenue['Run_Month'] = monthly_revenue['Run_Month'].dt.to_timestamp()
            monthly_revenue = monthly_revenue.sort_values("Run_Month")

            fig_trend = px.line(
                monthly_revenue,
                x='Run_Month',
                y='Total_Revenue',
                title='Monthly Revenue Trend',
                labels={'Run_Month': 'Month', 'Total_Revenue': 'Total Revenue ($)'},
                markers=True
            )
            fig_trend.update_layout(yaxis_title="Total Revenue ($)", xaxis_title="Month")
            st.plotly_chart(fig_trend, use_container_width=True)


# --- Tab 7: Programme Deep Dive
with tab_8:
    st.subheader("Programme Deep Dive")

    prog_col = "Truncated Programme Name"
    if prog_col not in df_f.columns:
        st.info("Programme column not found in the filtered data.")
        st.stop()

    # Programme picker
    progs = (
        df_f[prog_col].dropna().astype(str).sort_values().unique().tolist()
        if not df_f.empty else []
    )
    if not progs:
        st.info("No programmes available under current filters.")
        st.stop()

    sel_prog = st.selectbox("Select a programme", progs, index=0, key="prog_dd_select")

    # Subset for the selected programme
    p = df_f[df_f[prog_col] == sel_prog].copy()
    if p.empty:
        st.info("No rows for this programme with current filters.")
        st.stop()

    # ---- KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Participants", f"{len(p):,}")
    with c2:
        st.metric("Unique Runs", int(p["Truncated Programme Run"].nunique()))   
    with c3:
        st.metric("Countries", int(p["Country Of Residence"].nunique()))
    with c4:
        st.metric("Median Age", f"{p['Age'].median():.0f}" if p["Age"].notna().any() else "—")
    date_min = pd.to_datetime(p["Programme Start Date"]).min()
    date_max = pd.to_datetime(p["Programme End Date"]).max()   
    if pd.notna(date_min) and pd.notna(date_max):
        st.caption(f"**Programme Date Range:** {date_min:%A, %d %B %Y} → {date_max:%A, %d %B %Y}")
        st.divider()

    # ---- Time series: participants by run month
    if "Run_Month" in p.columns:
        ts = p.groupby("Run_Month").size().reset_index(name="Participants").sort_values("Run_Month")
        fig_ts = px.line(ts, x="Run_Month", y="Participants", markers=True,
                         title="Participants over Time (by Run Month)")
        fig_ts.update_layout(yaxis_title="Participants", xaxis_title="Run Month")
        fig_ts.update_xaxes(tickformat="%b %Y")
        fig_ts.update_traces(hovertemplate="Run Month=%{x|%b %Y}<br>Participants=%{y}<extra></extra>")
        st.plotly_chart(fig_ts, use_container_width=True)

    colL, colR = st.columns(2)

    # ---- Left: Application Status distribution + Gender
    with colL:
        if "Application Status" in p.columns:
            status = p["Application Status"].fillna("Unknown").value_counts().reset_index()
            status.columns = ["Application Status", "Count"]
            fig_stat = px.bar(status, x="Application Status", y="Count",
                                title="Application Status Breakdown", text="Count")
            fig_stat.update_traces(textposition="outside", cliponaxis=False)
            fig_stat.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_stat, use_container_width=True)

        if "Gender" in p.columns:
            gender = p["Gender"].fillna("Unknown").str.capitalize().value_counts().reset_index()
            gender.columns = ["Gender", "Participants"]
            fig_gender = px.pie(gender, names="Gender", values="Participants", title="Gender Split")
            st.plotly_chart(fig_gender, use_container_width=True)

    # ---- Right: Top countries + Top organisations
    with colR:
        if "Country Of Residence" in p.columns:
            ctry = p["Country Of Residence"].fillna("Unknown").value_counts().reset_index()
            ctry.columns = ["Country", "Participants"]
            ctry_top = ctry.head(top_k)
            fig_ctry = px.bar(ctry_top, x="Participants", y="Country", orientation="h",
                                title=f"Top {top_k} Countries (Participants)")
            st.plotly_chart(fig_ctry, use_container_width=True)

        org_col = "Organisation Name: Organisation Name"
        if org_col in p.columns:
            orgs = p[org_col].fillna("Unknown").value_counts().reset_index()
            orgs.columns = ["Organisation", "Participants"]
            orgs_top = orgs.head(top_k)
            fig_org = px.bar(orgs_top, x="Participants", y="Organisation", orientation="h",
                                 title=f"Top {top_k} Organisations (Participants)")
            st.plotly_chart(fig_org, use_container_width=True)

    # ---- Seniority + Age distribution
    colA, colB = st.columns(2)
    with colA:
        if "Seniority" in p.columns:
            sen = p["Seniority"].fillna("Unknown").value_counts().reset_index()
            sen.columns = ["Seniority", "Participants"]
            fig_sen = px.bar(sen, x="Seniority", y="Participants", title="Seniority Mix")
            st.plotly_chart(fig_sen, use_container_width=True)

    with colB:
        # Build/ensure Age_Group (same logic as load_data)
        if "Age" in p.columns:
            age_num = pd.to_numeric(p["Age"], errors="coerce")
            bins    = [0, 34, 44, 54, 64, 200]
            labels = ["<35", "35–44", "45–54", "55–64", "65+"]
            p["Age_Group"] = pd.cut(age_num, bins=bins, labels=labels, right=True).astype("string")
            p.loc[age_num.isna(), "Age_Group"] = "Unknown"

            agec = p["Age_Group"].value_counts().reindex(
                ["<35","35–44","45–54","55–64","65+","Unknown"]
            ).dropna().reset_index()
            agec.columns = ["Age Group", "Participants"]
            fig_age = px.bar(agec, x="Age Group", y="Participants", title="Age Group Distribution")
            st.plotly_chart(fig_age, use_container_width=True)

    # ---- Runs table (optional quick view)
    st.markdown("##### Runs for this Programme")
    run_cols = [c for c in ["Truncated Programme Run", "Programme Start Date", "Programme End Date", "Country Of Residence", "Application Status"] if c in p.columns]
    st.dataframe(
        p.sort_values(["Run_Month","Programme Start Date"]).loc[:, run_cols].head(500),
        use_container_width=True,
        hide_index=True
    )

    # ---- Export button for this programme
    st.download_button(
        f"Download '{sel_prog}' rows (CSV)",
        data=p.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{sel_prog[:40].replace(' ','_')}_export.csv",
        mime="text/csv"
    ) 

# --- Tab 9: Data Preview
with tab_9:
    st.subheader("Filtered Data Preview")
    preview_cols = [c for c in [
        "Application ID", "Contact ID", "Application Status", "Applicant Type",
        "Organisation Name: Organisation Name", "Job Title Clean", "Seniority",
        "Truncated Programme Name", "Truncated Programme Run", "Primary Category", "Secondary Category",
        "Programme Start Date", "Programme End Date", "Run_Month", "Run_Month_Label",
        "Gender", "Age", "Country Of Residence", "Domain", "Programme Cost"  
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
