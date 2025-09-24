import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="EE Analytics Dashboard", layout="wide")

# ------------- Data Load & Prep -------------
@st.cache_data
def load_data(path="dashboard_curated.csv"):
    df = pd.read_csv(path)
    # Parse dates
    for c in ["Programme Start Date", "Programme End Date", "Run_Month"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Friendly label for month 
    if "Run_Month_Label" not in df.columns and "Run_Month" in df.columns:
        df["Run_Month_Label"] = df["Run_Month"].dt.strftime("%b-%Y")

    # Age group (with Unknown bucket)
    if "Age" in df.columns:
        bins   = [0, 34, 44, 54, 64, 200]
        labels = ["<35", "35–44", "45–54", "55–64", "65+"]
        age_num = pd.to_numeric(df["Age"], errors="coerce")
        df["Age_Group"] = pd.cut(age_num, bins=bins, labels=labels, right=True)
        df["Age_Group"] = df["Age_Group"].astype("string")
        df.loc[age_num.isna(), "Age_Group"] = "Unknown"

    # Light category cleanup
    cat_cols = ["Application Status","Applicant Type","Primary Category","Secondary Category","Seniority",
                "Gender","Country Of Residence","Truncated Programme Name"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    return df

# Allow user to upload a newer CSV (optional)
uploaded = st.sidebar.file_uploader("Upload a curated CSV (optional)", type=["csv"])
data_path = uploaded if uploaded is not None else "dashboard_curated.csv"
df = load_data(data_path)

if df.empty:
    st.info("No data loaded. Please upload a CSV or place dashboard_curated.csv next to this script.")
    st.stop()

st.title("Executive Education Analytics Dashboard")

# ------------- Sidebar Filters -------------
with st.sidebar:
    st.header("Filters")

    # Time range by Run_Month
    if "Run_Month" in df.columns:
        min_m, max_m = df["Run_Month"].min(), df["Run_Month"].max()
        default = (min_m, max_m)
        date_range = st.date_input("Run month range", value=default, min_value=min_m.date(), max_value=max_m.date())
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df["Run_Month"] >= start_d) & (df["Run_Month"] <= end_d)]

    # Core slicers
    def multiselect_with_select_all(label, series: pd.Series, default_all=True):
        # Work on a copy to avoid modifying original df
        s = series.copy()

        # Add 'Unknown' if there are NaNs
        if s.isna().any():
            s = s.fillna("Unknown")

        options = sorted(s.unique().tolist())

        all_selected = st.checkbox(f"Select all {label}", value=default_all, key=f"{label}_all")
        if all_selected:
            selected = st.multiselect(label, options, default=options, key=f"{label}_multi")
        else:
            selected = st.multiselect(label, options, key=f"{label}_multi")

        return selected


    sel_status   = multiselect_with_select_all("Application Status", df["Application Status"])
    sel_app_type = multiselect_with_select_all("Applicant Type", df["Applicant Type"])
    sel_primcat  = multiselect_with_select_all("Primary Category", df["Primary Category"])
    sel_secncat  = multiselect_with_select_all("Secondary Category", df["Secondary Category"])
    sel_country  = multiselect_with_select_all("Country", df["Country Of Residence"])
    sel_senior   = multiselect_with_select_all("Seniority", df["Seniority"])

    top_k = st.number_input("Top K (for Top-X charts)", min_value=3, max_value=50, value=10, step=1)

# Apply sidebar filters
def apply_filter(series: pd.Series, selected):
    if selected is None or len(selected) == 0:
        # No filter applied → keep all rows
        return pd.Series(True, index=series.index)

    # Convert NaN → "Unknown" to align with filter values
    s = series.fillna("Unknown")

    # If user has selected *all available options*, skip filtering
    all_opts = set(s.unique())
    if set(selected) == all_opts:
        return pd.Series(True, index=series.index)

    return s.isin(selected)

mask = (
    apply_filter(df.get("Application Status", pd.Series(index=df.index)), sel_status) &
    apply_filter(df.get("Applicant Type", pd.Series(index=df.index)), sel_app_type) &
    apply_filter(df.get("Primary Category", pd.Series(index=df.index)), sel_primcat) &
    apply_filter(df.get("Secondary Category", pd.Series(index=df.index)), sel_secncat) &
    apply_filter(df.get("Country Of Residence", pd.Series(index=df.index)), sel_country) &
    apply_filter(df.get("Seniority", pd.Series(index=df.index)), sel_senior)
)

df_f = df[mask].copy()
st.caption(f"Filtered rows: {len(df_f):,} of {len(df):,}")

# ------------- Tabs -------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Time Series", "🗺️ Geography", "🏷️ Programmes × Country",
    "👔 Titles & Orgs", "🧮 Age & Demographics", "ℹ️ Data Preview"
])

# ---------- Tab 1: Time Series ----------
with tab1:
    st.subheader("Participants over Time")
    if "Run_Month" in df_f.columns:
        # Count applications as proxy for participants
        ts = df_f.groupby("Run_Month").size().reset_index(name="Participants")
        ts = ts.sort_values("Run_Month")
        fig = px.line(ts, x="Run_Month", y="Participants", markers=True)
        fig.update_layout(yaxis_title="Participants", xaxis_title="Run Month")
        st.plotly_chart(fig, use_container_width=True)

    # Seasonal bars by month / quarter
    col_a, col_b = st.columns(2)
    if "Programme Start Date" in df_f.columns:
        df_f["Start_Month_Num"] = df_f["Programme Start Date"].dt.month
        df_f["Start_Quarter"]   = df_f["Programme Start Date"].dt.quarter

        with col_a:
            mon = df_f.groupby("Start_Month_Num").size().reset_index(name="Applications")
            mon["Month"] = mon["Start_Month_Num"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                                                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
            mon = mon.sort_values("Start_Month_Num")
            figm = px.bar(mon, x="Month", y="Applications", title="Applications by Start Month")
            st.plotly_chart(figm, use_container_width=True)

        with col_b:
            q = df_f.groupby("Start_Quarter").size().reset_index(name="Applications")
            figq = px.bar(q, x="Start_Quarter", y="Applications", title="Applications by Start Quarter")
            st.plotly_chart(figq, use_container_width=True)

# ---------- Tab 2: Geography ----------
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
        # Top K countries overall to focus the time series
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

# ---------- Tab 3: Programmes × Country ----------
with tab3:
    st.subheader("Top Programmes & Country Breakdown")
    prog_col = "Truncated Programme Name"
    if prog_col in df_f.columns and "Country Of Residence" in df_f.columns:
        # Top K programmes overall
        top_progs = (
            df_f[prog_col].value_counts().nlargest(top_k).index.tolist()
        )
        df_top = df_f[df_f[prog_col].isin(top_progs)].copy()

        # Stacked bar: participants by programme (x) & stacked by country
        agg = df_top.groupby([prog_col, "Country Of Residence"]).size().reset_index(name="Participants")
        # Focus on top K countries within this subset for readability
        top_c_in_subset = agg.groupby("Country Of Residence")["Participants"].sum().nlargest(top_k).index.tolist()
        agg = agg[agg["Country Of Residence"].isin(top_c_in_subset)]

        fig = px.bar(
            agg, x=prog_col, y="Participants", color="Country Of Residence",
            title=f"Participants by Programme (Top {top_k}) and Country", barmode="stack"
        )
        fig.update_layout(xaxis_title="Programme (Anon)", yaxis_title="Participants")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 4: Titles & Organisations ----------
with tab4:
    st.subheader("Top Job Titles & Organisations")
    col1, col2 = st.columns(2)

    if "Job Title Clean" in df_f.columns:
        top_titles = df_f["Job Title Clean"].value_counts().nlargest(top_k).reset_index()
        top_titles.columns = ["Job Title", "Participants"]
        fig1 = px.bar(top_titles, x="Participants", y="Job Title", orientation="h",
                      title=f"Top {top_k} Job Titles")
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

    if "Organisation Name: Organisation Name" in df_f.columns:
        top_orgs = df_f["Organisation Name: Organisation Name"].value_counts().nlargest(top_k).reset_index()
        top_orgs.columns = ["Organisation", "Participants"]
        fig2 = px.bar(top_orgs, x="Participants", y="Organisation", orientation="h",
                      title=f"Top {top_k} Organisations")
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

    # Seniority distribution 
    if "Seniority" in df_f.columns:
        sen = df_f["Seniority"].value_counts().reset_index()
        sen.columns = ["Seniority", "Participants"]
        fig3 = px.bar(sen, x="Seniority", y="Participants", title="Participants by Seniority")
        st.plotly_chart(fig3, use_container_width=True)

# ---------- Tab 5: Age & Demographics ----------
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

# ---------- Tab 6: Data Preview ----------
with tab6:
    st.subheader("Filtered Data Preview")
    st.dataframe(
        df_f.sort_values("Run_Month").head(500),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_export.csv",
        mime="text/csv"
    )
