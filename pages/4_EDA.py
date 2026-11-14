"""
pages/4_EDA.py - Exploratory Data Analysis (EDA)
AI Student Performance & Academic Guidance System
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from utils import (
    DEFAULT_DATASET_PATH,
    apply_custom_css,
    compute_eda_questions,
    generate_sample_student_dataset,
    get_active_df,
    get_dataset_info,
    get_statistical_glossary,
    init_session_state,
    render_feature_chart,
    render_footer,
    render_header,
    render_sidebar_status,
    render_top_navbar,
    set_session_dataset,
)

# ---------------------------------------------------------
# Page Configuration & Global Shell
# ---------------------------------------------------------
st.set_page_config(
    page_title="Exploratory Data Analysis | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/4_EDA.py")
render_sidebar_status()

render_header(
    title="Exploratory Data Analysis",
    subtitle="Examine dataset structure, statistical metrics, feature distributions, correlations, and core cohort questions.",
    badge_text="Module 04",
    badge_variant="indigo",
)

df, is_cleaned = get_active_df()

if df is None:
    with st.container(border=True):
        st.markdown("**No active dataset loaded in session state**")
        st.caption("Upload a dataset or load the cleaned balanced cohort before running Exploratory Data Analysis.")
        c1, c2, _ = st.columns([1.5, 1.4, 2.4])
        with c1:
            if st.button("Load Cleaned Cohort", type="primary", width="stretch"):
                if os.path.exists(DEFAULT_DATASET_PATH):
                    sample_df = pd.read_csv(DEFAULT_DATASET_PATH)
                    set_session_dataset(sample_df, os.path.basename(DEFAULT_DATASET_PATH), mark_cleaned=True)
                else:
                    sample_df = generate_sample_student_dataset(n_samples=200, inject_imperfections=False)
                    set_session_dataset(sample_df, "sample_student_data_200.csv", mark_cleaned=True)
                st.rerun()
        with c2:
            if st.button("Open Upload Dataset", width="stretch"):
                st.switch_page("pages/2_Upload_Dataset.py")
else:
    st.caption(
        f"Active dataset source: {'cleaned_df' if is_cleaned else 'raw_df'} "
        f"({len(df):,} rows, {len(df.columns)} columns)."
    )

    # ---------------------------------------------------------
    # Dedicated 4-Question Cohort Answers Panel
    # ---------------------------------------------------------
    st.markdown("#### Key Cohort Questions Answered")
    eda_q = compute_eda_questions(df)

    with st.container(border=True):
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown("**1. How many students are included?**")
            st.metric("Total Students", f"{eda_q['student_count']:,}")
        with q2:
            st.markdown("**2. What are their average marks?**")
            if eda_q["avg_marks_map"]:
                for label, val in eda_q["avg_marks_map"].items():
                    st.write(f"- {label}: **{val:.2f}**")
            else:
                st.metric("Average Numeric Value", f"{eda_q['primary_avg_marks']:.2f}")
        with q3:
            st.markdown("**3. How many students have high attendance (>= 75%)?**")
            if eda_q["attendance_col"]:
                st.metric(
                    "Attendance >= 75%",
                    f"{eda_q['high_attendance_count']:,}",
                    f"{eda_q['high_attendance_pct']}% of valid records",
                )
            else:
                st.caption("Attendance column not detected in active dataset.")
        with q4:
            st.markdown("**4. Is there any relationship between study hours and academic performance?**")
            if eda_q["correlation_r"] is not None:
                st.metric("Correlation (r)", f"{eda_q['correlation_r']:.3f}", eda_q["relationship_strength"])
                st.caption(eda_q["relationship_summary"])
            else:
                st.caption(eda_q["relationship_summary"])

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # A) Data Overview
    # ---------------------------------------------------------
    st.markdown("#### A. Data Overview")
    info = get_dataset_info(df)
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Total Records (Rows)", f"{info['rows']:,}")
    ov2.metric("Total Features (Columns)", f"{info['columns']}")
    ov3.metric("Missing Cells", f"{info['missing_total']}")
    ov4.metric("Duplicate Rows", f"{info['duplicate_total']}")

    with st.container(border=True):
        st.markdown("**Column Names and Data Types**")
        st.dataframe(info["dtypes_df"], width="stretch", hide_index=True)

    # ---------------------------------------------------------
    # B) Statistical Summary + Reference Glossary
    # ---------------------------------------------------------
    st.markdown("#### B. Statistical Summary")
    num_df = df.select_dtypes(include=[np.number])

    c_stat_tbl, c_glossary = st.columns([1.6, 1.1])
    with c_stat_tbl:
        with st.container(border=True):
            st.markdown("**Numerical Summary Statistics**")
            if not num_df.empty:
                desc_df = num_df.describe().T.reset_index().rename(columns={"index": "Feature"})
                desc_df = desc_df.rename(
                    columns={
                        "count": "Count",
                        "mean": "Mean",
                        "std": "Standard deviation",
                        "min": "Minimum",
                        "max": "Maximum",
                    }
                )
                st.dataframe(desc_df.round(2), width="stretch", hide_index=True)
            else:
                st.info("No numerical columns available for statistical summary.")

    with c_glossary:
        with st.container(border=True):
            st.markdown("**Statistical Term Reference**")
            st.dataframe(get_statistical_glossary(), width="stretch", hide_index=True)

    # ---------------------------------------------------------
    # C) Understanding Data Distribution
    # ---------------------------------------------------------
    st.markdown("#### C. Understanding Data Distribution")
    num_cols = num_df.columns.tolist()
    if num_cols:
        with st.container(border=True):
            dist_col = st.selectbox(
                "Select Numerical Feature to Inspect Distribution",
                options=num_cols,
                index=num_cols.index(eda_q["perf_col"]) if eda_q["perf_col"] in num_cols else 0,
            )
            c_dist_chart, c_dist_box = st.columns(2)
            with c_dist_chart:
                fig_hist, hist_insight = render_feature_chart(df, dist_col, "Histogram with KDE")
                st.pyplot(fig_hist, width="stretch")
                plt.close(fig_hist)
                st.caption(hist_insight)
            with c_dist_box:
                fig_box, box_insight = render_feature_chart(df, dist_col, "Box Plot")
                st.pyplot(fig_box, width="stretch")
                plt.close(fig_box)
                st.caption(box_insight)
    else:
        st.info("No numerical columns available for distribution plots.")

    # ---------------------------------------------------------
    # D) Finding Relationships (Correlation Table & Matrix)
    # ---------------------------------------------------------
    st.markdown("#### D. Finding Relationships")
    if len(num_cols) >= 2:
        corr_matrix = num_df.corr().round(3)
        c_corr_tbl, c_corr_fig = st.columns([1.1, 1.3])
        with c_corr_tbl:
            with st.container(border=True):
                st.markdown("**Correlation Table (Pearson r)**")
                st.dataframe(corr_matrix, width="stretch")
        with c_corr_fig:
            with st.container(border=True):
                fig_corr, corr_insight = render_feature_chart(df, num_cols[0], "Correlation Heatmap")
                st.pyplot(fig_corr, width="stretch")
                plt.close(fig_corr)
                st.caption(corr_insight)
    else:
        st.info("At least two numerical columns are needed to compute feature correlations.")

    c_next1, c_next2, _ = st.columns([1.4, 1.4, 3])
    with c_next1:
        if st.button("Proceed to Data Visualization", type="primary", width="stretch"):
            st.switch_page("pages/5_Data_Visualization.py")
    with c_next2:
        if st.button("Return to Data Cleaning", width="stretch"):
            st.switch_page("pages/3_Data_Cleaning.py")

render_footer()
