"""
app.py - Welcome Section and entry point for the
AI Student Performance & Academic Guidance System.
"""

import os
import pandas as pd
import streamlit as st
from utils import (
    DEFAULT_DATASET_PATH,
    apply_custom_css,
    compute_eda_questions,
    generate_sample_student_dataset,
    get_active_df,
    init_session_state,
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
    page_title="Welcome | AI Student Performance & Academic Guidance System",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
apply_custom_css()
render_top_navbar("app.py")
render_sidebar_status()

# ---------------------------------------------------------
# a) Project Name & Brief Introduction
# ---------------------------------------------------------
render_header(
    title="AI Student Performance & Academic Guidance System",
    subtitle=(
        "An AI-powered student analytics platform for dataset preparation, exploratory profiling, "
        "performance forecasting, and academic guidance."
    ),
    badge_text="Welcome",
    badge_variant="indigo",
)

active_df, is_cleaned = get_active_df()

with st.container(border=True):
    c_intro, c_action = st.columns([2.6, 1.4])
    with c_intro:
        st.markdown("#### Platform Overview")
        st.write(
            "This AI-powered student analytics platform processes student academic records from ingestion "
            "through predictive modeling and individualized advising. Educators and students can clean raw CSV "
            "files, inspect attendance and study patterns, train regression or classification models, and query "
            "an integrated Groq AI assistant."
        )
        if active_df is not None:
            eda = compute_eda_questions(active_df)
            state_label = "Cleaned dataset active" if is_cleaned else "Raw dataset active"
            st.caption(
                f"Session status: {state_label} ({eda['student_count']} records, "
                f"primary average score: {eda['primary_avg_marks']})."
            )
    with c_action:
        st.markdown("#### Quick Start")
        if st.button("Open Main Dashboard", type="primary", width="stretch"):
            st.switch_page("pages/1_Home.py")
        if active_df is None:
            if st.button("Load Default Dataset (datasets/)", type="primary", width="stretch"):
                if os.path.exists(DEFAULT_DATASET_PATH):
                    sample_df = pd.read_csv(DEFAULT_DATASET_PATH)
                    set_session_dataset(sample_df, os.path.basename(DEFAULT_DATASET_PATH), mark_cleaned=True)
                else:
                    sample_df = generate_sample_student_dataset(n_samples=200, inject_imperfections=False)
                    set_session_dataset(sample_df, "sample_student_data_200.csv", mark_cleaned=True)
                st.rerun()
            if st.button("Open Upload Dataset", width="stretch"):
                st.switch_page("pages/2_Upload_Dataset.py")
        else:
            if st.button("Open Upload Dataset", width="stretch"):
                st.switch_page("pages/2_Upload_Dataset.py")

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# b) AI Intelligence Section
# ---------------------------------------------------------
st.markdown("#### AI Intelligence")
ai_cols = st.columns(4)

ai_pillars = [
    (
        "Data Analysis",
        "Automated detection of missing values, duplicate records, summary statistics, and feature correlations.",
    ),
    (
        "Machine Learning",
        "Supervised Linear Regression and Logistic Regression pipelines with configurable train/test splits.",
    ),
    (
        "Student Performance Prediction",
        "Individual outcome forecasting from Study hrs, Attendance, Previous marks, and Assignment scores.",
    ),
    (
        "Groq AI Assistant",
        "Conversational academic advisor backed by the Groq API with an offline dataset-aware fallback.",
    ),
]

for col, (heading, desc) in zip(ai_cols, ai_pillars):
    with col:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.caption(desc)

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# c) Platform Highlights
# ---------------------------------------------------------
st.markdown("#### Platform Highlights")
hl_cols = st.columns(4)

highlights = [
    (
        "Dataset Handling",
        "CSV ingestion, schema inspection, null handling (drop, mean, median, mode), and deduplication.",
    ),
    (
        "Interactive Analysis",
        "Structured exploratory statistics, 4-question cohort evaluation, and Matplotlib/Seaborn charts.",
    ),
    (
        "ML",
        "Target encoding automation, R2/MAE and Accuracy/Precision/Recall/F1-score metrics, and Joblib persistence.",
    ),
    (
        "AI Academic Assistance",
        "Metric-driven guidance plans and interactive academic chat supporting transcript download.",
    ),
]

for col, (heading, desc) in zip(hl_cols, highlights):
    with col:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.caption(desc)

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# d) How the Platform Works (4-Step Flow)
# ---------------------------------------------------------
st.markdown("#### How the Platform Works")
step_cols = st.columns(4)

steps = [
    ("1. Upload dataset", "Import a student CSV file or generate the standard 6-column sample dataset."),
    ("2. Clean and prepare the data", "Inspect missing values, impute numerical or categorical nulls, and drop duplicates."),
    ("3. Analyse student performance", "Review statistical summaries, answer core cohort questions, and plot feature patterns."),
    ("4. Predict performance and provide guidance", "Train Linear or Logistic Regression, predict scores, and generate study guidance."),
]

for col, (step_title, step_desc) in zip(step_cols, steps):
    with col:
        with st.container(border=True):
            st.markdown(f"**{step_title}**")
            st.caption(step_desc)

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# e) Technology Stack Overview
# ---------------------------------------------------------
st.markdown("#### Technology Stack")
with st.container(border=True):
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Core Application & Data**")
        st.caption("Python, Streamlit, Pandas, NumPy")
    with t2:
        st.markdown("**Visualization & ML**")
        st.caption("Matplotlib, Seaborn, Scikit-learn, Joblib")
    with t3:
        st.markdown("**Conversational Guidance**")
        st.caption("Groq AI API (groq library) with local rule-based fallback")

render_footer()
