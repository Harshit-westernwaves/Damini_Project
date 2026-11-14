"""
pages/1_Home.py - Main Dashboard Page
AI Student Performance & Academic Guidance System
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
    page_title="Main Dashboard | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/1_Home.py")
render_sidebar_status()

render_header(
    title="Main Dashboard",
    subtitle="Central navigation interface for dataset processing, analysis, modeling, and prediction.",
    badge_text="Module 01",
    badge_variant="indigo",
)

# ---------------------------------------------------------
# Active Workspace Snapshot
# ---------------------------------------------------------
active_df, is_cleaned = get_active_df()

if active_df is None:
    with st.container(border=True):
        st.markdown("**No active dataset loaded in session state**")
        st.caption("Start by uploading a CSV file or generating the standard 6-column sample dataset.")
        c_b1, c_b2, _ = st.columns([1.4, 1.4, 2.2])
        with c_b1:
            if st.button("Load Default Cohort", type="primary", width="stretch"):
                if os.path.exists(DEFAULT_DATASET_PATH):
                    sample_df = pd.read_csv(DEFAULT_DATASET_PATH)
                    set_session_dataset(sample_df, os.path.basename(DEFAULT_DATASET_PATH), mark_cleaned=True)
                else:
                    sample_df = generate_sample_student_dataset(n_samples=200, inject_imperfections=False)
                    set_session_dataset(sample_df, "sample_student_data_200.csv", mark_cleaned=True)
                st.rerun()
        with c_b2:
            if st.button("Open Dataset Uploading", width="stretch"):
                st.switch_page("pages/2_Upload_Dataset.py")
else:
    eda = compute_eda_questions(active_df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students Included", f"{eda['student_count']:,}", "Cleaned" if is_cleaned else "Raw")
    k2.metric("Average Marks", f"{eda['primary_avg_marks']:.2f}")
    k3.metric("Attendance >= 75%", f"{eda['high_attendance_count']}", f"{eda['high_attendance_pct']}% of cohort")
    model_status = st.session_state.get("model_type") or "Not Trained"
    k4.metric("Active ML Model", model_status)

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Organized Core Workflow Navigation
# ---------------------------------------------------------
st.markdown("#### Workflow Navigation")

row1 = st.columns(3)
row2 = st.columns(3)

core_modules = [
    (
        row1[0],
        "1. Dataset Uploading",
        "Upload a student CSV file or generate the 6-column sample dataset and view dataset info.",
        "Open Dataset Uploading",
        "pages/2_Upload_Dataset.py",
        "dash_upload",
    ),
    (
        row1[1],
        "2. Data Cleaning",
        "Detect nulls, apply drop or mean/median/mode imputation, remove duplicates, and export cleaned CSV.",
        "Open Data Cleaning",
        "pages/3_Data_Cleaning.py",
        "dash_clean",
    ),
    (
        row1[2],
        "3. Exploratory Data Analysis",
        "Inspect overview, statistical summary glossary, distributions, correlations, and 4 cohort questions.",
        "Open Exploratory Data Analysis",
        "pages/4_EDA.py",
        "dash_eda",
    ),
    (
        row2[0],
        "4. Data Visualization",
        "Generate Matplotlib and Seaborn charts by feature and review observed data patterns.",
        "Open Data Visualization",
        "pages/5_Data_Visualization.py",
        "dash_vis",
    ),
    (
        row2[1],
        "5. Model Training",
        "Configure features and target, split data, train Linear or Logistic Regression, and save with Joblib.",
        "Open Model Training",
        "pages/6_Model_Training.py",
        "dash_train",
    ),
    (
        row2[2],
        "6. Performance Prediction",
        "Enter Study hrs, Attendance, Previous marks, and Assignment scores for prediction and guidance.",
        "Open Performance Prediction",
        "pages/7_Prediction.py",
        "dash_pred",
    ),
]

for col, title, desc, btn_label, target, key in core_modules:
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            if st.button(btn_label, key=key, width="stretch"):
                st.switch_page(target)

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Support & Documentation Modules
# ---------------------------------------------------------
st.markdown("#### Advisory & Documentation")
sup1, sup2 = st.columns(2)

with sup1:
    with st.container(border=True):
        st.markdown("**AI Academic Chatbot**")
        st.caption("Ask study strategy, attendance, ML concept, or general guidance questions via Groq AI or local fallback.")
        if st.button("Open AI Chatbot", key="dash_chat", width="stretch"):
            st.switch_page("pages/8_AI_Chatbot.py")

with sup2:
    with st.container(border=True):
        st.markdown("**About Project**")
        st.caption("Review project purpose, objectives, module-by-module features, and system architecture.")
        if st.button("Open About Project", key="dash_about", width="stretch"):
            st.switch_page("pages/9_About_Project.py")

render_footer()
