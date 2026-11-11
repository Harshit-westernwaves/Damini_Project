"""
pages/9_About_Project.py - System Documentation and Architecture Overview
AI Student Performance & Academic Guidance System
"""

import pandas as pd
import streamlit as st
from utils import (
    apply_custom_css,
    get_active_df,
    init_session_state,
    render_footer,
    render_header,
    render_sidebar_status,
    render_top_navbar,
)

# ---------------------------------------------------------
# Page Configuration & Global Shell
# ---------------------------------------------------------
st.set_page_config(
    page_title="About Project | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/9_About_Project.py")
render_sidebar_status()

render_header(
    title="About Project",
    subtitle="Documentation covering project purpose, objectives, module features, and working architecture.",
    badge_text="Module 09",
    badge_variant="slate",
)

# ---------------------------------------------------------
# 1. Project Purpose
# ---------------------------------------------------------
st.markdown("#### 1. Project Purpose")
with st.container(border=True):
    st.write(
        "The AI Student Performance & Academic Guidance System is an interactive data analytics and "
        "machine learning application designed to evaluate student academic records, detect performance "
        "patterns early, predict academic outcomes, and deliver practical study guidance."
    )
    st.write(
        "Instead of waiting for end-of-term grades to reveal academic difficulties, the platform connects "
        "student indicators (Study hrs, Attendance, Previous marks, and Assignment scores) into a unified "
        "pipeline from raw CSV ingestion to interpretable predictions and AI-supported advising."
    )

# ---------------------------------------------------------
# 2. Core Objectives
# ---------------------------------------------------------
st.markdown("#### 2. Core Objectives")
obj_c1, obj_c2 = st.columns(2)
with obj_c1:
    with st.container(border=True):
        st.markdown("**Data Preparation and Statistical Profiling**")
        st.write("- Standardize student CSV ingestion and schema verification.")
        st.write("- Provide structured missing-value treatment (drop, mean, median, mode) and deduplication.")
        st.write("- Compute summary statistics, quartile glossaries, and direct answers to key cohort questions.")
with obj_c2:
    with st.container(border=True):
        st.markdown("**Predictive Modeling and Academic Guidance**")
        st.write("- Visualize feature distributions and correlations using Matplotlib and Seaborn.")
        st.write("- Train and evaluate Linear Regression and Logistic Regression models with automatic target encoding.")
        st.write("- Generate targeted academic guidance plans and support Groq AI conversational Q&A.")

# ---------------------------------------------------------
# 3. Module-by-Module Features
# ---------------------------------------------------------
st.markdown("#### 3. Module-by-Module Features")

module_docs = [
    ("app.py", "Welcome Section", "Project introduction, AI Intelligence overview, platform highlights, 4-step workflow, and technology stack."),
    ("pages/1_Home.py", "Main Dashboard", "Central workspace summary and organized navigation buttons to all analysis, ML, and prediction modules."),
    ("pages/2_Upload_Dataset.py", "Upload Dataset", "CSV upload, one-click 6-column sample generator, row/column counts, column names, and tabular preview."),
    ("pages/3_Data_Cleaning.py", "Data Cleaning", "Null count/percentage table, drop/mean/median/mode handling, duplicate removal, updated shape/dtypes, and CSV export."),
    ("pages/4_EDA.py", "Exploratory Data Analysis", "Data overview, statistical summary + term glossary, distributions, correlation table, and 4-question cohort panel."),
    ("pages/5_Data_Visualization.py", "Data Visualization", "Matplotlib and Seaborn charts driven by numeric/categorical dropdowns with computed visual insights."),
    ("pages/6_Model_Training.py", "Model Training", "Overview of 3 ML approaches, feature/target selection, train/test split, Linear/Logistic Regression, evaluation metrics, and Joblib persistence."),
    ("pages/7_Prediction.py", "Prediction & Guidance", "Controls for Study hrs, Attendance, Previous marks, Assignment scores, Final performance inference, and structured guidance."),
    ("pages/8_AI_Chatbot.py", "AI Chatbot", "Groq AI integration with local dataset/rule-based fallback, 4 advisory topics, chat history, Clear Chat, and Download Chat."),
    ("pages/9_About_Project.py", "About Project", "Full system documentation, module inventory, working architecture, and live session registry inspection."),
]

modules_df = pd.DataFrame(module_docs, columns=["File", "Module", "Implemented Features"])
st.dataframe(modules_df, width="stretch", hide_index=True)

# ---------------------------------------------------------
# 4. Overall Working Architecture
# ---------------------------------------------------------
st.markdown("#### 4. Overall Working Architecture")
arch_c1, arch_c2, arch_c3 = st.columns(3)

with arch_c1:
    with st.container(border=True):
        st.markdown("**Data & State Layer**")
        st.caption(
            "Shared utilities in utils.py synchronize raw_df, cleaned_df, trained_model, "
            "model_features, and chat_history across all pages via Streamlit session state."
        )
with arch_c2:
    with st.container(border=True):
        st.markdown("**Analytics & ML Layer**")
        st.caption(
            "Pandas and NumPy process tabular metrics; Matplotlib and Seaborn render static plots; "
            "Scikit-learn trains Linear Regression and Logistic Regression models persisted with Joblib."
        )
with arch_c3:
    with st.container(border=True):
        st.markdown("**Guidance & AI Layer**")
        st.caption(
            "Prediction outputs feed a deterministic academic guidance generator, while the AI Chatbot "
            "connects to the Groq AI API with an automatic local fallback."
        )

# ---------------------------------------------------------
# Live Session State Diagnostic View
# ---------------------------------------------------------
active_df, is_cleaned = get_active_df()
with st.expander("Session State Summary", expanded=False):
    st.write(f"- Active Dataset: {'cleaned_df' if is_cleaned else ('raw_df' if active_df is not None else 'None')}")
    st.write(f"- Dataset Rows: {len(active_df) if active_df is not None else 0}")
    st.write(f"- Active Model: {st.session_state.get('model_type') or 'None'}")
    st.write(f"- Model Features: {st.session_state.get('model_features', [])}")
    st.write(f"- Chat Messages: {len(st.session_state.get('chat_history', []))}")

render_footer()
