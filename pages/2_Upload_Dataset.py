"""
pages/2_Upload_Dataset.py - Dataset Uploading and Sample Dataset Generation
AI Student Performance & Academic Guidance System
"""

import os
import pandas as pd
import streamlit as st
from utils import (
    CANONICAL_COLUMNS,
    DATASETS_DIR,
    apply_custom_css,
    generate_sample_student_dataset,
    get_dataset_info,
    init_session_state,
    list_available_datasets,
    load_dataset_from_dir,
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
    page_title="Upload Dataset | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/2_Upload_Dataset.py")
render_sidebar_status()

render_header(
    title="Upload Dataset",
    subtitle="Upload a student performance CSV file or generate a sample dataset with standard schema columns.",
    badge_text="Module 02",
    badge_variant="indigo",
)

# ---------------------------------------------------------
# Ingestion Flow & Controls
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown("**Ingestion Pipeline**")
    st.caption(
        "Choose an option: Load a pre-generated dataset from datasets/, upload your own CSV file, "
        "or generate a synthetic dataset with configurable parameters."
    )

# 1. Pre-Loaded Datasets from datasets/ Folder
available_csvs = list_available_datasets(DATASETS_DIR)

DATASET_DESCRIPTIONS = {
    "1_raw_dirty_student_data.csv": "500 rows with 5% missing values & 15 duplicate rows (Designed for 3_Data_Cleaning.py demo)",
    "2_cleaned_balanced_cohort.csv": "500 clean rows, zero nulls, realistic distributions (Ideal for 4_EDA, 5_Visualization, 6_Modeling)",
    "3_high_risk_cohort_data.csv": "300 rows skewed low in attendance (< 75%) and study hours (Tests intervention alerts)",
    "4_large_university_batch.csv": "1,200 diverse student records for scalability and ML benchmarking",
    "btech_cse_student_data.csv": "Historical B.Tech CSE cohort records",
}

with st.container(border=True):
    st.markdown("#### 1. Load Pre-Generated Dataset from `datasets/`")
    st.caption("Select any ready-to-use CSV cohort directly from the dedicated datasets folder.")

    if available_csvs:
        c_sel, c_btn = st.columns([2.5, 1.0])
        with c_sel:
            selected_csv = st.selectbox(
                "Select Pre-Loaded CSV File",
                options=available_csvs,
                index=1 if "2_cleaned_balanced_cohort.csv" in available_csvs else 0,
                key="datasets_folder_select",
            )
            desc_text = DATASET_DESCRIPTIONS.get(
                selected_csv, f"Custom dataset file located in {DATASETS_DIR}/"
            )
            st.caption(f"**Cohort Profile:** {desc_text}")
        with c_btn:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            if st.button("Load Selected Dataset", type="primary", width="stretch", key="btn_load_preloaded"):
                try:
                    loaded_df = load_dataset_from_dir(selected_csv, DATASETS_DIR)
                    is_clean = (int(loaded_df.isna().sum().sum()) == 0) and (int(loaded_df.duplicated().sum()) == 0)
                    set_session_dataset(loaded_df, selected_csv, mark_cleaned=is_clean)
                    st.success(f"Successfully loaded '{selected_csv}' ({len(loaded_df):,} rows) into active session.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to load '{selected_csv}': {str(exc)}")
    else:
        st.warning(f"No CSV files found in '{DATASETS_DIR}/' directory.")

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

col_up, col_gen = st.columns([1.5, 1.3])

with col_up:
    with st.container(border=True):
        st.markdown("#### 2. Upload CSV File from Disk")
        st.caption(
            "Expected schema: " + ", ".join(CANONICAL_COLUMNS)
        )
        uploaded_file = st.file_uploader(
            "Select CSV File",
            type=["csv"],
            key=f"csv_uploader_{st.session_state.get('_csv_uploader_id', 0)}",
            help="Upload a comma-separated values (.csv) student records file.",
        )

        if uploaded_file is not None:
            file_signature = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("_last_uploaded_sig") != file_signature:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    set_session_dataset(uploaded_df, uploaded_file.name, mark_cleaned=False)
                    st.session_state["_last_uploaded_sig"] = file_signature
                    st.success(f"Loaded {uploaded_file.name} into session state.")
                except Exception as exc:
                    st.error(f"Failed to read CSV file: {str(exc)}")
        else:
            st.session_state["_last_uploaded_sig"] = None

with col_gen:
    with st.container(border=True):
        st.markdown("#### 3. One-Click Sample Dataset")
        st.caption(
            "Generates a realistic cohort using exact columns: Student ID, Study hrs, Attendance, "
            "Previous marks, Assignment scores, and Final performance (with sample nulls and duplicates for cleaning)."
        )
        sample_rows = st.number_input("Sample Size (Rows)", min_value=50, max_value=1000, value=200, step=50)
        if st.button("Generate Sample Dataset", width="stretch", key="btn_gen_sample"):
            sample_df = generate_sample_student_dataset(
                n_samples=int(sample_rows),
                inject_imperfections=True,
                seed=42,
            )
            set_session_dataset(sample_df, f"sample_student_data_{int(sample_rows)}.csv", mark_cleaned=False)
            st.session_state["_csv_uploader_id"] = st.session_state.get("_csv_uploader_id", 0) + 1
            st.success(f"Generated sample dataset with {len(sample_df)} rows and 6 standard columns.")
            st.rerun()

# ---------------------------------------------------------
# 4 Explicit Dataset Info Items
# ---------------------------------------------------------
raw_df = st.session_state.get("raw_df")

if raw_df is not None:
    info = get_dataset_info(raw_df)
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Dataset Information")

    # 1. Total number of rows | 2. Total number of columns
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Number of Rows", f"{info['rows']:,}")
    m2.metric("Total Number of Columns", f"{info['columns']}")
    m3.metric("Missing Cells", f"{info['missing_total']}")
    m4.metric("Duplicate Rows", f"{info['duplicate_total']}")

    # 3. Column names
    with st.container(border=True):
        st.markdown("**Column Names**")
        col_list_df = pd.DataFrame(
            {
                "Index": list(range(1, len(info["column_names"]) + 1)),
                "Column Name": info["column_names"],
                "Data Type": [str(raw_df[c].dtype) for c in info["column_names"]],
            }
        )
        st.dataframe(col_list_df, width="stretch", hide_index=True)

    # 4. Tabular Dataset Preview
    st.markdown("#### Tabular Dataset Preview")
    st.dataframe(raw_df.head(20), width="stretch")

    c_nav1, c_nav2, _ = st.columns([1.4, 1.4, 3])
    with c_nav1:
        if st.button("Proceed to Data Cleaning", type="primary", width="stretch"):
            st.switch_page("pages/3_Data_Cleaning.py")
    with c_nav2:
        csv_bytes = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Raw CSV",
            data=csv_bytes,
            file_name=st.session_state.get("dataset_name") or "raw_dataset.csv",
            mime="text/csv",
            width="stretch",
        )
else:
    st.info("No dataset is currently loaded. Upload a CSV file or click Generate Sample Dataset above.")

render_footer()
