"""
pages/3_Data_Cleaning.py - Structured Data Cleaning Pipeline
AI Student Performance & Academic Guidance System
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from utils import (
    DATASETS_DIR,
    apply_custom_css,
    drop_missing_rows,
    generate_sample_student_dataset,
    get_dataset_info,
    get_missing_summary,
    impute_categorical_missing,
    impute_numerical_missing,
    init_session_state,
    remove_duplicate_records,
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
    page_title="Data Cleaning | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/3_Data_Cleaning.py")
render_sidebar_status()

render_header(
    title="Data Cleaning",
    subtitle="Detect and resolve missing values, remove duplicate records, inspect dataset structure, and export cleaned data.",
    badge_text="Module 03",
    badge_variant="amber",
)

# ---------------------------------------------------------
# Raw Dataset Availability Check
# ---------------------------------------------------------
if st.session_state.get("raw_df") is None:
    with st.container(border=True):
        st.markdown("**No dataset loaded in session state**")
        st.caption("Upload a CSV dataset or load the raw demo dataset before running data cleaning.")
        c1, c2, _ = st.columns([1.6, 1.4, 2.3])
        with c1:
            if st.button("Load Raw Dirty Dataset (Demo)", type="primary", width="stretch"):
                dirty_path = os.path.join(DATASETS_DIR, "1_raw_dirty_student_data.csv")
                if os.path.exists(dirty_path):
                    sample_df = pd.read_csv(dirty_path)
                    set_session_dataset(sample_df, "1_raw_dirty_student_data.csv", mark_cleaned=False)
                else:
                    sample_df = generate_sample_student_dataset(n_samples=200, inject_imperfections=True)
                    set_session_dataset(sample_df, "sample_student_data_200.csv", mark_cleaned=False)
                st.rerun()
        with c2:
            if st.button("Open Upload Dataset", width="stretch"):
                st.switch_page("pages/2_Upload_Dataset.py")
else:
    if st.session_state.get("staging_df") is None:
        st.session_state["staging_df"] = st.session_state["raw_df"].copy()

    staging_df = st.session_state["staging_df"]
    raw_df = st.session_state["raw_df"]

    info_now = get_dataset_info(staging_df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Rows", f"{info_now['rows']:,}", f"Raw: {len(raw_df):,}")
    k2.metric("Current Columns", f"{info_now['columns']}")
    k3.metric("Total Missing Cells", f"{info_now['missing_total']}")
    k4.metric("Duplicate Rows", f"{info_now['duplicate_total']}")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # A) Detect Missing Values
    # ---------------------------------------------------------
    st.markdown("#### A. Detect Missing Values")
    missing_summary_df = get_missing_summary(staging_df)
    st.dataframe(missing_summary_df, width="stretch", hide_index=True)

    # ---------------------------------------------------------
    # B) Handling Missing Values
    # ---------------------------------------------------------
    st.markdown("#### B. Handle Missing Values")
    with st.container(border=True):
        cols_with_nulls = missing_summary_df[missing_summary_df["Missing Count"] > 0]["Column"].tolist()
        if not cols_with_nulls:
            st.info("No missing values remain in the staged dataset.")
        else:
            num_null_cols = [
                c for c in cols_with_nulls if pd.api.types.is_numeric_dtype(staging_df[c])
            ]
            cat_null_cols = [
                c for c in cols_with_nulls if not pd.api.types.is_numeric_dtype(staging_df[c])
            ]

            c_b1, c_b2, c_b3 = st.columns(3)

            with c_b1:
                st.markdown("**1. Remove Rows with Missing Values**")
                st.caption("Drops any record containing one or more missing cells.")
                if st.button("Drop Missing Rows", key="btn_drop_na", width="stretch"):
                    updated_df, dropped = drop_missing_rows(staging_df)
                    st.session_state["staging_df"] = updated_df
                    st.session_state["cleaned_df"] = updated_df.copy()
                    st.session_state["cleaning_history"].append(f"Dropped {dropped} rows containing missing values.")
                    st.success(f"Removed {dropped} rows with missing values and saved to cleaned_df.")
                    st.rerun()

            with c_b2:
                st.markdown("**2. Numerical Missing Values**")
                st.caption(f"Affected numeric columns: {', '.join(num_null_cols) if num_null_cols else 'None'}")
                num_method = st.selectbox(
                    "Numerical Strategy",
                    options=["Mean", "Median"],
                    key="num_impute_strategy",
                )
                if st.button("Apply Numerical Imputation", key="btn_num_impute", type="primary", width="stretch"):
                    updated_df, filled = impute_numerical_missing(staging_df, strategy=num_method.lower())
                    st.session_state["staging_df"] = updated_df
                    st.session_state["cleaned_df"] = updated_df.copy()
                    st.session_state["cleaning_history"].append(
                        f"Imputed {filled} numerical missing cells using {num_method}."
                    )
                    st.success(f"Imputed {filled} numerical cells with {num_method} and saved to cleaned_df.")
                    st.rerun()

            with c_b3:
                st.markdown("**3. Categorical Missing Values**")
                st.caption(f"Affected categorical columns: {', '.join(cat_null_cols) if cat_null_cols else 'None'}")
                st.text_input("Categorical Strategy", value="Mode (Most Frequent)", disabled=True)
                if st.button("Apply Mode Imputation", key="btn_cat_impute", width="stretch"):
                    updated_df, filled = impute_categorical_missing(staging_df)
                    st.session_state["staging_df"] = updated_df
                    st.session_state["cleaned_df"] = updated_df.copy()
                    st.session_state["cleaning_history"].append(
                        f"Imputed {filled} categorical missing cells using Mode."
                    )
                    if filled > 0:
                        st.success(f"Imputed {filled} categorical cells with Mode and saved to cleaned_df.")
                    else:
                        st.info("No categorical missing values found to impute.")
                    st.rerun()

    # ---------------------------------------------------------
    # C) Detecting and Removing Duplicate Records
    # ---------------------------------------------------------
    st.markdown("#### C. Detect and Remove Duplicate Records")
    with st.container(border=True):
        exact_dups = int(staging_df.duplicated().sum())
        id_col = "Student ID" if "Student ID" in staging_df.columns else None
        id_dups = int(staging_df.duplicated(subset=[id_col]).sum()) if id_col else 0

        d_c1, d_c2, d_c3 = st.columns([1.2, 1.2, 1.4])
        d_c1.metric("Exact Duplicate Rows", f"{exact_dups}")
        d_c2.metric("Duplicate Student IDs", f"{id_dups}")
        with d_c3:
            dup_scope = st.selectbox(
                "Duplicate Matching Rule",
                options=["All Columns (Exact Match)"] + (["Student ID"] if id_col else []),
            )

        if exact_dups > 0:
            with st.expander("Preview Duplicate Rows", expanded=False):
                st.dataframe(staging_df[staging_df.duplicated(keep=False)], width="stretch")

        if st.button("Remove Duplicate Records", key="btn_remove_dups", type="primary"):
            subset = [id_col] if dup_scope == "Student ID" and id_col else None
            updated_df, removed = remove_duplicate_records(staging_df, subset=subset, keep="first")
            st.session_state["staging_df"] = updated_df
            st.session_state["cleaned_df"] = updated_df.copy()
            st.session_state["cleaning_history"].append(
                f"Removed {removed} duplicate records ({dup_scope})."
            )
            st.success(f"Removed {removed} duplicate records and saved to cleaned_df.")
            st.rerun()

    # ---------------------------------------------------------
    # D) Checking Dataset Information (Updated Shape & Data Types)
    # ---------------------------------------------------------
    st.markdown("#### D. Check Updated Dataset Information")
    updated_info = get_dataset_info(st.session_state["staging_df"])
    with st.container(border=True):
        st.write(f"**Updated Shape:** {updated_info['rows']} rows x {updated_info['columns']} columns")
        st.dataframe(updated_info["dtypes_df"], width="stretch", hide_index=True)

    # ---------------------------------------------------------
    # Save Session State & Download Cleaned Dataset
    # ---------------------------------------------------------
    st.markdown("#### Save and Export Cleaned Dataset")
    with st.container(border=True):
        st.caption(
            "Save the cleaned DataFrame into st.session_state['cleaned_df'] for downstream EDA, "
            "Visualizations, and Model Training, or download the CSV directly."
        )
        cleaned_export_df = st.session_state["staging_df"]
        c_s1, c_s2, c_s3 = st.columns([1.4, 1.4, 1.4])
        with c_s1:
            if st.button("Save Cleaned Dataset", type="primary", width="stretch"):
                st.session_state["cleaned_df"] = cleaned_export_df.copy()
                st.success(f"Saved {len(cleaned_export_df)} rows to st.session_state['cleaned_df'].")
        with c_s2:
            csv_bytes = cleaned_export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Dataset",
                data=csv_bytes,
                file_name="cleaned_student_dataset.csv",
                mime="text/csv",
                width="stretch",
            )
        with c_s3:
            if st.button("Reset to Raw Dataset", width="stretch"):
                st.session_state["staging_df"] = raw_df.copy()
                st.session_state["cleaned_df"] = None
                st.session_state["cleaning_history"] = []
                st.rerun()

    if st.session_state.get("cleaning_history"):
        with st.expander("Cleaning Audit Log", expanded=False):
            for idx, entry in enumerate(st.session_state["cleaning_history"], 1):
                st.write(f"{idx}. {entry}")

render_footer()
