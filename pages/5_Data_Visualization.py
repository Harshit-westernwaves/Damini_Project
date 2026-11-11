"""
pages/5_Data_Visualization.py - Matplotlib & Seaborn Data Visualization
AI Student Performance & Academic Guidance System
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from utils import (
    apply_custom_css,
    generate_sample_student_dataset,
    get_active_df,
    init_session_state,
    render_feature_chart,
    render_footer,
    render_header,
    render_sidebar_status,
    render_top_navbar,
    resolve_column,
    set_session_dataset,
)

# ---------------------------------------------------------
# Page Configuration & Global Shell
# ---------------------------------------------------------
st.set_page_config(
    page_title="Data Visualization | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/5_Data_Visualization.py")
render_sidebar_status()

render_header(
    title="Data Visualization",
    subtitle="Explore numeric and categorical student attributes using Matplotlib and Seaborn charts.",
    badge_text="Module 05",
    badge_variant="teal",
)

# ---------------------------------------------------------
# Step 1: Load Student Dataset
# ---------------------------------------------------------
df, is_cleaned = get_active_df()

if df is None:
    with st.container(border=True):
        st.markdown("**No active dataset loaded in session state**")
        st.caption("Upload a dataset or generate the sample dataset to create Matplotlib and Seaborn charts.")
        c1, c2, _ = st.columns([1.4, 1.4, 2.5])
        with c1:
            if st.button("Generate Sample Dataset", type="primary", width="stretch"):
                sample_df = generate_sample_student_dataset(n_samples=200, inject_imperfections=True)
                set_session_dataset(sample_df, "sample_student_data_200.csv", mark_cleaned=False)
                st.rerun()
        with c2:
            if st.button("Open Upload Dataset", width="stretch"):
                st.switch_page("pages/2_Upload_Dataset.py")
else:
    with st.container(border=True):
        st.markdown("**Visualization Flow**")
        st.caption(
            "Load student dataset -> Select numeric or categorical feature via dropdown -> "
            "Generate graph/chart -> Display visual insights and patterns"
        )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    default_target = resolve_column(df, ["Final performance", "Final_CGPA"]) or (num_cols[-1] if num_cols else None)

    # ---------------------------------------------------------
    # Step 2: Select Numeric or Categorical Feature via Dropdown
    # ---------------------------------------------------------
    c_ctrl, c_view = st.columns([1.1, 2.2])

    with c_ctrl:
        with st.container(border=True):
            st.markdown("#### Chart Configuration")
            feature_group = st.radio(
                "Feature Type",
                options=["Numeric", "Categorical"] if (num_cols and cat_cols) else (["Numeric"] if num_cols else ["Categorical"]),
                horizontal=True,
            )

            if feature_group == "Numeric":
                selected_feature = st.selectbox(
                    "Select Numeric Feature",
                    options=num_cols,
                    index=0,
                )
                chart_options = ["Histogram with KDE", "Box Plot", "Scatter vs Target", "Correlation Heatmap"]
            else:
                selected_feature = st.selectbox(
                    "Select Categorical Feature",
                    options=cat_cols,
                    index=0,
                )
                chart_options = ["Count Plot", "Category Mean vs Target"]

            selected_chart = st.selectbox("Select Chart Type", options=chart_options)

            target_choice = default_target
            if selected_chart in {"Scatter vs Target", "Category Mean vs Target"} and num_cols:
                other_nums = [c for c in num_cols if c != selected_feature] or num_cols
                default_idx = other_nums.index(default_target) if default_target in other_nums else 0
                target_choice = st.selectbox("Comparison Target", options=other_nums, index=default_idx)

    # ---------------------------------------------------------
    # Step 3 & 4: Generate Chart & Display Visual Insights
    # ---------------------------------------------------------
    with c_view:
        with st.container(border=True):
            st.markdown(f"#### {selected_chart}: {selected_feature}")
            fig, insight_text = render_feature_chart(
                df=df,
                feature=selected_feature,
                chart_type=selected_chart,
                target_col=target_choice,
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with st.container(border=True):
            st.markdown("**Visual Insights and Patterns**")
            st.write(insight_text)
            if pd.api.types.is_numeric_dtype(df[selected_feature]):
                s = df[selected_feature].dropna()
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Mean", f"{s.mean():.2f}")
                s2.metric("Median", f"{s.median():.2f}")
                s3.metric("Min", f"{s.min():.2f}")
                s4.metric("Max", f"{s.max():.2f}")

    # ---------------------------------------------------------
    # Quick Multi-Feature Overview Grid (Matplotlib & Seaborn)
    # ---------------------------------------------------------
    if len(num_cols) >= 2:
        st.markdown("#### Pairwise Feature Overview")
        ov_c1, ov_c2 = st.columns(2)
        study_col = resolve_column(df, ["Study hrs", "Study_Hours_Per_Day"]) or num_cols[0]
        att_col = resolve_column(df, ["Attendance", "Attendance_Percentage"]) or num_cols[min(1, len(num_cols) - 1)]

        with ov_c1:
            with st.container(border=True):
                fig_s, ins_s = render_feature_chart(df, study_col, "Scatter vs Target", target_col=default_target)
                st.pyplot(fig_s, width="stretch")
                plt.close(fig_s)
                st.caption(ins_s)
        with ov_c2:
            with st.container(border=True):
                fig_a, ins_a = render_feature_chart(df, att_col, "Scatter vs Target", target_col=default_target)
                st.pyplot(fig_a, width="stretch")
                plt.close(fig_a)
                st.caption(ins_a)

    c_nav1, c_nav2, _ = st.columns([1.4, 1.4, 3])
    with c_nav1:
        if st.button("Proceed to Model Training", type="primary", width="stretch"):
            st.switch_page("pages/6_Model_Training.py")
    with c_nav2:
        if st.button("Return to EDA", width="stretch"):
            st.switch_page("pages/4_EDA.py")

render_footer()
