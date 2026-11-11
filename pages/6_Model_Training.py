"""
pages/6_Model_Training.py - Model Training, Evaluation & Persistence
AI Student Performance & Academic Guidance System
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils import (
    DEFAULT_MODEL_PATH,
    ML_APPROACHES,
    apply_custom_css,
    generate_sample_student_dataset,
    get_active_df,
    get_metric_purpose_table,
    init_session_state,
    plot_model_evaluation,
    render_footer,
    render_header,
    render_sidebar_status,
    render_top_navbar,
    resolve_column,
    save_model_to_disk,
    set_session_dataset,
    train_evaluate_ml_model,
)

# ---------------------------------------------------------
# Page Configuration & Global Shell
# ---------------------------------------------------------
st.set_page_config(
    page_title="Model Training | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/6_Model_Training.py")
render_sidebar_status()

render_header(
    title="Model Training",
    subtitle="Configure features and target, split train/test sets, fit Linear or Logistic Regression, evaluate metrics, and save via Joblib.",
    badge_text="Module 06",
    badge_variant="teal",
)

# ---------------------------------------------------------
# Overview of the 3 Main Machine Learning Approaches
# ---------------------------------------------------------
st.markdown("#### Overview of Machine Learning Approaches")
ap_cols = st.columns(3)
for col, item in zip(ap_cols, ML_APPROACHES):
    with col:
        with st.container(border=True):
            st.markdown(f"**{item['Approach']}**")
            st.caption(item["Definition"])
            st.write(f"System Role: {item['Application in System']}")

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Active Dataset Verification
# ---------------------------------------------------------
df, is_cleaned = get_active_df()

if df is None:
    with st.container(border=True):
        st.markdown("**No active dataset loaded in session state**")
        st.caption("Upload or generate a student dataset before training a predictive model.")
        c1, c2, _ = st.columns([1.4, 1.4, 2.5])
        with c1:
            if st.button("Generate Sample Dataset", type="primary", width="stretch"):
                sample_df = generate_sample_student_dataset(n_samples=200, inject_imperfections=False)
                set_session_dataset(sample_df, "sample_student_data_200.csv", mark_cleaned=True)
                st.rerun()
        with c2:
            if st.button("Open Upload Dataset", width="stretch"):
                st.switch_page("pages/2_Upload_Dataset.py")
else:
    st.caption(
        f"Training source: {'cleaned_df' if is_cleaned else 'raw_df'} "
        f"({len(df):,} rows, {len(df.columns)} columns)."
    )

    all_cols = df.columns.tolist()
    default_target = resolve_column(df, ["Final performance", "Final_CGPA"]) or all_cols[-1]
    id_col = resolve_column(df, ["Student ID", "Student_ID"])

    # ---------------------------------------------------------
    # Steps A, B, C: Target, Features, Split & Algorithm
    # ---------------------------------------------------------
    st.markdown("#### Model Configuration Workflow")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns([1.2, 1.8, 1.2])

        with col_a:
            st.markdown("**A. Target Column**")
            target_col = st.selectbox(
                "Select Target Column (y)",
                options=all_cols,
                index=all_cols.index(default_target),
            )

        candidate_features = [c for c in all_cols if c != target_col and c != id_col]
        preferred_defaults = [
            c
            for c in ["Study hrs", "Attendance", "Previous marks", "Assignment scores"]
            if c in candidate_features
        ]
        default_features = preferred_defaults if preferred_defaults else candidate_features

        with col_b:
            st.markdown("**A. Feature Columns (X)**")
            selected_features = st.multiselect(
                "Select Input Features",
                options=candidate_features,
                default=default_features,
            )

        with col_c:
            st.markdown("**B & C. Split and Algorithm**")
            test_split_pct = st.slider("Test Split (%)", min_value=10, max_value=40, value=20, step=5)
            algorithm = st.selectbox(
                "Select ML Algorithm",
                options=["Linear Regression", "Logistic Regression"],
            )
            use_scaler = st.checkbox("Standardize Features (StandardScaler)", value=True)

        train_clicked = st.button("Train and Evaluate Model", type="primary", width="stretch")

    # ---------------------------------------------------------
    # Step D: Training & Persistence (Session State + Disk)
    # ---------------------------------------------------------
    if train_clicked:
        if not selected_features:
            st.error("Select at least one feature column (X) to train the model.")
        else:
            try:
                result = train_evaluate_ml_model(
                    df=df,
                    feature_cols=selected_features,
                    target_col=target_col,
                    algorithm=algorithm,
                    test_size=test_split_pct / 100.0,
                    use_scaler=use_scaler,
                    random_state=42,
                )

                bundle = {
                    "model": result["model"],
                    "scaler": result["scaler"],
                    "model_type": result["algorithm"],
                    "task_type": result["task_type"],
                    "features": result["features"],
                    "target_col": result["target_col"],
                    "target_meta": result["target_meta"],
                    "metrics": result["metrics"],
                    "classes": result["classes"],
                }
                saved_path = save_model_to_disk(bundle, DEFAULT_MODEL_PATH)

                st.session_state["trained_model"] = result["model"]
                st.session_state["model_scaler"] = result["scaler"]
                st.session_state["model_type"] = result["algorithm"]
                st.session_state["model_task"] = result["task_type"]
                st.session_state["model_features"] = result["features"]
                st.session_state["model_target"] = result["target_col"]
                st.session_state["model_target_meta"] = result["target_meta"]
                st.session_state["model_metrics"] = result["metrics"]
                st.session_state["model_classes"] = result["classes"]
                st.session_state["model_eval_data"] = {
                    "y_test": result["y_test"],
                    "y_pred": result["y_pred"],
                    "train_size": result["train_size"],
                    "test_size": result["test_size"],
                    "task_type": result["task_type"],
                    "classes": result["classes"],
                }
                st.session_state["model_path"] = saved_path
                st.success(
                    f"Trained {algorithm} ({result['train_size']} train / {result['test_size']} test rows) "
                    f"and saved to session state and disk ({saved_path})."
                )
            except Exception as exc:
                st.error(f"Model training failed: {str(exc)}")

    # ---------------------------------------------------------
    # Step E & F: Model Evaluation & Saved Model Status
    # ---------------------------------------------------------
    if st.session_state.get("trained_model") is not None:
        m_type = st.session_state.get("model_type", "Linear Regression")
        m_task = st.session_state.get("model_task", "Regression")
        m_metrics = st.session_state.get("model_metrics", {})
        m_meta = st.session_state.get("model_target_meta", {})
        eval_data = st.session_state.get("model_eval_data", {})
        saved_path = st.session_state.get("model_path")

        st.markdown("#### E. Model Evaluation")
        if m_meta.get("note"):
            st.info(m_meta["note"])

        metric_cols = st.columns(max(len(m_metrics), 1) + 2)
        for idx, (k, v) in enumerate(m_metrics.items()):
            metric_cols[idx].metric(k, f"{v:.4f}")
        metric_cols[-2].metric("Training Rows", f"{eval_data.get('train_size', '-')}")
        metric_cols[-1].metric("Testing Rows", f"{eval_data.get('test_size', '-')}")

        c_eval_tbl, c_eval_plot = st.columns([1.1, 1.3])
        with c_eval_tbl:
            with st.container(border=True):
                st.markdown(f"**Metric and Purpose ({m_type})**")
                st.dataframe(get_metric_purpose_table(m_type), width="stretch", hide_index=True)

                st.markdown("**F. Saved Model Status**")
                st.write(f"- Session state: `st.session_state['trained_model']` ({m_type})")
                st.write(f"- Features: {', '.join(st.session_state.get('model_features', []))}")
                if saved_path:
                    st.write(f"- Disk artifact: `{saved_path}`")
                    with open(saved_path, "rb") as f_model:
                        st.download_button(
                            label="Download Saved Model (.joblib)",
                            data=f_model.read(),
                            file_name="student_performance_model.joblib",
                            mime="application/octet-stream",
                            width="stretch",
                        )

        with c_eval_plot:
            if eval_data.get("y_test") is not None:
                with st.container(border=True):
                    st.markdown("**Diagnostic Plot (Test Set)**")
                    fig_eval = plot_model_evaluation(
                        {
                            "task_type": m_task,
                            "y_test": eval_data["y_test"],
                            "y_pred": eval_data["y_pred"],
                            "classes": eval_data.get("classes"),
                        }
                    )
                    st.pyplot(fig_eval, width="stretch")
                    plt.close(fig_eval)

        c_nav1, c_nav2, _ = st.columns([1.5, 1.5, 3])
        with c_nav1:
            if st.button("Proceed to Prediction & Guidance", type="primary", width="stretch"):
                st.switch_page("pages/7_Prediction.py")
        with c_nav2:
            if st.button("Return to Data Visualization", width="stretch"):
                st.switch_page("pages/5_Data_Visualization.py")

render_footer()
