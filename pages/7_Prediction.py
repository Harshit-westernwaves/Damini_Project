"""
pages/7_Prediction.py - Student Performance Prediction & Academic Guidance
AI Student Performance & Academic Guidance System
"""

import os
import pandas as pd
import streamlit as st
from utils import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    apply_custom_css,
    generate_academic_guidance,
    generate_sample_student_dataset,
    get_active_df,
    get_badge_html,
    init_session_state,
    load_model_from_disk,
    predict_student_performance,
    render_footer,
    render_header,
    render_sidebar_status,
    render_top_navbar,
    save_model_to_disk,
    set_session_dataset,
    train_evaluate_ml_model,
)

# ---------------------------------------------------------
# Page Configuration & Global Shell
# ---------------------------------------------------------
st.set_page_config(
    page_title="Prediction & Guidance | AI Student Performance",
    layout="wide",
)

init_session_state()
apply_custom_css()
render_top_navbar("pages/7_Prediction.py")
render_sidebar_status()

render_header(
    title="Prediction and Academic Guidance",
    subtitle="Enter Study hrs, Attendance, Previous marks, and Assignment scores to predict Final performance and generate an action plan.",
    badge_text="Module 07",
    badge_variant="emerald",
)

# ---------------------------------------------------------
# Model Availability Check (Session State or Disk or Baseline)
# ---------------------------------------------------------
if st.session_state.get("trained_model") is None:
    disk_bundle = load_model_from_disk(DEFAULT_MODEL_PATH)
    if disk_bundle is not None:
        st.session_state["trained_model"] = disk_bundle["model"]
        st.session_state["model_scaler"] = disk_bundle.get("scaler")
        st.session_state["model_type"] = disk_bundle.get("model_type", "Linear Regression")
        st.session_state["model_task"] = disk_bundle.get("task_type", "Regression")
        st.session_state["model_features"] = disk_bundle.get("features", [])
        st.session_state["model_target"] = disk_bundle.get("target_col", "Final performance")
        st.session_state["model_target_meta"] = disk_bundle.get("target_meta", {})
        st.session_state["model_metrics"] = disk_bundle.get("metrics", {})
        st.session_state["model_classes"] = disk_bundle.get("classes")
        st.session_state["model_path"] = DEFAULT_MODEL_PATH

model = st.session_state.get("trained_model")
model_features = st.session_state.get("model_features", [])

if model is None or not model_features:
    with st.container(border=True):
        st.markdown("**No trained model found in session state**")
        st.caption(
            "Train a Linear Regression or Logistic Regression model in Model Training, "
            "or auto-calibrate a baseline Linear Regression model on the standard 6-column dataset."
        )
        c1, c2, _ = st.columns([1.6, 1.5, 2.5])
        with c1:
            if st.button("Train Baseline Linear Regression", type="primary", width="stretch"):
                active_df, _ = get_active_df()
                if active_df is None:
                    if os.path.exists(DEFAULT_DATASET_PATH):
                        active_df = pd.read_csv(DEFAULT_DATASET_PATH)
                        set_session_dataset(active_df, os.path.basename(DEFAULT_DATASET_PATH), mark_cleaned=True)
                    else:
                        active_df = generate_sample_student_dataset(n_samples=200, inject_imperfections=False)
                        set_session_dataset(active_df, "sample_student_data_200.csv", mark_cleaned=True)

                base_feats = [
                    c
                    for c in ["Study hrs", "Attendance", "Previous marks", "Assignment scores"]
                    if c in active_df.columns
                ]
                target = "Final performance" if "Final performance" in active_df.columns else active_df.columns[-1]
                if not base_feats:
                    base_feats = [c for c in active_df.select_dtypes(include="number").columns if c != target][:4]

                res = train_evaluate_ml_model(
                    df=active_df,
                    feature_cols=base_feats,
                    target_col=target,
                    algorithm="Linear Regression",
                    test_size=0.2,
                    use_scaler=True,
                )
                bundle = {
                    "model": res["model"],
                    "scaler": res["scaler"],
                    "model_type": res["algorithm"],
                    "task_type": res["task_type"],
                    "features": res["features"],
                    "target_col": res["target_col"],
                    "target_meta": res["target_meta"],
                    "metrics": res["metrics"],
                    "classes": res["classes"],
                }
                saved_path = save_model_to_disk(bundle, DEFAULT_MODEL_PATH)
                st.session_state["trained_model"] = res["model"]
                st.session_state["model_scaler"] = res["scaler"]
                st.session_state["model_type"] = res["algorithm"]
                st.session_state["model_task"] = res["task_type"]
                st.session_state["model_features"] = res["features"]
                st.session_state["model_target"] = res["target_col"]
                st.session_state["model_target_meta"] = res["target_meta"]
                st.session_state["model_metrics"] = res["metrics"]
                st.session_state["model_classes"] = res["classes"]
                st.session_state["model_path"] = saved_path
                st.rerun()
        with c2:
            if st.button("Open Model Training", width="stretch"):
                st.switch_page("pages/6_Model_Training.py")
else:
    model_type = st.session_state.get("model_type", "Linear Regression")
    task_type = st.session_state.get("model_task", "Regression")
    target_col = st.session_state.get("model_target", "Final performance")

    st.caption(
        f"Active Model: {model_type} ({task_type}) | Target: {target_col} | "
        f"Features: {', '.join(model_features)}"
    )

    # ---------------------------------------------------------
    # Student Input Form (Required 4 Controls + Dynamic Fallback)
    # ---------------------------------------------------------
    active_df, _ = get_active_df()

    with st.form("prediction_input_form"):
        st.markdown("#### Student Input Parameters")
        c1, c2 = st.columns(2)

        with c1:
            study_hrs = st.slider(
                "Study hrs (Daily Hours)",
                min_value=0.0,
                max_value=12.0,
                value=4.5,
                step=0.5,
            )
            previous_marks = st.slider(
                "Previous marks (0 - 100)",
                min_value=0.0,
                max_value=100.0,
                value=68.0,
                step=1.0,
            )

        with c2:
            attendance = st.slider(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=78.0,
                step=1.0,
            )
            assignment_scores = st.slider(
                "Assignment scores (0 - 100)",
                min_value=0.0,
                max_value=100.0,
                value=72.0,
                step=1.0,
            )

        extra_features = [
            f
            for f in model_features
            if f not in {"Study hrs", "Attendance", "Previous marks", "Assignment scores"}
        ]
        extra_inputs = {}
        if extra_features:
            st.markdown("##### Additional Model Features")
            ex_cols = st.columns(2)
            for idx, feat in enumerate(extra_features):
                with ex_cols[idx % 2]:
                    if active_df is not None and feat in active_df.columns:
                        f_min = float(active_df[feat].min())
                        f_max = float(active_df[feat].max())
                        f_med = float(active_df[feat].median())
                    else:
                        f_min, f_max, f_med = 0.0, 100.0, 50.0
                    if f_min >= f_max:
                        f_max = f_min + 1.0
                    extra_inputs[feat] = st.number_input(
                        feat,
                        min_value=f_min,
                        max_value=f_max,
                        value=f_med,
                        key=f"extra_feat_{feat}",
                    )

        predict_btn = st.form_submit_button("Predict Final Performance and Generate Guidance", type="primary", width="stretch")

    # ---------------------------------------------------------
    # Run Prediction & Structured Academic Guidance Plan
    # ---------------------------------------------------------
    if predict_btn:
        input_payload = {
            "Study hrs": study_hrs,
            "Attendance": attendance,
            "Previous marks": previous_marks,
            "Assignment scores": assignment_scores,
            **extra_inputs,
        }

        bundle = {
            "model": model,
            "scaler": st.session_state.get("model_scaler"),
            "features": model_features,
            "task_type": task_type,
        }

        pred_result = predict_student_performance(bundle, input_payload)

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Prediction Output")
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted Final performance", pred_result["display_label"])
        p2.metric("Performance Tier", pred_result["tier"])
        p3.metric("Attendance Gate (>= 75%)", "Eligible" if attendance >= 75.0 else "Below Threshold")

        # Structured Guidance Plan
        st.markdown("#### Structured Academic Guidance Plan")
        guidance_items = generate_academic_guidance(
            study_hrs=study_hrs,
            attendance=attendance,
            previous_marks=previous_marks,
            assignment_scores=assignment_scores,
            predicted_label=pred_result["display_label"],
        )

        for item in guidance_items:
            badge_html = get_badge_html(item["status"], item["variant"])
            steps_html = "".join([f"<li>{step}</li>" for step in item["action_steps"]])
            st.markdown(
                f"""
                <div class="dev-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem; flex-wrap: wrap; gap: 0.5rem;">
                        <strong>{item['area']}</strong>
                        {badge_html}
                    </div>
                    <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">{item['recommendation']}</p>
                    <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.86rem;">
                        {steps_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Downloadable Guidance Plan Report
        report_lines = [
            "# Student Performance Prediction and Guidance Report",
            f"Model Used: {model_type} ({task_type})",
            f"Predicted Final performance: {pred_result['display_label']} ({pred_result['tier']})",
            "",
            "## Input Metrics",
            f"- Study hrs: {study_hrs}",
            f"- Attendance: {attendance}%",
            f"- Previous marks: {previous_marks}",
            f"- Assignment scores: {assignment_scores}",
            "",
            "## Academic Guidance Plan",
        ]
        for item in guidance_items:
            report_lines.append(f"### {item['area']} [{item['status']}]")
            report_lines.append(item["recommendation"])
            for step in item["action_steps"]:
                report_lines.append(f"- {step}")
            report_lines.append("")

        c_exp1, c_exp2, _ = st.columns([1.6, 1.6, 2.5])
        with c_exp1:
            st.download_button(
                label="Download Guidance Plan (.md)",
                data="\n".join(report_lines),
                file_name="academic_guidance_plan.md",
                mime="text/markdown",
                width="stretch",
            )
        with c_exp2:
            if st.button("Discuss with AI Chatbot", width="stretch"):
                st.switch_page("pages/8_AI_Chatbot.py")

render_footer()
