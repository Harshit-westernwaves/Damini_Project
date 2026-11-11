"""
utils.py - Shared configuration, UI components, data processing, visualization,
machine learning, and Groq AI assistant helpers for the AI Student Performance
& Academic Guidance System.
"""

import os
import re
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------
# Theme Palette Constants (Clean Daylight Developer Theme)
# ---------------------------------------------------------
COLOR_BG = "#FAFAFA"
COLOR_SIDEBAR = "#F4F4F5"
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#E4E4E7"
COLOR_TEXT = "#18181B"
COLOR_MUTED = "#71717A"
COLOR_PRIMARY = "#0F766E"
COLOR_PRIMARY_HOVER = "#115E59"

COLOR_TEAL = "#0F766E"
COLOR_SLATE = "#64748B"
COLOR_CORAL = "#DC2626"
COLOR_AMBER = "#D97706"
COLOR_EMERALD = "#059669"

CANONICAL_COLUMNS = [
    "Student ID",
    "Study hrs",
    "Attendance",
    "Previous marks",
    "Assignment scores",
    "Final performance",
]

DEFAULT_MODEL_PATH = os.path.join("models", "student_performance_model.joblib")

NAV_ITEMS = [
    ("Welcome", "app.py"),
    ("Home", "pages/1_Home.py"),
    ("Upload", "pages/2_Upload_Dataset.py"),
    ("Clean", "pages/3_Data_Cleaning.py"),
    ("EDA", "pages/4_EDA.py"),
    ("Visualize", "pages/5_Data_Visualization.py"),
    ("Train ML", "pages/6_Model_Training.py"),
    ("Predict", "pages/7_Prediction.py"),
    ("AI Chat", "pages/8_AI_Chatbot.py"),
    ("About", "pages/9_About_Project.py"),
]


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def init_session_state():
    """Initializes shared session state keys if not already present."""
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "staging_df": None,
        "dataset_name": None,
        "dataset_timestamp": None,
        "_last_uploaded_sig": None,
        "_csv_uploader_id": 0,
        "cleaning_history": [],
        "trained_model": None,
        "model_type": None,
        "model_task": "Regression",
        "model_features": [],
        "model_target": None,
        "model_metrics": {},
        "model_scaler": None,
        "model_classes": None,
        "model_target_meta": {},
        "model_eval_data": {},
        "model_path": None,
        "groq_api_key": "",
        "last_chat_source": None,
        "chat_history": [
            {
                "role": "assistant",
                "content": (
                    "Academic guidance assistant ready. Ask questions about study planning, "
                    "attendance improvement, machine learning concepts, or dataset insights."
                ),
            }
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------
# Clean Daylight CSS Injection (GitHub / Linear Style)
# ---------------------------------------------------------
def apply_custom_css():
    """Applies flat daylight styling with crisp 1px borders and 6px radius."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                color: {COLOR_TEXT};
            }}

            .stApp {{
                background-color: {COLOR_BG} !important;
            }}

            #MainMenu {{ visibility: hidden; }}
            footer {{ visibility: hidden; }}
            header [data-testid="stDecoration"] {{ display: none; }}
            [data-testid="stToolbar"] {{ visibility: hidden; }}

            [data-testid="stSidebar"] {{
                background-color: {COLOR_SIDEBAR} !important;
                border-right: 1px solid {COLOR_BORDER} !important;
            }}

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
                color: {COLOR_MUTED};
                font-size: 0.85rem;
            }}

            h1, h2, h3, h4, h5, h6 {{
                color: {COLOR_TEXT} !important;
                font-weight: 600;
                letter-spacing: -0.015em;
            }}

            /* Flat White Cards and Containers */
            [data-testid="stVerticalBlockBorderWrapper"] {{
                background-color: {COLOR_CARD} !important;
                border: 1px solid {COLOR_BORDER} !important;
                border-radius: 6px !important;
                box-shadow: none !important;
            }}

            .dev-card {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                padding: 1rem 1.25rem;
                margin-bottom: 0.85rem;
                box-shadow: none;
            }}

            .kpi-card {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                padding: 0.85rem 1rem;
                box-shadow: none;
                text-align: left;
            }}

            .kpi-label {{
                font-size: 0.74rem;
                font-weight: 600;
                color: {COLOR_MUTED};
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.25rem;
            }}

            .kpi-value {{
                font-size: 1.45rem;
                font-weight: 700;
                color: {COLOR_TEXT};
                line-height: 1.2;
                letter-spacing: -0.02em;
            }}

            .kpi-subtext {{
                font-size: 0.78rem;
                color: {COLOR_MUTED};
                margin-top: 0.25rem;
            }}

            [data-testid="stMetric"] {{
                background-color: {COLOR_CARD} !important;
                border: 1px solid {COLOR_BORDER} !important;
                border-radius: 6px !important;
                padding: 0.75rem 0.95rem !important;
                box-shadow: none !important;
            }}

            .status-badge {{
                display: inline-block;
                padding: 0.15rem 0.5rem;
                border-radius: 4px;
                font-size: 0.72rem;
                font-weight: 500;
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: 0.02em;
                white-space: nowrap;
            }}

            .badge-emerald, .badge-teal {{
                background-color: #F0FDFA;
                color: #0F766E;
                border: 1px solid #99F6E4;
            }}

            .badge-amber {{
                background-color: #FFFBEB;
                color: #92400E;
                border: 1px solid #FDE68A;
            }}

            .badge-rose {{
                background-color: #FEF2F2;
                color: #991B1B;
                border: 1px solid #FECACA;
            }}

            .badge-slate {{
                background-color: #F4F4F5;
                color: #3F3F46;
                border: 1px solid #E4E4E7;
            }}

            /* Buttons (Flat, crisp border, Deep Teal primary) */
            .stButton > button, .stDownloadButton > button {{
                background-color: #FFFFFF !important;
                color: {COLOR_TEXT} !important;
                font-weight: 500 !important;
                font-size: 0.84rem !important;
                border-radius: 6px !important;
                border: 1px solid {COLOR_BORDER} !important;
                padding: 0.4rem 0.8rem !important;
                box-shadow: none !important;
            }}

            .stButton > button:hover, .stDownloadButton > button:hover {{
                border-color: #A1A1AA !important;
                background-color: #F4F4F5 !important;
            }}

            .stButton > button[kind="primary"] {{
                background-color: {COLOR_PRIMARY} !important;
                color: #FFFFFF !important;
                border: 1px solid {COLOR_PRIMARY} !important;
            }}

            .stButton > button[kind="primary"]:hover {{
                background-color: {COLOR_PRIMARY_HOVER} !important;
                border-color: {COLOR_PRIMARY_HOVER} !important;
            }}

            /* Form Controls & Tables */
            .stTextInput input, .stSelectbox select, .stNumberInput input {{
                background-color: #FFFFFF !important;
                color: {COLOR_TEXT} !important;
                border: 1px solid {COLOR_BORDER} !important;
                border-radius: 6px !important;
                font-size: 0.85rem !important;
            }}

            [data-testid="stDataFrame"], [data-testid="stTable"] {{
                border: 1px solid {COLOR_BORDER} !important;
                border-radius: 6px !important;
                background-color: #FFFFFF !important;
            }}

            code {{
                font-family: 'JetBrains Mono', monospace !important;
                background-color: #F4F4F5 !important;
                color: {COLOR_TEXT} !important;
                padding: 0.1rem 0.3rem !important;
                border-radius: 4px !important;
                border: 1px solid {COLOR_BORDER} !important;
                font-size: 0.84em !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Shared Navigation, Header, Sidebar & Footer Components
# ---------------------------------------------------------
def get_badge_html(text: str, variant: str = "slate") -> str:
    """Returns HTML markup for a flat status badge."""
    valid_variants = {"emerald", "teal", "amber", "rose", "slate"}
    v = variant.lower() if variant.lower() in valid_variants else "slate"
    return f'<span class="status-badge badge-{v}">{text}</span>'


def render_top_navbar(active_page: str = "app.py"):
    """Renders the shared horizontal navigation button bar at the top of every page."""
    cols = st.columns(len(NAV_ITEMS))
    for idx, (label, target_path) in enumerate(NAV_ITEMS):
        with cols[idx]:
            btn_type = "primary" if active_page == target_path else "secondary"
            if st.button(
                label,
                key=f"topnav_{idx}_{active_page}",
                type=btn_type,
                width="stretch",
            ):
                if active_page != target_path:
                    st.switch_page(target_path)
    st.markdown(
        f"<div style='border-bottom: 1px solid {COLOR_BORDER}; margin: 0.35rem 0 1rem 0;'></div>",
        unsafe_allow_html=True,
    )


def render_header(
    title: str,
    subtitle: str = None,
    badge_text: str = None,
    badge_variant: str = "slate",
):
    """Renders a clean page header banner with hairline bottom border."""
    badge_markup = f"<div>{get_badge_html(badge_text, badge_variant)}</div>" if badge_text else ""
    sub_markup = (
        f'<div style="font-size: 0.86rem; color: {COLOR_MUTED}; margin-top: 0.25rem;">{subtitle}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="border-bottom: 1px solid {COLOR_BORDER}; padding-bottom: 0.8rem; margin-bottom: 1.2rem; display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <h2 style="margin: 0; font-size: 1.35rem; font-weight: 600; color: {COLOR_TEXT}; letter-spacing: -0.015em;">{title}</h2>
                {sub_markup}
            </div>
            {badge_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_status_badge() -> str:
    """Returns a status badge HTML string reflecting active dataset state."""
    if st.session_state.get("cleaned_df") is not None:
        count = len(st.session_state["cleaned_df"])
        return get_badge_html(f"cleaned_df: {count} rows", "teal")
    if st.session_state.get("raw_df") is not None:
        count = len(st.session_state["raw_df"])
        return get_badge_html(f"raw_df: {count} rows", "amber")
    return get_badge_html("no dataset loaded", "slate")


def render_sidebar_status():
    """Renders persistent workspace status inside the sidebar."""
    model_type = st.session_state.get("model_type")
    model_badge = (
        get_badge_html(f"Model: {model_type}", "teal")
        if model_type
        else get_badge_html("Model: Not trained", "slate")
    )
    dataset_name = st.session_state.get("dataset_name") or "None"

    with st.sidebar:
        st.markdown(
            f"""
            <div style="background-color: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.75rem 0.85rem; margin-bottom: 1rem;">
                <div style="font-size: 0.7rem; text-transform: uppercase; color: {COLOR_MUTED}; font-weight: 600; letter-spacing: 0.04em; margin-bottom: 0.4rem;">
                    Workspace Status
                </div>
                <div style="margin-bottom: 0.45rem;">
                    {render_dataset_status_badge()}
                </div>
                <div style="margin-bottom: 0.45rem;">
                    {model_badge}
                </div>
                <div style="font-size: 0.75rem; color: {COLOR_MUTED}; line-height: 1.4;">
                    File: {dataset_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    """Renders a consistent footer at the bottom of every page."""
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="border-top: 1px solid {COLOR_BORDER}; padding-top: 0.75rem; margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; color: {COLOR_MUTED}; font-size: 0.76rem;">
            <div>AI Student Performance &amp; Academic Guidance System</div>
            <div>Python | Streamlit | Pandas | Scikit-learn | Matplotlib | Seaborn | Groq AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_active_df():
    """Returns (active_df, is_cleaned) prioritizing cleaned_df over raw_df."""
    if st.session_state.get("cleaned_df") is not None:
        return st.session_state["cleaned_df"], True
    if st.session_state.get("raw_df") is not None:
        return st.session_state["raw_df"], False
    return None, False


def set_session_dataset(df: pd.DataFrame, dataset_name: str, mark_cleaned: bool = False):
    """Stores a dataset in session state and resets derived cleaning and upload-tracking state."""
    st.session_state["raw_df"] = df
    st.session_state["staging_df"] = df.copy()
    st.session_state["cleaned_df"] = df.copy() if mark_cleaned else None
    st.session_state["cleaning_history"] = []
    st.session_state["dataset_name"] = dataset_name
    st.session_state["dataset_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["_last_uploaded_sig"] = None


# ---------------------------------------------------------
# Sample Dataset Generator (Exact 6-Column Specification)
# ---------------------------------------------------------
def generate_sample_student_dataset(
    n_samples: int = 200,
    inject_imperfections: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a student performance dataset with exact specification columns:
    Student ID, Study hrs, Attendance, Previous marks, Assignment scores, Final performance.
    """
    rng = np.random.default_rng(seed)

    student_ids = [f"STU{str(i).zfill(3)}" for i in range(1, n_samples + 1)]
    study_hrs = np.clip(np.round(rng.normal(loc=4.8, scale=1.8, size=n_samples), 1), 0.5, 10.0)
    attendance = np.clip(np.round(rng.beta(a=5.2, b=1.8, size=n_samples) * 58 + 42, 1), 42.0, 100.0)
    previous_marks = np.clip(np.round(rng.normal(loc=70.0, scale=12.0, size=n_samples), 1), 32.0, 98.0)
    assignment_raw = (
        0.45 * previous_marks
        + 0.25 * attendance
        + 2.4 * study_hrs
        + rng.normal(loc=6.0, scale=6.0, size=n_samples)
    )
    assignment_scores = np.clip(np.round(assignment_raw, 1), 35.0, 100.0)

    final_raw = (
        0.38 * previous_marks
        + 0.28 * assignment_scores
        + 0.18 * attendance
        + 2.2 * study_hrs
        + rng.normal(loc=0.0, scale=3.2, size=n_samples)
    )
    final_performance = np.clip(np.round(final_raw, 1), 30.0, 99.5)

    df = pd.DataFrame(
        {
            "Student ID": student_ids,
            "Study hrs": study_hrs,
            "Attendance": attendance,
            "Previous marks": previous_marks,
            "Assignment scores": assignment_scores,
            "Final performance": final_performance,
        }
    )

    if inject_imperfections and n_samples >= 20:
        num_cols = ["Study hrs", "Attendance", "Previous marks", "Assignment scores", "Final performance"]
        for col in num_cols:
            nan_idx = rng.choice(n_samples, size=max(2, int(n_samples * 0.03)), replace=False)
            df.loc[nan_idx, col] = np.nan

        dup_count = min(5, max(2, n_samples // 40))
        dup_idx = rng.choice(n_samples, size=dup_count, replace=False)
        df = pd.concat([df, df.iloc[dup_idx].copy()], ignore_index=True)

    return df


# ---------------------------------------------------------
# Data Cleaning Helpers
# ---------------------------------------------------------
def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns per-column null count, percentage, and data type."""
    total_rows = max(len(df), 1)
    rows = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Missing Count": null_count,
                "Missing (%)": round((null_count / total_rows) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def drop_missing_rows(df: pd.DataFrame, columns: list | None = None) -> tuple[pd.DataFrame, int]:
    """Removes rows containing missing values across specified or all columns."""
    before = len(df)
    cleaned = df.dropna(subset=columns).reset_index(drop=True) if columns else df.dropna().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def impute_numerical_missing(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: list | None = None,
) -> tuple[pd.DataFrame, int]:
    """Replaces missing numerical values with Mean or Median."""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    targets = [c for c in (columns or num_cols) if c in num_cols]
    filled_total = 0

    for col in targets:
        null_count = int(out[col].isna().sum())
        if null_count == 0:
            continue
        series = out[col].dropna()
        if series.empty:
            continue
        fill_val = series.median() if strategy.lower() == "median" else series.mean()
        out[col] = out[col].fillna(round(float(fill_val), 2))
        filled_total += null_count

    return out, filled_total


def impute_categorical_missing(
    df: pd.DataFrame,
    columns: list | None = None,
) -> tuple[pd.DataFrame, int]:
    """Replaces missing categorical (non-numeric) values with Mode."""
    out = df.copy()
    cat_cols = out.select_dtypes(exclude=[np.number]).columns.tolist()
    targets = [c for c in (columns or cat_cols) if c in cat_cols]
    filled_total = 0

    for col in targets:
        null_count = int(out[col].isna().sum())
        if null_count == 0:
            continue
        mode_vals = out[col].dropna().mode()
        if len(mode_vals) > 0:
            out[col] = out[col].fillna(mode_vals.iloc[0])
            filled_total += null_count

    return out, filled_total


def remove_duplicate_records(
    df: pd.DataFrame,
    subset: list | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, int]:
    """Detects and removes duplicate records."""
    before = len(df)
    cleaned = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    return cleaned, before - len(cleaned)


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Returns structural metadata including shape, column types, nulls, and duplicates."""
    dtypes_df = pd.DataFrame(
        {
            "Column": df.columns.tolist(),
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Non-Null Count": [int(df[c].notna().sum()) for c in df.columns],
            "Null Count": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes_df": dtypes_df,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_total": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------
# EDA & Column Resolution Helpers
# ---------------------------------------------------------
def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Finds the first matching column name in df using exact or case-insensitive lookup."""
    if df is None or df.empty:
        return None
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower().strip() in lower_map:
            return lower_map[cand.lower().strip()]
    return None


def get_statistical_glossary() -> pd.DataFrame:
    """Returns the standard statistical summary reference table."""
    return pd.DataFrame(
        [
            {"Statistical Term": "Count", "Meaning": "Number of available values"},
            {"Statistical Term": "Mean", "Meaning": "Average value"},
            {"Statistical Term": "Standard deviation", "Meaning": "Measures variation in data"},
            {"Statistical Term": "Minimum", "Meaning": "Smallest value"},
            {"Statistical Term": "Maximum", "Meaning": "Largest value"},
            {"Statistical Term": "25%, 50%, 75%", "Meaning": "Quartile values"},
        ]
    )


def compute_eda_questions(df: pd.DataFrame) -> dict:
    """
    Computes answers to the 4 core analytical questions from the active dataset:
    1) How many students are included?
    2) What are their average marks?
    3) How many students have high attendance (>= 75%)?
    4) Is there any relationship between study hours and academic performance?
    """
    total_students = int(len(df))

    prev_col = resolve_column(df, ["Previous marks", "MST_Marks", "Previous_SGPA"])
    assign_col = resolve_column(df, ["Assignment scores", "Lab_Score"])
    perf_col = resolve_column(df, ["Final performance", "Final_CGPA"])
    att_col = resolve_column(df, ["Attendance", "Attendance_Percentage"])
    study_col = resolve_column(df, ["Study hrs", "Study_Hours_Per_Day"])

    avg_marks_map = {}
    for label, col in [
        ("Previous marks", prev_col),
        ("Assignment scores", assign_col),
        ("Final performance", perf_col),
    ]:
        if col and pd.api.types.is_numeric_dtype(df[col]):
            avg_marks_map[label] = round(float(df[col].dropna().mean()), 2)

    if avg_marks_map:
        primary_avg_marks = avg_marks_map.get("Final performance") or next(iter(avg_marks_map.values()))
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        primary_avg_marks = round(float(df[num_cols[0]].dropna().mean()), 2) if num_cols else 0.0

    if att_col and pd.api.types.is_numeric_dtype(df[att_col]):
        valid_att = df[att_col].dropna()
        high_att_count = int((valid_att >= 75.0).sum())
        high_att_pct = round((high_att_count / max(len(valid_att), 1)) * 100, 1)
    else:
        high_att_count = 0
        high_att_pct = 0.0

    corr_r = None
    rel_strength = "Unavailable"
    rel_summary = "Study hours and numeric performance columns were not found."
    if (
        study_col
        and perf_col
        and pd.api.types.is_numeric_dtype(df[study_col])
        and pd.api.types.is_numeric_dtype(df[perf_col])
    ):
        pair_df = df[[study_col, perf_col]].dropna()
        if len(pair_df) >= 2 and pair_df[study_col].std() > 0 and pair_df[perf_col].std() > 0:
            corr_r = round(float(pair_df[study_col].corr(pair_df[perf_col])), 3)
            abs_r = abs(corr_r)
            if abs_r >= 0.7:
                rel_strength = "Strong"
            elif abs_r >= 0.4:
                rel_strength = "Moderate"
            elif abs_r >= 0.2:
                rel_strength = "Weak"
            else:
                rel_strength = "Negligible"
            direction = "positive" if corr_r >= 0 else "negative"
            rel_summary = (
                f"Yes. There is a {rel_strength.lower()} {direction} correlation "
                f"(r = {corr_r:.3f}) between {study_col} and {perf_col}."
            )

    return {
        "student_count": total_students,
        "avg_marks_map": avg_marks_map,
        "primary_avg_marks": primary_avg_marks,
        "high_attendance_count": high_att_count,
        "high_attendance_pct": high_att_pct,
        "attendance_col": att_col,
        "study_col": study_col,
        "perf_col": perf_col,
        "correlation_r": corr_r,
        "relationship_strength": rel_strength,
        "relationship_summary": rel_summary,
    }


# ---------------------------------------------------------
# Matplotlib & Seaborn Visualization Helpers (Zero Plotly)
# ---------------------------------------------------------
def _style_matplotlib_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """Applies clean daylight developer styling to a Matplotlib axis."""
    ax.set_facecolor(COLOR_CARD)
    ax.set_title(title, fontsize=11, fontweight="semibold", color=COLOR_TEXT, pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=COLOR_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=COLOR_TEXT)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8.5)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLOR_BORDER, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(COLOR_BORDER)


def render_feature_chart(
    df: pd.DataFrame,
    feature: str,
    chart_type: str,
    target_col: str | None = None,
) -> tuple[plt.Figure, str]:
    """
    Generates a Matplotlib/Seaborn figure and computed visual insight text
    for a selected numeric or categorical feature.
    """
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": COLOR_BORDER, "figure.facecolor": COLOR_CARD})
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=COLOR_CARD)
    is_numeric = pd.api.types.is_numeric_dtype(df[feature])
    insight = ""

    if chart_type == "Correlation Heatmap":
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            _style_matplotlib_ax(ax, "Correlation Heatmap Unavailable")
            ax.text(0.5, 0.5, "At least two numeric columns are required.", ha="center", va="center", color=COLOR_MUTED)
            insight = "Insufficient numeric features to compute a correlation matrix."
        else:
            corr = num_df.corr()
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap=sns.diverging_palette(10, 175, as_cmap=True),
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                linecolor=COLOR_BORDER,
                ax=ax,
            )
            ax.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="semibold", color=COLOR_TEXT, pad=10, loc="left")
            ax.tick_params(labelsize=8.5, colors=COLOR_TEXT)
            plt.xticks(rotation=25, ha="right")
            plt.yticks(rotation=0)
            corr_pairs = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack()
            if not corr_pairs.empty:
                top_pair = corr_pairs.abs().idxmax()
                top_val = corr.loc[top_pair[0], top_pair[1]]
                insight = f"Strongest pairwise linear relationship: {top_pair[0]} and {top_pair[1]} (r = {top_val:.2f})."

    elif is_numeric:
        series = df[feature].dropna()
        if chart_type == "Histogram with KDE":
            sns.histplot(series, kde=True, bins=20, color=COLOR_PRIMARY, edgecolor=COLOR_CARD, alpha=0.82, ax=ax)
            mean_v, med_v = series.mean(), series.median()
            ax.axvline(mean_v, color=COLOR_AMBER, linestyle="--", linewidth=1.4, label=f"Mean: {mean_v:.2f}")
            ax.axvline(med_v, color=COLOR_SLATE, linestyle=":", linewidth=1.4, label=f"Median: {med_v:.2f}")
            ax.legend(frameon=True, facecolor=COLOR_CARD, edgecolor=COLOR_BORDER, fontsize=8.5)
            _style_matplotlib_ax(ax, f"Distribution of {feature}", xlabel=feature, ylabel="Student Count")
            insight = (
                f"{feature} ranges from {series.min():.2f} to {series.max():.2f} "
                f"with Mean = {mean_v:.2f}, Median = {med_v:.2f}, and Std Dev = {series.std():.2f}."
            )

        elif chart_type == "Box Plot":
            if series.empty:
                _style_matplotlib_ax(ax, f"Quartile Spread and Outliers: {feature}", xlabel=feature)
                ax.text(0.5, 0.5, "No non-null numeric values available.", ha="center", va="center", color=COLOR_MUTED)
                insight = f"No non-null values found in {feature}."
            else:
                ax.boxplot(
                    series.values,
                    orientation="horizontal",
                    patch_artist=True,
                    widths=0.45,
                    boxprops={"facecolor": COLOR_PRIMARY, "edgecolor": COLOR_TEXT, "alpha": 0.85, "linewidth": 1.1},
                    medianprops={"color": COLOR_TEXT, "linewidth": 1.8},
                    whiskerprops={"color": COLOR_BORDER, "linewidth": 1.4},
                    capprops={"color": COLOR_BORDER, "linewidth": 1.4},
                    flierprops={"marker": "o", "markerfacecolor": COLOR_CARD, "markeredgecolor": COLOR_MUTED, "markersize": 5, "alpha": 0.8},
                )
                _style_matplotlib_ax(ax, f"Quartile Spread and Outliers: {feature}", xlabel=feature)
                ax.set_yticks([])
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
                insight = (
                    f"IQR spans {q1:.2f} to {q3:.2f} (median {series.median():.2f}); "
                    f"detected {len(outliers)} statistical outliers beyond 1.5x IQR."
                )

        else:  # Scatter vs Target
            y_col = target_col if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]) else None
            if y_col is None:
                num_candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c != feature]
                y_col = num_candidates[0] if num_candidates else feature

            clean_pair = df[[feature, y_col]].dropna()
            sns.regplot(
                data=clean_pair,
                x=feature,
                y=y_col,
                scatter_kws={"color": COLOR_PRIMARY, "alpha": 0.72, "s": 36, "edgecolors": COLOR_CARD},
                line_kws={"color": COLOR_TEXT, "linewidth": 1.6},
                ax=ax,
            )
            _style_matplotlib_ax(ax, f"{feature} vs {y_col}", xlabel=feature, ylabel=y_col)
            r_val = clean_pair[feature].corr(clean_pair[y_col]) if len(clean_pair) > 1 and feature != y_col else 1.0
            insight = f"Linear correlation between {feature} and {y_col}: r = {r_val:.3f} across {len(clean_pair)} valid records."

    else:
        clean_cat = df[feature].dropna().astype(str)
        top_cats = clean_cat.value_counts().head(12)
        if chart_type == "Category Mean vs Target" and target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            plot_df = df[[feature, target_col]].dropna().copy()
            plot_df[feature] = plot_df[feature].astype(str)
            plot_df = plot_df[plot_df[feature].isin(top_cats.index)]
            if plot_df.empty:
                _style_matplotlib_ax(ax, f"Mean {target_col} by {feature}", xlabel=feature, ylabel=f"Mean {target_col}")
                ax.text(0.5, 0.5, "No complete pairs available for category averages.", ha="center", va="center", color=COLOR_MUTED)
                insight = f"No valid pairs available between {feature} and {target_col}."
            else:
                sns.barplot(
                    data=plot_df,
                    x=feature,
                    y=target_col,
                    estimator="mean",
                    errorbar=None,
                    color=COLOR_PRIMARY,
                    ax=ax,
                )
                _style_matplotlib_ax(ax, f"Mean {target_col} by {feature}", xlabel=feature, ylabel=f"Mean {target_col}")
                plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
                means = plot_df.groupby(feature)[target_col].mean()
                insight = f"Highest average {target_col} is observed in '{means.idxmax()}' ({means.max():.2f})."
        else:
            if top_cats.empty:
                _style_matplotlib_ax(ax, f"Category Frequency: {feature}", xlabel=feature, ylabel="Count")
                ax.text(0.5, 0.5, "No non-null categorical values available.", ha="center", va="center", color=COLOR_MUTED)
                insight = f"No non-null values found in {feature}."
            else:
                count_df = top_cats.reset_index()
                count_df.columns = [feature, "Count"]
                sns.barplot(data=count_df, x=feature, y="Count", color=COLOR_PRIMARY, ax=ax)
                _style_matplotlib_ax(ax, f"Category Frequency: {feature}", xlabel=feature, ylabel="Count")
                plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
                insight = (
                    f"'{top_cats.index[0]}' is the most frequent value ({top_cats.iloc[0]} records) "
                    f"out of {clean_cat.nunique()} unique categories in {feature}."
                )

    fig.tight_layout()
    return fig, insight


# ---------------------------------------------------------
# Machine Learning Helpers (ONLY Linear & Logistic Regression)
# ---------------------------------------------------------
ML_APPROACHES = [
    {
        "Approach": "Supervised Learning",
        "Definition": "Models learn from labeled input-output pairs (features X mapped to known target y).",
        "Application in System": "Linear Regression predicts continuous scores; Logistic Regression classifies performance tiers.",
    },
    {
        "Approach": "Unsupervised Learning",
        "Definition": "Algorithms identify hidden groupings or patterns in unlabeled data without predefined target values.",
        "Application in System": "Useful conceptually for clustering students with similar study habits or attendance profiles.",
    },
    {
        "Approach": "Reinforcement Learning",
        "Definition": "An agent learns sequential actions in an environment by maximizing cumulative reward signals.",
        "Application in System": "Applicable to adaptive study recommendation scheduling over successive academic terms.",
    },
]


def prepare_target_series(y: pd.Series, algorithm: str) -> tuple[pd.Series, dict]:
    """
    Automatically encodes or discretizes the target column to match
    Linear Regression (continuous) or Logistic Regression (categorical).
    """
    if algorithm == "Linear Regression":
        if pd.api.types.is_numeric_dtype(y):
            return y.astype(float), {"encoded": False, "mapping": None, "note": "Numeric continuous target used directly."}
        encoder = LabelEncoder()
        y_enc = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
        mapping = {cls: int(idx) for idx, cls in enumerate(encoder.classes_)}
        return y_enc.astype(float), {
            "encoded": True,
            "mapping": mapping,
            "encoder": encoder,
            "note": f"Categorical target automatically encoded to numeric indices: {mapping}.",
        }

    # Logistic Regression
    unique_count = y.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(y) or unique_count <= 6:
        y_cls = y.astype(str)
        return y_cls, {"encoded": False, "binned": False, "note": "Discrete categorical target used directly."}

    # Continuous numeric target -> automatically bin into 3 balanced performance classes
    labels = ["Needs Support", "Average", "High Performing"]
    try:
        y_binned, bins = pd.qcut(y.astype(float), q=3, labels=labels, retbins=True, duplicates="drop")
        if y_binned.nunique() < 2:
            raise ValueError("Insufficient distinct quantiles")
        ranges = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
        return y_binned.astype(str), {
            "encoded": True,
            "binned": True,
            "bin_ranges": dict(zip(labels[: len(ranges)], ranges)),
            "note": f"Continuous target automatically discretized into classes: {dict(zip(labels[:len(ranges)], ranges))}.",
        }
    except Exception:
        median_val = float(y.median())
        y_binned = pd.Series(np.where(y >= median_val, "High", "Low"), index=y.index, name=y.name)
        return y_binned, {
            "encoded": True,
            "binned": True,
            "bin_ranges": {"Low": f"< {median_val:.1f}", "High": f">= {median_val:.1f}"},
            "note": f"Continuous target split at median ({median_val:.1f}) into Low / High classes.",
        }


def get_metric_purpose_table(algorithm: str) -> pd.DataFrame:
    """Returns the Metric & Purpose table for Linear or Logistic Regression."""
    if algorithm == "Linear Regression":
        return pd.DataFrame(
            [
                {"Metric": "R2 Score", "Purpose": "Measures proportion of variance in Final performance explained by the features (closer to 1.0 is better)."},
                {"Metric": "MAE", "Purpose": "Mean Absolute Error; average absolute difference between predicted and actual student scores."},
            ]
        )
    return pd.DataFrame(
        [
            {"Metric": "Accuracy", "Purpose": "Overall percentage of student performance classes predicted correctly."},
            {"Metric": "Precision", "Purpose": "Proportion of predicted positive class assignments that are actually correct."},
            {"Metric": "Recall", "Purpose": "Proportion of actual class members correctly identified by the classifier."},
            {"Metric": "F1-score", "Purpose": "Harmonic mean of Precision and Recall, balancing false positives and false negatives."},
        ]
    )


def train_evaluate_ml_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    algorithm: str,
    test_size: float = 0.2,
    use_scaler: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Trains and evaluates ONLY Linear Regression or Logistic Regression on prepared data.
    """
    if algorithm not in {"Linear Regression", "Logistic Regression"}:
        raise ValueError("Supported algorithms are strictly 'Linear Regression' and 'Logistic Regression'.")

    work_df = df.dropna(subset=feature_cols + [target_col]).copy()
    if len(work_df) < 10:
        raise ValueError("At least 10 complete rows are required for model training and evaluation.")

    X = work_df[feature_cols].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    y_raw = work_df[target_col]
    y_prepared, target_meta = prepare_target_series(y_raw, algorithm)

    task_type = "Regression" if algorithm == "Linear Regression" else "Classification"
    stratify_y = None
    if task_type == "Classification":
        min_count = y_prepared.value_counts().min()
        if min_count >= 2 and int(len(y_prepared) * test_size) >= y_prepared.nunique():
            stratify_y = y_prepared

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_prepared,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    scaler = StandardScaler() if use_scaler else None
    if scaler is not None:
        X_train_proc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_proc = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    else:
        X_train_proc = X_train
        X_test_proc = X_test

    if algorithm == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        metrics = {
            "R2 Score": round(float(r2_score(y_test, y_pred)), 4),
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }
        classes = None
    else:
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        metrics = {
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "F1-score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
        classes = list(model.classes_)

    return {
        "model": model,
        "scaler": scaler,
        "algorithm": algorithm,
        "task_type": task_type,
        "features": feature_cols,
        "target_col": target_col,
        "target_meta": target_meta,
        "metrics": metrics,
        "y_test": y_test.values,
        "y_pred": y_pred,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "classes": classes,
    }


def save_model_to_disk(model_bundle: dict, filepath: str = DEFAULT_MODEL_PATH) -> str:
    """Persists trained model bundle to disk using Joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_bundle, filepath)
    return filepath


def load_model_from_disk(filepath: str = DEFAULT_MODEL_PATH) -> dict | None:
    """Loads a previously saved Joblib model bundle from disk if available."""
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None


def plot_model_evaluation(result: dict) -> plt.Figure:
    """Creates a Matplotlib/Seaborn evaluation plot for Linear or Logistic Regression."""
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": COLOR_BORDER, "figure.facecolor": COLOR_CARD})
    fig, ax = plt.subplots(figsize=(6.8, 4.0), facecolor=COLOR_CARD)

    y_true = result["y_test"]
    y_pred = result["y_pred"]

    if result["task_type"] == "Regression":
        ax.scatter(y_true, y_pred, color=COLOR_PRIMARY, alpha=0.75, s=36, edgecolors=COLOR_CARD)
        min_v = float(min(np.min(y_true), np.min(y_pred)))
        max_v = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([min_v, max_v], [min_v, max_v], color=COLOR_TEXT, linestyle="--", linewidth=1.4, label="Ideal Fit (y = x)")
        ax.legend(frameon=True, facecolor=COLOR_CARD, edgecolor=COLOR_BORDER, fontsize=8.5)
        _style_matplotlib_ax(ax, "Actual vs. Predicted Values", xlabel="Actual Value", ylabel="Predicted Value")
    else:
        classes = result.get("classes") or sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(COLOR_PRIMARY, as_cmap=True),
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            linecolor=COLOR_BORDER,
            ax=ax,
        )
        ax.set_title("Confusion Matrix (Test Set)", fontsize=11, fontweight="semibold", color=COLOR_TEXT, pad=10, loc="left")
        ax.set_xlabel("Predicted Class", fontsize=9.5, color=COLOR_TEXT)
        ax.set_ylabel("Actual Class", fontsize=9.5, color=COLOR_TEXT)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Prediction & Academic Guidance Plan Helpers
# ---------------------------------------------------------
def predict_student_performance(model_bundle: dict, input_values: dict) -> dict:
    """Runs inference using a trained Linear or Logistic Regression bundle."""
    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    features = model_bundle["features"]
    task_type = model_bundle.get("task_type", "Regression")

    row_df = pd.DataFrame([{f: input_values.get(f, 0.0) for f in features}])[features]
    if scaler is not None:
        proc_df = pd.DataFrame(scaler.transform(row_df), columns=features)
    else:
        proc_df = row_df

    raw_pred = model.predict(proc_df)[0]

    if task_type == "Regression":
        pred_score = float(np.clip(round(float(raw_pred), 2), 0.0, 100.0))
        if pred_score >= 85.0:
            tier, variant = "High Performing", "teal"
        elif pred_score >= 70.0:
            tier, variant = "Good Standing", "teal"
        elif pred_score >= 55.0:
            tier, variant = "Average", "amber"
        else:
            tier, variant = "At-Risk", "rose"
        return {
            "task_type": "Regression",
            "predicted_value": pred_score,
            "display_label": f"{pred_score:.2f}",
            "tier": tier,
            "variant": variant,
        }

    pred_class = str(raw_pred)
    lower_cls = pred_class.lower()
    if any(k in lower_cls for k in ["high", "out", "good", "pass", "excel"]):
        variant = "teal"
    elif any(k in lower_cls for k in ["avg", "average", "mod"]):
        variant = "amber"
    else:
        variant = "rose"
    return {
        "task_type": "Classification",
        "predicted_value": pred_class,
        "display_label": pred_class,
        "tier": pred_class,
        "variant": variant,
    }


def generate_academic_guidance(
    study_hrs: float,
    attendance: float,
    previous_marks: float,
    assignment_scores: float,
    predicted_label: str,
) -> list[dict]:
    """Generates structured academic guidance directives based on weak metrics."""
    directives = []

    if attendance < 75.0:
        directives.append(
            {
                "area": "Attendance Recovery",
                "status": f"Critical ({attendance:.1f}% < 75.0% threshold)",
                "variant": "rose",
                "recommendation": "Attendance is below the 75% minimum requirement. Prioritize attending all upcoming lectures and practical sessions without absence.",
                "action_steps": [
                    "Attend 100% of scheduled classes over the next 3 weeks to rebuild margin.",
                    "Submit approved medical or academic leave documentation to your department coordinator.",
                ],
            }
        )
    elif attendance < 82.0:
        directives.append(
            {
                "area": "Attendance Buffer",
                "status": f"Caution ({attendance:.1f}%)",
                "variant": "amber",
                "recommendation": "Attendance meets the minimum 75% threshold but leaves a narrow safety buffer.",
                "action_steps": [
                    "Maintain attendance above 85% to protect against unforeseen absences.",
                ],
            }
        )

    if study_hrs < 3.0:
        directives.append(
            {
                "area": "Daily Study Hours",
                "status": f"Low ({study_hrs:.1f} hrs/day)",
                "variant": "rose",
                "recommendation": "Daily self-study time is insufficient for consistent concept retention.",
                "action_steps": [
                    "Increase focused self-study to at least 4.0 hours/day using fixed 50-minute study blocks.",
                    "Review lecture notes on the same day each class is held.",
                ],
            }
        )
    elif study_hrs < 4.5:
        directives.append(
            {
                "area": "Study Consistency",
                "status": f"Moderate ({study_hrs:.1f} hrs/day)",
                "variant": "amber",
                "recommendation": "Adding 1 additional hour of structured daily practice can lift exam readiness.",
                "action_steps": [
                    "Dedicate 1 extra evening hour to practice problem sets and previous exam papers.",
                ],
            }
        )

    if previous_marks < 60.0:
        directives.append(
            {
                "area": "Foundational Marks",
                "status": f"Needs Reinforcement ({previous_marks:.1f})",
                "variant": "rose",
                "recommendation": "Previous marks indicate foundational topic gaps that may carry forward.",
                "action_steps": [
                    "Identify the lowest-scoring topics from past tests and schedule faculty office hours.",
                    "Complete weekly timed diagnostic quizzes to verify conceptual mastery.",
                ],
            }
        )

    if assignment_scores < 65.0:
        directives.append(
            {
                "area": "Assignment Scores",
                "status": f"Below Target ({assignment_scores:.1f})",
                "variant": "amber",
                "recommendation": "Low assignment scores reduce internal assessment cushion before final exams.",
                "action_steps": [
                    "Start assignments at least 5 days before deadlines and review grading rubrics carefully.",
                    "Verify solutions with peers or lab instructors prior to final submission.",
                ],
            }
        )

    if not directives:
        directives.append(
            {
                "area": "Academic Momentum",
                "status": f"On Track (Predicted: {predicted_label})",
                "variant": "teal",
                "recommendation": "All monitored indicators are above benchmark levels. Maintain current study cadence.",
                "action_steps": [
                    "Continue balanced daily study hours and timely assignment submissions.",
                    "Explore advanced projects, peer tutoring, or competitive problem solving.",
                ],
            }
        )

    return directives


# ---------------------------------------------------------
# Groq AI & Local Fallback Chatbot Helpers
# ---------------------------------------------------------
def build_cohort_context(df: pd.DataFrame | None, model_info: dict | None = None) -> str:
    """Builds concise dataset and model context for the AI advisor."""
    lines = []
    if df is not None and not df.empty:
        eda = compute_eda_questions(df)
        lines.append(f"Active dataset rows: {eda['student_count']}")
        if eda["avg_marks_map"]:
            marks_str = ", ".join([f"{k}: {v}" for k, v in eda["avg_marks_map"].items()])
            lines.append(f"Average marks: {marks_str}")
        if eda["attendance_col"]:
            lines.append(
                f"Students with >=75% attendance: {eda['high_attendance_count']} ({eda['high_attendance_pct']}%)"
            )
        lines.append(f"Study-performance correlation: {eda['relationship_summary']}")
    else:
        lines.append("No dataset is currently loaded in session state.")

    if model_info and model_info.get("model_type"):
        lines.append(
            f"Active trained model: {model_info['model_type']} ({model_info.get('model_task', '')}), "
            f"metrics: {model_info.get('model_metrics', {})}"
        )
    else:
        lines.append("No ML model has been trained in the current session.")

    return "\n".join(lines)


_ML_CONCEPT_PATTERN = re.compile(
    r"\b(?:"
    r"ml|machine\s+learning|linear\s+regression|logistic\s+regression|"
    r"r2|mae|f1(?:[-\s]?score)?|accuracy|precision|(?<!active\s)recall|"
    r"supervised|unsupervised|reinforcement|train(?:ing)?|train/test|split"
    r")\b"
)
_STUDY_PLAN_PATTERN = re.compile(
    r"\b(?:"
    r"study|plan(?:ning|s)?|schedule(?:s|d)?|strateg(?:y|ies)|"
    r"revision|learning|exam(?:s|ination|inations)?"
    r")\b"
)
_ATTENDANCE_IMPROVEMENT_PATTERN = re.compile(
    r"(?<!\w)75%|"
    r"\b(?:"
    r"attendance|absent|recover(?:y|ing)?|average\s+marks|"
    r"previous\s+marks|assignment\s+scores?|improve(?:ment|s|d)?|cohort"
    r")\b"
)


def offline_academic_advisor(
    prompt: str,
    df: pd.DataFrame | None = None,
    model_info: dict | None = None,
) -> str:
    """
    Local dataset-aware and rule-based fallback advisor covering:
    1. Study planning and learning strategies
    2. Attendance and academic improvement
    3. Understanding ML concepts
    4. General educational guidance
    """
    p = prompt.lower()
    eda = compute_eda_questions(df) if df is not None and not df.empty else None

    # Area 3: Understanding ML concepts (checked first to avoid 'learning'/'improve'/'score'/'dataset' collisions)
    if _ML_CONCEPT_PATTERN.search(p):
        model_note = ""
        if model_info and model_info.get("model_type"):
            model_note = f"\n\n**Current Session Model:** `{model_info['model_type']}` with metrics `{model_info.get('model_metrics', {})}`."
        return (
            "### Machine Learning Concepts in This System\n\n"
            "- **Supervised Learning:** Learns from labeled examples (`X` features mapped to known `y` outcomes).\n"
            "- **Linear Regression:** Predicts continuous numeric targets such as `Final performance`. Evaluated using **R2 Score** (variance explained) and **MAE** (average absolute error).\n"
            "- **Logistic Regression:** Predicts discrete performance categories. Evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-score**.\n"
            "- **Train/Test Split:** Separates historical records into training data (to fit weights) and unseen test data (to measure generalization)."
            f"{model_note}"
        )

    # Area 1: Study planning and learning strategies
    if _STUDY_PLAN_PATTERN.search(p):
        corr_note = ""
        if eda and eda["correlation_r"] is not None:
            corr_note = (
                f"\n\n**Dataset Context:** In the active dataset ({eda['student_count']} students), "
                f"the correlation between study hours and final performance is **r = {eda['correlation_r']:.3f}**."
            )
        return (
            "### Study Planning and Learning Strategies\n\n"
            "A structured weekly study plan balances concept review, active recall, and assignment work:\n\n"
            "1. **Daily Time Blocks:** Aim for 4 to 5 focused study hours per day split into 50-minute sessions with 10-minute breaks.\n"
            "2. **Active Recall & Spaced Repetition:** Test yourself on lecture material within 24 hours, again at day 7, and before assessments.\n"
            "3. **Assignment-First Reinforcement:** Complete practice problems and assignments early to expose conceptual weak spots before exams.\n"
            "4. **Weekly Audit:** Reserve 2 hours each weekend to review missed questions and update weak-topic checklists."
            f"{corr_note}"
        )

    # Area 2: Attendance and academic improvement
    if _ATTENDANCE_IMPROVEMENT_PATTERN.search(p):
        stats_block = ""
        if eda:
            marks_str = ", ".join([f"{k}: {v}" for k, v in eda["avg_marks_map"].items()]) or "N/A"
            stats_block = (
                f"\n\n**Active Dataset Metrics ({eda['student_count']} students):**\n"
                f"- Average marks: {marks_str}\n"
                f"- High attendance (>= 75%): {eda['high_attendance_count']} students ({eda['high_attendance_pct']}%)\n"
                f"- Study vs. performance insight: {eda['relationship_summary']}"
            )
        return (
            "### Attendance and Academic Improvement\n\n"
            "Maintaining attendance at or above **75%** keeps students eligible for examinations and directly supports internal marks:\n\n"
            "1. **Protect Practical & Lab Sessions:** Labs and tutorials carry high continuous-evaluation weight and are difficult to make up.\n"
            "2. **Recovery Rule:** After any missed lecture, review slides and solve at least 3 related exercises within 48 hours.\n"
            "3. **Early Intervention:** Pair attendance tracking with assignment completion so drops are addressed before final evaluations."
            f"{stats_block}"
        )

    # Area 4: General educational guidance & fallback
    if eda:
        att_part = (
            f", with **{eda['high_attendance_count']}** students ({eda['high_attendance_pct']}%) at >= 75% attendance"
            if eda["attendance_col"]
            else ""
        )
        corr_part = (
            f" Study hours show a **{eda['relationship_strength'].lower()}** relationship with performance (r = {eda['correlation_r']:.3f})."
            if eda["correlation_r"] is not None
            else ""
        )
        dataset_line = (
            f"Active dataset has **{eda['student_count']} students** with primary average score "
            f"**{eda['primary_avg_marks']}**{att_part}.{corr_part}"
        )
    else:
        dataset_line = "Upload or generate a dataset in **Upload Dataset** to enable live cohort calculations."
    return (
        "### Educational Guidance Summary\n\n"
        f"{dataset_line}\n\n"
        "Consistent habits across core dataset indicators (`Study hrs`, `Attendance`, `Previous marks`, "
        "and `Assignment scores`) support steady academic outcomes:\n"
        "1. **Study planning and learning strategies** (daily schedules, revision techniques)\n"
        "2. **Attendance and academic improvement** (reaching the 75% threshold, boosting assignment scores)\n"
        "3. **Understanding ML concepts** (Linear vs. Logistic Regression, R2, MAE, F1-score)\n"
        "4. **General educational guidance** (goal setting, exam preparation, and time management)"
    )


def generate_chat_response(
    prompt: str,
    df: pd.DataFrame | None = None,
    model_info: dict | None = None,
    api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
) -> tuple[str, str]:
    """
    Generates an advisor response using the Groq AI API when an API key is provided,
    or falls back to the local dataset/rule-based advisor.
    Returns (response_text, mode_label).
    """
    clean_key = (api_key or "").strip()
    if not clean_key:
        return offline_academic_advisor(prompt, df, model_info), "Offline Fallback"

    try:
        from groq import Groq

        client = Groq(api_key=clean_key)
        context_str = build_cohort_context(df, model_info)
        system_msg = (
            "You are a concise, practical academic guidance assistant for the AI Student Performance "
            "& Academic Guidance System. Support questions across: (1) Study planning and learning strategies, "
            "(2) Attendance and academic improvement, (3) Understanding ML concepts (Linear Regression, "
            "Logistic Regression, R2, MAE, Accuracy, Precision, Recall, F1-score), and (4) General educational guidance. "
            "Do not use emojis or marketing buzzwords. Base numeric claims on the provided context.\n\n"
            f"Session Context:\n{context_str}"
        )
        completion = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        text = completion.choices[0].message.content if completion.choices else ""
        if text and text.strip():
            return text.strip(), f"Groq API ({groq_model})"
        return offline_academic_advisor(prompt, df, model_info), "Offline Fallback (Empty API Response)"
    except Exception as exc:
        fallback = offline_academic_advisor(prompt, df, model_info)
        return (
            f"Groq API request failed ({str(exc)}). Returning local guidance:\n\n{fallback}",
            "Offline Fallback (API Error)",
        )


def format_chat_history_export(chat_history: list[dict]) -> str:
    """Formats session chat history as clean Markdown text for download."""
    lines = [
        "# AI Academic Advisor Chat Transcript",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for msg in chat_history:
        role = msg.get("role", "assistant").upper()
        content = msg.get("content", "")
        lines.append(f"## {role}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)
