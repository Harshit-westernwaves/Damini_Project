"""
generate_dataset.py - Multi-Cohort Synthetic Dataset Generator
AI Student Performance & Academic Guidance System

Generates four distinct synthetic student performance datasets inside the 'datasets/' directory,
all adhering to the canonical 6-column specification:
['Student ID', 'Study hrs', 'Attendance', 'Previous marks', 'Assignment scores', 'Final performance']

Target Datasets:
1. datasets/1_raw_dirty_student_data.csv (500 rows, 5% missing numerical/categorical values, 15 duplicate rows)
2. datasets/2_cleaned_balanced_cohort.csv (500 clean rows, zero nulls, realistic distributions)
3. datasets/3_high_risk_cohort_data.csv (300 rows, skewed low attendance & study hours for intervention alerts)
4. datasets/4_large_university_batch.csv (1,200 rows, large-scale batch for model scalability & metrics)
"""

import os
import argparse
import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "Student ID",
    "Study hrs",
    "Attendance",
    "Previous marks",
    "Assignment scores",
    "Final performance",
]


def generate_raw_dirty_dataset(
    n_total_rows: int = 500,
    n_duplicates: int = 15,
    missing_pct: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates 500 rows containing 5% missing numerical & categorical values and 15 duplicate rows.
    Designed specifically to demonstrate live cleaning in 3_Data_Cleaning.py.
    """
    rng = np.random.default_rng(seed)
    n_base = n_total_rows - n_duplicates  # 485 base unique rows

    student_ids = [f"STU{str(i).zfill(3)}" for i in range(1, n_base + 1)]
    study_hrs = np.clip(np.round(rng.normal(loc=4.6, scale=1.9, size=n_base), 1), 0.5, 10.0)
    attendance = np.clip(np.round(rng.beta(a=5.0, b=2.0, size=n_base) * 58 + 42, 1), 40.0, 99.5)
    previous_marks = np.clip(np.round(rng.normal(loc=68.5, scale=13.0, size=n_base), 1), 30.0, 98.0)

    assignment_raw = (
        0.42 * previous_marks
        + 0.26 * attendance
        + 2.5 * study_hrs
        + rng.normal(loc=5.0, scale=6.0, size=n_base)
    )
    assignment_scores = np.clip(np.round(assignment_raw, 1), 32.0, 100.0)

    final_raw = (
        0.36 * previous_marks
        + 0.28 * assignment_scores
        + 0.18 * attendance
        + 2.3 * study_hrs
        + rng.normal(loc=0.0, scale=3.5, size=n_base)
    )
    final_performance = np.clip(np.round(final_raw, 1), 28.0, 99.5)

    base_df = pd.DataFrame(
        {
            "Student ID": student_ids,
            "Study hrs": study_hrs,
            "Attendance": attendance,
            "Previous marks": previous_marks,
            "Assignment scores": assignment_scores,
            "Final performance": final_performance,
        }
    )

    # Pick exactly 15 rows to duplicate to reach n_total_rows (500)
    dup_candidate_slice = np.arange(n_base - n_duplicates, n_base)
    dup_df = base_df.iloc[dup_candidate_slice].copy()
    df = pd.concat([base_df, dup_df], ignore_index=True)

    # Rows 0 to (n_base - n_duplicates) are candidates for null injection
    # This guarantees the 15 duplicate pairs remain identical while hitting exactly 150 missing cells (5.0%)
    candidate_rows = np.arange(0, n_base - n_duplicates)

    # 15 categorical missing values in "Student ID"
    cat_null_count = 15
    cat_null_idx = rng.choice(candidate_rows, size=cat_null_count, replace=False)
    df.loc[cat_null_idx, "Student ID"] = np.nan

    # 135 numerical missing values across remaining 5 columns (27 per column)
    num_cols = ["Study hrs", "Attendance", "Previous marks", "Assignment scores", "Final performance"]
    for col in num_cols:
        col_null_idx = rng.choice(candidate_rows, size=27, replace=False)
        df.loc[col_null_idx, col] = np.nan

    return df


def generate_cleaned_balanced_dataset(
    n_rows: int = 500,
    seed: int = 101,
) -> pd.DataFrame:
    """
    Generates 500 clean rows with zero nulls, zero duplicates, and realistic distributions
    for immediate use in 4_EDA.py, 5_Data_Visualization.py, and 6_Model_Training.py.
    """
    rng = np.random.default_rng(seed)

    student_ids = [f"STU{str(i).zfill(3)}" for i in range(1, n_rows + 1)]
    study_hrs = np.clip(np.round(rng.normal(loc=4.9, scale=1.7, size=n_rows), 1), 1.0, 9.8)
    attendance = np.clip(np.round(rng.beta(a=5.5, b=1.9, size=n_rows) * 55 + 45, 1), 48.0, 99.5)
    previous_marks = np.clip(np.round(rng.normal(loc=71.5, scale=11.5, size=n_rows), 1), 38.0, 98.0)

    assignment_raw = (
        0.44 * previous_marks
        + 0.25 * attendance
        + 2.4 * study_hrs
        + rng.normal(loc=5.5, scale=5.0, size=n_rows)
    )
    assignment_scores = np.clip(np.round(assignment_raw, 1), 40.0, 100.0)

    final_raw = (
        0.38 * previous_marks
        + 0.28 * assignment_scores
        + 0.18 * attendance
        + 2.2 * study_hrs
        + rng.normal(loc=0.0, scale=3.0, size=n_rows)
    )
    final_performance = np.clip(np.round(final_raw, 1), 35.0, 99.5)

    return pd.DataFrame(
        {
            "Student ID": student_ids,
            "Study hrs": study_hrs,
            "Attendance": attendance,
            "Previous marks": previous_marks,
            "Assignment scores": assignment_scores,
            "Final performance": final_performance,
        }
    )


def generate_high_risk_dataset(
    n_rows: int = 300,
    seed: int = 202,
) -> pd.DataFrame:
    """
    Generates 300 rows skewed toward low attendance (< 75%) and low study hours (< 3.0 hrs/day)
    specifically designed to test intervention alerts in 7_Prediction.py and 8_AI_Chatbot.py.
    """
    rng = np.random.default_rng(seed)

    student_ids = [f"RISK{str(i).zfill(3)}" for i in range(1, n_rows + 1)]
    # Study hours skewed low: mean ~1.8 hrs/day, bounded [0.5, 3.2]
    study_hrs = np.clip(np.round(rng.normal(loc=1.8, scale=0.6, size=n_rows), 1), 0.5, 3.2)

    # Attendance skewed low: beta centered around 48%, range [25.0, 71.0] (all below 75% gate)
    attendance = np.clip(np.round(rng.beta(a=2.5, b=2.8, size=n_rows) * 44 + 26, 1), 25.0, 71.0)

    # Previous marks lower: mean ~46.0
    previous_marks = np.clip(np.round(rng.normal(loc=46.0, scale=8.5, size=n_rows), 1), 25.0, 66.0)

    assignment_raw = (
        0.40 * previous_marks
        + 0.22 * attendance
        + 2.0 * study_hrs
        + rng.normal(loc=3.0, scale=4.5, size=n_rows)
    )
    assignment_scores = np.clip(np.round(assignment_raw, 1), 22.0, 68.0)

    final_raw = (
        0.35 * previous_marks
        + 0.28 * assignment_scores
        + 0.18 * attendance
        + 2.0 * study_hrs
        + rng.normal(loc=-1.0, scale=3.0, size=n_rows)
    )
    final_performance = np.clip(np.round(final_raw, 1), 20.0, 65.0)

    return pd.DataFrame(
        {
            "Student ID": student_ids,
            "Study hrs": study_hrs,
            "Attendance": attendance,
            "Previous marks": previous_marks,
            "Assignment scores": assignment_scores,
            "Final performance": final_performance,
        }
    )


def generate_large_university_batch(
    n_rows: int = 1200,
    seed: int = 303,
) -> pd.DataFrame:
    """
    Generates 1,200 rows for testing model scalability, cross-validation, and regression/classification metrics.
    """
    rng = np.random.default_rng(seed)

    student_ids = [f"UNI{str(i).zfill(4)}" for i in range(1, n_rows + 1)]

    # Multi-cohort mixture for university-scale diversity
    # 20% high performers, 60% average, 20% struggling
    cohort_types = rng.choice(["high", "mid", "low"], size=n_rows, p=[0.20, 0.60, 0.20])

    study_hrs = np.zeros(n_rows)
    attendance = np.zeros(n_rows)
    previous_marks = np.zeros(n_rows)

    for i, c_type in enumerate(cohort_types):
        if c_type == "high":
            study_hrs[i] = rng.normal(loc=6.8, scale=1.2)
            attendance[i] = rng.beta(a=7.0, b=1.5) * 20 + 80
            previous_marks[i] = rng.normal(loc=84.0, scale=6.5)
        elif c_type == "mid":
            study_hrs[i] = rng.normal(loc=4.6, scale=1.3)
            attendance[i] = rng.beta(a=5.0, b=2.0) * 40 + 58
            previous_marks[i] = rng.normal(loc=71.0, scale=8.0)
        else:  # low
            study_hrs[i] = rng.normal(loc=2.2, scale=0.8)
            attendance[i] = rng.beta(a=3.0, b=2.5) * 35 + 40
            previous_marks[i] = rng.normal(loc=52.0, scale=9.0)

    study_hrs = np.clip(np.round(study_hrs, 1), 0.5, 11.5)
    attendance = np.clip(np.round(attendance, 1), 38.0, 100.0)
    previous_marks = np.clip(np.round(previous_marks, 1), 30.0, 99.0)

    assignment_raw = (
        0.44 * previous_marks
        + 0.25 * attendance
        + 2.4 * study_hrs
        + rng.normal(loc=5.0, scale=5.2, size=n_rows)
    )
    assignment_scores = np.clip(np.round(assignment_raw, 1), 30.0, 100.0)

    final_raw = (
        0.38 * previous_marks
        + 0.28 * assignment_scores
        + 0.18 * attendance
        + 2.2 * study_hrs
        + rng.normal(loc=0.0, scale=3.0, size=n_rows)
    )
    final_performance = np.clip(np.round(final_raw, 1), 28.0, 99.5)

    return pd.DataFrame(
        {
            "Student ID": student_ids,
            "Study hrs": study_hrs,
            "Attendance": attendance,
            "Previous marks": previous_marks,
            "Assignment scores": assignment_scores,
            "Final performance": final_performance,
        }
    )


def generate_all_datasets(output_dir: str = "datasets", seed_base: int = 42) -> dict[str, pd.DataFrame]:
    """
    Generates and saves all 4 required CSV datasets inside output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    specs = [
        (
            "1_raw_dirty_student_data.csv",
            lambda: generate_raw_dirty_dataset(n_total_rows=500, n_duplicates=15, missing_pct=0.05, seed=seed_base),
            "500 rows with 5% missing numerical/categorical values & 15 duplicate rows (Data Cleaning demo)",
        ),
        (
            "2_cleaned_balanced_cohort.csv",
            lambda: generate_cleaned_balanced_dataset(n_rows=500, seed=seed_base + 1),
            "500 clean rows, zero nulls, realistic distributions (EDA, Viz & ML training)",
        ),
        (
            "3_high_risk_cohort_data.csv",
            lambda: generate_high_risk_dataset(n_rows=300, seed=seed_base + 2),
            "300 rows skewed toward low attendance (< 75%) and low study hours (Intervention alerts)",
        ),
        (
            "4_large_university_batch.csv",
            lambda: generate_large_university_batch(n_rows=1200, seed=seed_base + 3),
            "1,200 rows diverse university cohort (Scalability & ML benchmarking)",
        ),
    ]

    print("\n" + "=" * 75)
    print("       AI STUDENT PERFORMANCE PLATFORM - DATASET GENERATION SUITE        ")
    print("=" * 75)

    for filename, generator_fn, desc in specs:
        df = generator_fn()
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        generated[filename] = df

        null_total = int(df.isna().sum().sum())
        dup_total = int(df.duplicated().sum())

        print(f"\n[+] Generated: {filename}")
        print(f"    Target Path      : {os.path.abspath(path)}")
        print(f"    Description      : {desc}")
        print(f"    Rows x Cols      : {len(df)} rows x {len(df.columns)} columns")
        print(f"    Columns          : {list(df.columns)}")
        print(f"    Missing Values   : {null_total} cells ({null_total / (len(df) * len(df.columns)) * 100:.2f}%)")
        print(f"    Duplicate Rows   : {dup_total}")
        print(f"    Study hrs Mean   : {df['Study hrs'].dropna().mean():.2f}")
        print(f"    Attendance Mean  : {df['Attendance'].dropna().mean():.2f}%")
        print(f"    Final Perf Mean  : {df['Final performance'].dropna().mean():.2f}")

    print("\n" + "=" * 75)
    print(f"[OK] All 4 datasets successfully generated and verified in: {os.path.abspath(output_dir)}")
    print("=" * 75 + "\n")

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic student datasets for the AI Student Performance platform.")
    parser.add_argument("--output-dir", type=str, default="datasets", help="Directory to save CSV datasets")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for generation")
    args = parser.parse_args()

    generate_all_datasets(output_dir=args.output_dir, seed_base=args.seed)
