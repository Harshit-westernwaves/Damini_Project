"""
generate_dataset.py - Standalone Synthetic Dataset Generator for B.Tech CSE Cohorts
AI Student Performance & Academic Guidance System
"""

import os
import argparse
import numpy as np
import pandas as pd

def generate_btech_cse_dataset(
    n_rows: int = 600,
    missing_pct: float = 0.04,
    n_duplicates: int = 10,
    n_outliers: int = 8,
    random_seed: int = 42,
    output_path: str = "btech_cse_student_data.csv",
):
    """
    Generates a calibrated B.Tech Computer Science & Engineering student performance dataset
    with realistic correlations, missing values, duplicate records, and statistical outliers
    for live demonstration in 3_Data_Cleaning.py.
    """
    np.random.seed(random_seed)

    print(f"[*] Generating {n_rows} base B.Tech CSE student records (Seed: {random_seed})...")

    # 1. Student Identifiers (CSE2023001 .. CSE2023600)
    student_ids = [f"CSE2023{str(i).zfill(3)}" for i in range(1, n_rows + 1)]

    # 2. Previous SGPA (Mean ~7.35, Normal distribution, bounded 4.2 to 9.9)
    previous_sgpa = np.clip(np.random.normal(loc=7.35, scale=1.1, size=n_rows), 4.2, 9.9)

    # 3. Attendance Percentage (Beta distribution skewed towards 75-90%)
    attendance = np.clip(np.random.beta(a=6, b=2, size=n_rows) * 55 + 45, 50.0, 99.5)

    # 4. Mid-Semester Test (MST) Marks (out of 50, correlated with SGPA + noise)
    mst_base = (previous_sgpa / 10.0) * 44 + np.random.normal(0, 3.2, size=n_rows)
    mst_marks = np.clip(mst_base, 10.0, 50.0)

    # 5. Practical Lab Score (out of 100, correlated with Attendance + SGPA)
    lab_base = (previous_sgpa / 10.0) * 82 + (attendance / 100.0) * 16 + np.random.normal(0, 3.8, size=n_rows)
    lab_score = np.clip(lab_base, 42.0, 100.0)

    # 6. Study Hours Per Day (General Coursework: 1.0 to 9.5 hrs/day, Gamma distribution)
    study_hours = np.clip(np.random.gamma(shape=3.5, scale=1.2, size=n_rows), 1.0, 9.5)

    # 7. Coding Practice Hours Per Day (DSA & Problem Solving: 0.5 to 8.0 hrs/day, Gamma distribution)
    coding_hours = np.clip(np.random.gamma(shape=2.8, scale=1.2, size=n_rows), 0.5, 8.0)

    # 8. Backlogs (Discrete count: 0 to 4, skewed towards 0)
    backlog_probs = [0.72, 0.16, 0.08, 0.03, 0.01]
    backlogs = np.random.choice([0, 1, 2, 3, 4], size=n_rows, p=backlog_probs)

    # 9. Extracurricular Engagement Rating (1 to 10 scale)
    extracurricular = np.random.randint(1, 11, size=n_rows)

    # 10. Realistic Target: Final CGPA
    # Realistic weights: 35% Previous SGPA, 22% MST, 18% Coding, 15% Attendance, 10% Lab, -0.40 per Backlog
    mst_norm = (mst_marks / 50.0) * 10.0
    lab_norm = (lab_score / 100.0) * 10.0
    att_norm = (attendance / 100.0) * 10.0
    coding_norm = np.clip((coding_hours / 4.5) * 10.0, 2.0, 10.0)

    cgpa_raw = (
        0.35 * previous_sgpa +
        0.22 * mst_norm +
        0.18 * coding_norm +
        0.15 * att_norm +
        0.10 * lab_norm -
        0.40 * backlogs +
        np.random.normal(0, 0.22, size=n_rows)
    )
    final_cgpa = np.clip(np.round(cgpa_raw, 2), 4.10, 9.95)

    # 11. Performance Category Tier Classification
    def categorize_cgpa(cgpa):
        if cgpa >= 8.50:
            return "Outstanding"
        elif cgpa >= 7.00:
            return "Good"
        elif cgpa >= 5.50:
            return "Average"
        else:
            return "At-Risk"

    performance_categories = [categorize_cgpa(c) for c in final_cgpa]

    # Assemble base DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Attendance_Percentage": np.round(attendance, 1),
        "MST_Marks": np.round(mst_marks, 1),
        "Lab_Score": np.round(lab_score, 1),
        "Study_Hours_Per_Day": np.round(study_hours, 1),
        "Coding_Practice_Hours": np.round(coding_hours, 1),
        "Previous_SGPA": np.round(previous_sgpa, 2),
        "Backlogs": backlogs,
        "Extracurricular_Score": extracurricular,
        "Final_CGPA": final_cgpa,
        "Performance_Category": performance_categories,
    })

    print(f"[*] Base cohort created: {len(df)} rows across {len(df.columns)} columns.")

    # -------------------------------------------------------------
    # Ingestion Anomaly 1: 4% Random Missing Values (NaN)
    # -------------------------------------------------------------
    # Apply across feature columns (excluding Student_ID, Final_CGPA, Performance_Category)
    feature_cols = [
        "Attendance_Percentage",
        "MST_Marks",
        "Lab_Score",
        "Study_Hours_Per_Day",
        "Coding_Practice_Hours",
        "Previous_SGPA",
        "Extracurricular_Score",
    ]

    total_feature_cells = n_rows * len(feature_cols)
    n_nans_to_inject = int(total_feature_cells * missing_pct)
    print(f"[*] Injecting ~{missing_pct*100:.1f}% missing values ({n_nans_to_inject} NaN cells)...")

    for _ in range(n_nans_to_inject):
        row_idx = np.random.randint(0, n_rows)
        col_name = np.random.choice(feature_cols)
        df.loc[row_idx, col_name] = np.nan

    actual_nans = int(df.isna().sum().sum())
    print(f"[+] Total missing cells injected: {actual_nans} ({actual_nans / (n_rows * len(df.columns)) * 100:.2f}% of all cells)")

    # -------------------------------------------------------------
    # Ingestion Anomaly 2: 10 Duplicate Rows
    # -------------------------------------------------------------
    print(f"[*] Injecting {n_duplicates} duplicate records...")
    dup_row_indices = np.random.choice(n_rows, size=n_duplicates, replace=False)
    duplicates_df = df.iloc[dup_row_indices].copy()
    df = pd.concat([df, duplicates_df], ignore_index=True)
    print(f"[+] Appended {n_duplicates} duplicate rows. New total rows: {len(df)}")

    # -------------------------------------------------------------
    # Ingestion Anomaly 3: 8 Extreme Statistical Outliers (for IQR)
    # -------------------------------------------------------------
    print(f"[*] Injecting {n_outliers} extreme outliers...")
    outlier_indices = np.random.choice(n_rows, size=n_outliers, replace=False)

    # Inject physically impossible / extreme student metrics
    df.loc[outlier_indices[0], "Study_Hours_Per_Day"] = 23.5   # 23.5 hours study/day
    df.loc[outlier_indices[1], "Study_Hours_Per_Day"] = 21.0   # 21.0 hours study/day
    df.loc[outlier_indices[2], "Coding_Practice_Hours"] = 22.0 # 22.0 hours coding/day
    df.loc[outlier_indices[3], "Coding_Practice_Hours"] = 19.5 # 19.5 hours coding/day
    df.loc[outlier_indices[4], "MST_Marks"] = 88.0             # 88/50 (out of bounds)
    df.loc[outlier_indices[5], "MST_Marks"] = 92.5             # 92.5/50 (out of bounds)
    df.loc[outlier_indices[6], "Attendance_Percentage"] = 145.0 # 145% attendance
    df.loc[outlier_indices[7], "Attendance_Percentage"] = 13.0  # 13% extreme lower outlier

    print(f"[+] Injected 8 extreme outliers into Study_Hours_Per_Day, Coding_Practice_Hours, MST_Marks, Attendance_Percentage.")

    # -------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Successfully generated and saved dataset to: {os.path.abspath(output_path)}")

    # Print Quality Audit Summary
    print("\n" + "=" * 65)
    print("           B.TECH CSE DATASET QUALITY AUDIT SUMMARY           ")
    print("=" * 65)
    print(f"Total Rows In File       : {len(df)} (600 base + 10 duplicates = {len(df)})")
    print(f"Total Columns            : {len(df.columns)}")
    print(f"Duplicate Rows Count     : {df.duplicated().sum()}")
    print(f"Total Missing Values     : {df.isna().sum().sum()}")
    print("\nMissing Values Breakdown by Attribute:")
    nan_summary = df.isna().sum()
    print(nan_summary[nan_summary > 0].to_string())
    print("\nPerformance Category Breakdown:")
    print(df["Performance_Category"].value_counts().to_string())
    print(f"\nMean Final CGPA          : {df['Final_CGPA'].mean():.2f}")
    print(f"Median Final CGPA        : {df['Final_CGPA'].median():.2f}")
    print("=" * 65)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic B.Tech CSE dataset with imperfections.")
    parser.add_argument("--rows", type=int, default=600, help="Number of base student records (default: 600)")
    parser.add_argument("--output", type=str, default="btech_cse_student_data.csv", help="Output CSV filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_btech_cse_dataset(
        n_rows=args.rows,
        missing_pct=0.04,
        n_duplicates=10,
        n_outliers=8,
        random_seed=args.seed,
        output_path=args.output,
    )
