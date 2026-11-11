# AI Student Performance & Academic Guidance System

An interactive data analytics, machine learning, and academic guidance platform built with **Streamlit**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Scikit-learn**, **Joblib**, and **Groq AI**, styled in a **Clean Daylight Theme** (`#FAFAFA` canvas, `#F4F4F5` sidebar, `#FFFFFF` bordered cards, `#18181B` primary ink, `#71717A` muted text, and `#0F766E` deep teal accent).

---

## Quick Start Guide

### Option 1: One-Click Windows Launcher
Double-click `run_app.bat` in the project root directory to launch the Streamlit server using the configured Python environment (`C:\Users\jagde\anaconda3\python.exe`).

### Option 2: Command Line (PowerShell / Terminal)
```powershell
# Using Anaconda Python
& "C:\Users\jagde\anaconda3\python.exe" -m streamlit run app.py

# Or via standard environment
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Project Architecture & Directory Layout

```text
Damini's Project/
|-- .streamlit/
|   `-- config.toml                  # Clean Daylight Streamlit theme configuration
|-- pages/
|   |-- 1_Home.py                    # Main Dashboard & organized module navigation
|   |-- 2_Upload_Dataset.py          # CSV ingestion & 6-column sample dataset generator
|   |-- 3_Data_Cleaning.py           # Missing-value handling, deduplication & CSV export
|   |-- 4_EDA.py                     # Statistical summary, glossary, distributions & 4 cohort questions
|   |-- 5_Data_Visualization.py      # Matplotlib & Seaborn charts with computed visual insights
|   |-- 6_Model_Training.py          # Linear & Logistic Regression training, evaluation & Joblib save
|   |-- 7_Prediction.py              # Student performance inference & structured guidance plan
|   |-- 8_AI_Chatbot.py              # Groq AI academic advisor with local offline fallback
|   `-- 9_About_Project.py           # Project purpose, objectives, features & architecture docs
|-- app.py                           # Welcome section & entry point
|-- utils.py                         # Shared state, CSS, data cleaning, EDA, plotting, ML & Groq helpers
|-- requirements.txt                 # Python package dependencies
|-- run_app.bat                      # Windows batch launcher
`-- README.md                        # System documentation
```

---

## Clean Daylight Theme Specification

| Element | Hex Code / Setting | Usage |
| :--- | :--- | :--- |
| **Main Background** | `#FAFAFA` | Primary application canvas |
| **Sidebar Background** | `#F4F4F5` | Persistent navigation & status sidebar |
| **Card Surface** | `#FFFFFF` | Flat containers with `1px solid #E4E4E7` border |
| **Border & Hairlines** | `#E4E4E7` | Dividers, frames, and table borders |
| **Primary Text** | `#18181B` | Headings, labels, and body copy |
| **Muted Text** | `#71717A` | Captions, metadata, and secondary labels |
| **Primary Accent** | `#0F766E` | Active nav state, primary buttons, and charts (Hover `#115E59`) |
| **Status Badges** | Teal / Amber / Rose / Slate | Flat status badges for dataset, model, and guidance tiers |

---

## Core 6-Column Data Schema

The platform standardizes student records around six core columns (while also resolving common column aliases such as `btech_cse_student_data.csv` during exploratory analysis):

1. `Student ID`: Unique student identifier (for example, `STU001` - `STU200`)
2. `Study hrs`: Daily self-study hours
3. `Attendance`: Student attendance percentage (institutional reference gate: `>= 75%`)
4. `Previous marks`: Prior assessment or exam marks (0 - 100 scale)
5. `Assignment scores`: Coursework and assignment evaluation scores (0 - 100 scale)
6. `Final performance`: Target academic performance score (0 - 100 scale)

---

## Module Breakdown & Capabilities

### Welcome (`app.py`) & Main Dashboard (`1_Home.py`)
- Project introduction, AI Intelligence overview, Platform Highlights, 4-step workflow guide, and technology stack summary.
- Live workspace status KPIs (Students Included, Average Marks, Attendance >= 75%, Active ML Model) and direct navigation across all modules.

### 2. Upload Dataset (`2_Upload_Dataset.py`)
- Drag-and-drop CSV file uploader storing `raw_df` and `staging_df` in Streamlit session state.
- One-click sample dataset generator producing the standard 6-column student dataset with configurable sample size (50 - 1,000 rows) and sample imperfections for cleaning practice.
- Displays row count, column count, missing cell count, duplicate row count, column data types, and a tabular preview.

### 3. Data Cleaning (`3_Data_Cleaning.py`)
- Per-column missing-value count, percentage, and data type table.
- Interactive null handling: Drop Missing Rows, Numerical Imputation (Mean or Median), and Categorical Imputation (Mode).
- Duplicate detection and removal by exact row match or `Student ID`.
- Updated dataset shape and data types inspection, cleaning audit log, commit to `cleaned_df`, and CSV download.

### 4. Exploratory Data Analysis (`4_EDA.py`)
- Direct answers to four cohort questions: total students, average marks, high attendance (`>= 75%`) count/percentage, and study-hours correlation (`r`).
- Full numerical summary statistics table paired with the 6-term Statistical Term Reference glossary (`Count`, `Mean`, `Standard deviation`, `Minimum`, `Maximum`, `25%, 50%, 75%`).
- Distribution inspection (Histogram with KDE + Box Plot) and Pearson correlation matrix heatmap rendered with Matplotlib and Seaborn.

### 5. Data Visualization (`5_Data_Visualization.py`)
- Interactive feature-group selector (`Numeric` / `Categorical`) and chart-type dropdown:
  - Numeric: `Histogram with KDE`, `Box Plot`, `Scatter vs Target`, `Correlation Heatmap`
  - Categorical: `Count Plot`, `Category Mean vs Target`
- All charts are rendered statically via Matplotlib and Seaborn with computed textual insights and pairwise scatter overviews.

### 6. Model Training (`6_Model_Training.py`)
- Conceptual overview of Supervised, Unsupervised, and Reinforcement Learning.
- Configurable target column (`y`), input features (`X`), test split (`10%` - `40%`), optional `StandardScaler`, and algorithm selection strictly between **Linear Regression** and **Logistic Regression**.
- Automatic target encoding/discretization (label encoding for categorical targets in Linear Regression; quantile binning for continuous targets in Logistic Regression).
- Evaluates **R2 Score** and **MAE** (Linear Regression) or **Accuracy**, **Precision**, **Recall**, and **F1-score** (Logistic Regression), plots Actual vs. Predicted scatter or Confusion Matrix, and persists bundles via **Joblib** (`models/student_performance_model.joblib`).

### 7. Performance Prediction & Guidance (`7_Prediction.py`)
- Loads the active session model or persisted Joblib bundle (or trains a baseline Linear Regression model in one click).
- Accepts `Study hrs`, `Attendance`, `Previous marks`, and `Assignment scores` inputs to predict `Final performance`, performance tier, and `>= 75%` attendance eligibility.
- Generates a structured **Academic Guidance Plan** with targeted action steps and downloadable Markdown export (`academic_guidance_plan.md`).

### 8. AI Academic Chatbot (`8_AI_Chatbot.py`)
- Connects to the **Groq AI API** (`groq` Python client with `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, or `mixtral-8x7b-32768`) when an API key is entered in the sidebar.
- Automatically falls back to an offline, dataset-aware rule-based advisor covering four core topics:
  1. Study planning and learning strategies
  2. Attendance and academic improvement
  3. Understanding ML concepts (Linear/Logistic Regression, R2, MAE, Accuracy, Precision, Recall, F1-score)
  4. General educational guidance
- Includes quick-topic prompts, `Clear Chat`, and real-time `Download Chat` (`.md` transcript export).

### 9. About Project (`9_About_Project.py`)
- Documents project purpose, core objectives, module-by-module inventory, 3-layer working architecture, and live session-state diagnostics.

---

## Requirements & Dependencies

Install dependencies via pip:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
- `streamlit>=1.30.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `scikit-learn>=1.3.0`
- `joblib>=1.3.0`
- `groq>=0.9.0`
