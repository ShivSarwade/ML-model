# Classroom Attendance Prediction

## Overview
This project implements an end-to-end Machine Learning pipeline to predict classroom attendance. It leverages historical attendance data, academic schedule features, and temporal momentum indicators to forecast attendance both exactly (Regression) and categorically (Classification).

## Getting Started
If you are starting from a completely raw state with only `data/raw/attendance_raw.csv`:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Complete End-to-End Pipeline:**
   The master orchestrator securely cleans the data, engineers all complex features (preventing leakage), and trains all 12 predictive models (XGBoost, Random Forest, SVM, etc.).
   ```bash
   cd src
   python run_pipeline.py
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

## Directory Structure
- `data/`: Contains raw and processed CSVs.
- `src/`: Modular Python scripts for ML.
- `notebooks/`: Jupyter notebook deliverables detailing step-by-step logic.
- `models/`: Saved serialized pipeline artifacts (`.pkl`).
- `dashboard/`: Interactive MLOps UI.
- `documentation/`: Data dictionaries and methodologies.

## Deliverables
- Check `Final_Report.md` in the root directory for a comprehensive analysis of the project's success.
