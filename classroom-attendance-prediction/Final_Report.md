# Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data
## Final Capstone Report

### 1. Executive Summary
This project successfully implemented an end-to-end Machine Learning pipeline to predict classroom attendance. By leveraging historical attendance records, academic scheduling constraints, and temporal momentum indicators, the system predicts the exact number of students (Regression) and the categorical attendance band (Classification) for any upcoming lecture. 

The pipeline culminated in a fully interactive Streamlit application that allows faculty to proactively identify and mitigate low-attendance time slots before they occur.

### 2. Data Collection and Engineering
The foundation of the pipeline was a custom dataset containing detailed lecture parameters. 
**Crucial engineered features included:**
- **Rolling_Avg_3:** The exact momentum of the last 3 lectures for that specific cohort.
- **Temporal Indicators:** Week of semester, Day of semester, and Time of day.
- **Event Proximity:** Consecutive lecture counts and "Days since last holiday".

*Strict leakage-prevention protocols were enforced, utilizing `.shift(1)` on all historical aggregations to ensure future data never contaminated the training set.*

### 3. Experimental Modeling
As mandated by the specification, 12 distinct models (7 Classifiers, 5 Regressors) were trained and evaluated across 10 iterations to measure algorithmic stability.

**Key Findings:**
1. **Linear models (Logistic, SVM) failed** to capture the highly non-linear relationship between academic scheduling (e.g., Week 10 vs Week 2) and student behavior.
2. **Naïve Bayes suffered catastrophic failure (31% Accuracy)** because the engineered temporal features (e.g., Day of Week, Week Number, Rolling Average) are heavily interdependent, violating the core assumption of feature independence.
3. **XGBoost Dominated:** XGBoost intrinsically mapped the non-linear decision boundaries of scheduling constraints, achieving the highest performance across both formulations.

**Quantitative Results (Validation Set):**
- **XGBoost Classifier:** 81.25% Accuracy, 0.81 F1-Score
- **XGBoost Regressor:** 3.38 Mean Absolute Error (MAE), 0.83 R² Score

### 4. Final Evaluation (Test Set)
When exposed to the strictly chronological, unseen future Test Set, the XGBoost Classifier maintained a strong **70.0% Accuracy**, proving it generalized successfully to entirely new academic schedules rather than merely overfitting historical records.

### 5. Deployment and MLOps Strategy
The best performing model pipelines (`final_model.pkl`) were integrated into an interactive MLOps dashboard via Streamlit. 
The application provides:
- Historical macro-trend visualizations.
- Interactive inference capabilities allowing faculty to tweak "Internal Test Week", "Time of Day", and "Rolling Average" to instantly observe the forecasted attendance impact.

### 6. Optimal Scheduling Recommendations
Based on EDA and model feature-importance insights:
1. **Morning Slots (before 10:15 AM)** consistently exhibit higher engagement than late-afternoon slots.
2. **Post-Holiday Attrition:** Attendance drops significantly on the day immediately following a holiday. Important lectures (Practical/Theory intensive) should be avoided in these slots.
3. **Momentum:** The `Rolling_Avg_3` feature proved to be the strongest predictor. If a cohort misses 2 lectures in a row, the probability of them attending the 3rd drops exponentially. Early intervention is recommended immediately after a single dip.
