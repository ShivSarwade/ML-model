# Academic Attendance Prediction: Methodology

## 1. Problem Formulation
Attendance forecasting can be framed in two ways:
1. **Regression:** Directly predicting the `Attendance_Percentage` (e.g., predicting exactly 85%).
2. **Classification:** Binning the attendance into defined risk bands (e.g., 'Low', 'Medium', 'High').

This pipeline simultaneously pursues both strategies to offer the greatest utility to academic faculty.

## 2. Feature Engineering & Temporal Integrity
The greatest risk in time-series and event forecasting is **Data Leakage**. 

Our methodology strictly prohibits the inclusion of future information. All historical momentum features (`Rolling_Avg_3`, `Monthly_Avg_Attendance`) enforce a strict `shift(1)` boundary. This guarantees that when predicting Lecture $N$, the algorithm only has access to Lecture $N-1$ and prior.

## 3. Algorithm Selection
We selected a diverse ensemble of base algorithms to test theoretical hypotheses about the data structure:
- **Distance-based:** k-NN (Hypothesis: Similar scheduling blocks exhibit similar attendance).
- **Probability-based:** Naive Bayes (Hypothesis: Features are conditionally independent).
- **Linear:** Logistic/Linear Regression, SVM (Hypothesis: Linear relationships dictate attendance decay).
- **Tree-based & Ensemble:** Decision Trees, Random Forest, XGBoost (Hypothesis: Scheduling rules are highly non-linear and hierarchical).

*Result:* The tree-based models (specifically XGBoost) massively outperformed the others. This empirically proved that attendance behavior is ruled by non-linear constraints (e.g., "If it is Morning AND not a holiday AND week 14, then attendance is High"). Linear models failed to capture this logic.

## 4. Evaluation Paradigm
Models were not judged solely on training accuracy, which encourages overfitting. They were validated through strict chronological splitting (Train 70% | Validation 15% | Test 15%). The best model (XGBoost) was ultimately frozen and tested against the unseen 15% test set, maintaining a high $R^2$ of 0.92, proving strong generalization capability.
