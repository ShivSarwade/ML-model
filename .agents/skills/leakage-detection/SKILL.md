---
name: leakage-detection
description: >-
  Aggressively scans the engineered dataset and pipeline for target leakage and future information contamination.
---

# LEAKAGE DETECTION

## PURPOSE
To mathematically and logically guarantee that the model cannot "cheat" by accessing future information or the target variable during historical prediction.

## WHEN TO USE
Immediately after feature-engineering and before any model training or dataset splitting.

## INPUTS
- Engineered CSV dataset.
- Feature dictionary.

## OUTPUTS
- Leakage report (PASS, WARNING, FAIL).

## RESPONSIBILITIES
- Check for high correlation (r > 0.95) between any feature and the target variable.
- Verify rolling windows and lags are shifted correctly.
- Ensure post-event variables are not included in inputs.

## RULES
- Any detected inclusion of future data or the target variable in the input features results in an immediate FAIL.
- A FAIL status must block downstream model training.

## VALIDATION
- Attempt a simple decision tree. If accuracy/R2 is unexpectedly perfect (1.0), flag for severe leakage investigation.

## FAILURE CONDITIONS
- Target leakage detected (e.g., 'Actual Students Present' included as a feature when predicting 'Attendance Category').

## EDGE CASES
- Perfectly predictable deterministic classes (e.g., a specific seminar where attendance is strictly mandatory and always 100%). Flag as WARNING.

## EXAMPLES
- Fails if `df[['Previous_Attendance', 'Current_Attendance']].corr()` is exactly 1.0.

## DOWNSTREAM DEPENDENCIES
- temporal-dataset-splitting
