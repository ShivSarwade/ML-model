---
name: model-evaluation
description: >-
  Calculates all required mathematical metrics for model performance distinguishing between train, val, and test sets.
---

# MODEL EVALUATION

## PURPOSE
To quantitatively measure and report model accuracy and error exactly as specified by the PDF requirements.

## WHEN TO USE
After model training, tuning, or cross-validation.

## INPUTS
- Fitted models
- train_dataset, validation_dataset (or test_dataset when authorized)

## OUTPUTS
- Metric reports (TRAIN vs VALIDATION vs TEST).

## RESPONSIBILITIES
- Calculate Regression metrics: MAE, RMSE, MAPE, R2.
- Calculate Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Generate diagnostic outputs (Confusion Matrix, Residual Plots) internally.

## RULES
- NEVER report test performance as validation performance.
- Distinctly label which dataset partition (Train/Val/Test) the metrics belong to.

## VALIDATION
- Ensure metrics logically align (e.g., high R2 should correspond to low RMSE).

## FAILURE CONDITIONS
- NaN or Infinity in metric calculations (usually due to division by zero in MAPE).

## EDGE CASES
- Classification with zero true positives in a class: Precision/Recall will throw warnings; handle gracefully with `zero_division` parameters.

## EXAMPLES
- `mape = mean_absolute_percentage_error(y_true, y_pred)`

## DOWNSTREAM DEPENDENCIES
- error-analysis, experiment-tracking
