---
name: final-model-validation
description: >-
  Executes the ultimate quality gate, running the selected model against the untouched final test set.
---

# FINAL MODEL VALIDATION

## PURPOSE
To act as the final, rigorous deployment gate. Unveils the test dataset for the first time to ascertain true generalization performance.

## WHEN TO USE
Strictly ONCE, after the prediction-pipeline is built and model-selection is complete.

## INPUTS
- prediction-pipeline
- Untouched test_dataset

## OUTPUTS
- Final Test Metrics report.
- Gate Status (PASS/DEPLOYMENT BLOCKED).

## RESPONSIBILITIES
- Verify no leakage exists in the pipeline logic.
- Run predictions on the test dataset.
- Record final performance.

## RULES
- If performance drops drastically compared to validation (indicating leakage or severe overfitting), trigger DEPLOYMENT BLOCKED.
- Do NOT retrain or tune the model based on these test results.

## VALIDATION
- Pipeline produces valid predictions for 100% of the test set without errors.

## FAILURE CONDITIONS
- Test metrics fail catastrophic thresholds (e.g., R2 < 0 or Accuracy < Random Chance).

## EDGE CASES
- Model depends on unavailable future information during pipeline execution: BLOCKED.

## EXAMPLES
- Status: `PASS. Test MAE: 5.1% (Consistent with Validation MAE: 4.9%)`

## DOWNSTREAM DEPENDENCIES
- streamlit-deployment, project-documentation
