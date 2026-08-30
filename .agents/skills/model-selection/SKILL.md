---
name: model-selection
description: >-
  Objectively selects the final model based on validation stability, avoiding blind selection of highest training scores.
---

# MODEL SELECTION

## PURPOSE
To make the final, documented decision on which algorithm and feature set will be deployed and evaluated against the final test set.

## WHEN TO USE
When the iterative improvement loop is complete and hyperparameter tuning has exhausted meaningful gains.

## INPUTS
- Experiment tracking history.
- Validation and Cross-Validation metrics for all tuned models.

## OUTPUTS
- selected_model (Model object).
- Selection rationale document (reason, metrics, alternatives_rejected).

## RESPONSIBILITIES
- Compare models holistically (Validation score, CV stability, error distribution, complexity).
- Lock in the final model.

## RULES
- Do NOT simply select the model with the highest training score.
- The final test set MUST NOT be used to make this decision.

## VALIDATION
- Rationale must explicitly mention why alternative top models were rejected (e.g., "Random Forest had 1% better validation MAE, but XGBoost was selected due to tighter CV variance").

## FAILURE CONDITIONS
- All models perform worse than a naive baseline (e.g., predicting the mean attendance every time).

## EDGE CASES
- Ties in validation performance: Select the simpler (more interpretable/faster) model.

## EXAMPLES
- Rejecting an overfit model: "Model A Train R2: 0.99, Val R2: 0.50. Rejected."

## DOWNSTREAM DEPENDENCIES
- final-model-validation, experiment-tracking
