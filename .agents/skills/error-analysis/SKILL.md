---
name: error-analysis
description: >-
  Investigates specific instances and cohorts where the model performs poorly to identify systematic bias.
---

# ERROR ANALYSIS

## PURPOSE
To look beyond aggregate metrics and understand exactly *where* and *why* the model is failing, enabling targeted feature improvements.

## WHEN TO USE
After initial model evaluation reveals suboptimal performance on the validation set.

## INPUTS
- Validation predictions.
- validation_dataset (with features).

## OUTPUTS
- Error breakdown by cohort (high-error groups, low-error groups).
- Identification of systematic bias (underprediction vs overprediction).

## RESPONSIBILITIES
- Slice absolute errors by categorical features (Subject, Day, Time, Section).
- Analyze performance during edge cases (Test week, Holiday proximity).

## RULES
- Do not analyze errors on the final test set until the project is absolutely finalized.

## VALIDATION
- Ensure the sum of residuals across all cohorts maps back to the global MAE/RMSE.

## FAILURE CONDITIONS
- Model predictions are completely degenerate (predicting a constant value for all inputs), rendering subgroup analysis useless.

## EDGE CASES
- Unstable segments: A single specific subject having erratic attendance causing global error spikes.

## EXAMPLES
- Grouping residuals: `val_df.groupby('Day of Week')['Absolute_Error'].mean()`

## DOWNSTREAM DEPENDENCIES
- feature-improvement
