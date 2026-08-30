---
name: cross-validation
description: >-
  Performs time-aware cross-validation to assess model stability and generalization.
---

# CROSS-VALIDATION

## PURPOSE
To robustly estimate future model performance across multiple historical time periods, avoiding the unreliability of a single validation split.

## WHEN TO USE
During hyperparameter optimization or final model assessment before testing.

## INPUTS
- train_dataset
- Fitted models

## OUTPUTS
- Cross-validation metric distributions (mean, std dev).
- Fold boundary records.

## RESPONSIBILITIES
- Implement TimeSeriesSplit (or equivalent expanding/sliding window approach).
- Evaluate stability across folds.

## RULES
- Do NOT use standard random K-Fold CV on temporal data. It causes future-to-past data leakage within folds.
- Preserve temporal ordering at all times.

## VALIDATION
- Verify that for every fold `k`, the training indices strictly precede the validation indices temporally.

## FAILURE CONDITIONS
- Dataset too small to support multiple temporal folds.

## EDGE CASES
- Highly fluctuating metrics across folds: Indicates unstable model or highly volatile periods (e.g., midterm weeks). Record and flag.

## EXAMPLES
- Sklearn `TimeSeriesSplit(n_splits=5)`

## DOWNSTREAM DEPENDENCIES
- model-evaluation, model-selection
