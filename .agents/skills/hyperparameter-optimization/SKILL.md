---
name: hyperparameter-optimization
description: >-
  Systematically tunes model hyperparameters using training and validation data without overfitting.
---

# HYPERPARAMETER OPTIMIZATION

## PURPOSE
To improve baseline model performance by searching for optimal algorithm configurations using structured search strategies.

## WHEN TO USE
After baseline models are trained and evaluated, targeting the most promising algorithms.

## INPUTS
- Top-performing baseline models.
- train_dataset
- validation_dataset

## OUTPUTS
- Tuned model objects.
- Trial records (best parameters, validation metrics).

## RESPONSIBILITIES
- Define reasonable, bounded search spaces for targeted algorithms.
- Execute Grid Search, Random Search, or Bayesian Optimization.
- Track all trials and select the best configuration based on validation metrics.

## RULES
- Tune ONLY using training/validation data. NEVER tune against the final test set.
- Keep the search space reasonable to prevent excessive compute time.

## VALIDATION
- Compare tuned model validation score against baseline model validation score. If worse, revert to baseline.

## FAILURE CONDITIONS
- Optimization loop fails to converge or times out.

## EDGE CASES
- Small validation set leading to extreme variance in trial scores: Flag risk of validation overfitting.

## EXAMPLES
- Random Forest search space: `max_depth: [5, 10, None], n_estimators: [50, 100, 200]`

## DOWNSTREAM DEPENDENCIES
- model-evaluation, model-selection
