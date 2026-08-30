---
name: feature-improvement
description: >-
  Iteratively proposes, tests, and validates new features based on error analysis insights.
---

# FEATURE IMPROVEMENT

## PURPOSE
To iteratively raise the performance ceiling of the model by addressing specific weaknesses found during Error Analysis without introducing leakage.

## WHEN TO USE
Inside the iterative model improvement loop, after Error Analysis identifies a systematic weakness.

## INPUTS
- Current best model.
- Error Analysis outputs.
- Base dataset.

## OUTPUTS
- Proposed new feature logic.
- Comparison metrics (Old Model vs New Model).

## RESPONSIBILITIES
- Propose feature transformations (e.g., interaction terms, longer rolling windows) targeting identified weaknesses.
- Pass the new feature through Leakage Detection.
- Retrain and validate.

## RULES
- Retain new features ONLY if they improve genuine validation/generalization performance.
- Reject features that improve training performance but degrade or stall validation performance (overfitting).
- Maintain a feature experiment history.

## VALIDATION
- Formal comparison of delta in Validation MAE / F1 before and after feature addition.

## FAILURE CONDITIONS
- Proposed feature triggers the leakage-detection skill (FAIL).

## EDGE CASES
- Feature improves performance for one subject but ruins it for another: Reject, or propose subject-specific interaction features.

## EXAMPLES
- Insight: "Model struggles on 8 AM classes on Mondays." -> Proposed Feature: `Is_Monday_Morning` boolean flag.

## DOWNSTREAM DEPENDENCIES
- leakage-detection, model-selection
