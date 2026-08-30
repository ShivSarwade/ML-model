---
name: experiment-tracking
description: >-
  Maintains a lightweight, non-destructive ledger of all ML experiments for final reporting.
---

# EXPERIMENT TRACKING

## PURPOSE
To ensure 100% reproducibility and provide the necessary data for the mandatory "Experiment Table" project deliverable.

## WHEN TO USE
Continuously throughout training, tuning, and selection phases.

## INPUTS
- Training configurations (features, model, hyperparams).
- Evaluation metrics.

## OUTPUTS
- experiment_results.csv (or similar ledger).

## RESPONSIBILITIES
- Record experiment_id, timestamp, feature version, model, hyperparams, periods, and metrics.

## RULES
- NEVER overwrite previous experiments. Always append.
- Ensure the format maps easily to a Markdown/PDF table for final documentation.

## VALIDATION
- Ensure every column required by the PDF experiment table is present in the ledger.

## FAILURE CONDITIONS
- Loss of experiment history due to file overwrite.

## EDGE CASES
- Crash during training: Log experiment status as 'FAILED' with traceback notes.

## EXAMPLES
- Append row: `[EXP_004, 2026-08-22, Feat_v2, XGBoost, {depth:3}, Val_MAE: 4.2]`

## DOWNSTREAM DEPENDENCIES
- project-documentation
