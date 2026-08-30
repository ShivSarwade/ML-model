---
name: temporal-dataset-splitting
description: >-
  Splits the dataset into train, validation, and final test sets preserving strict chronological order.
---

# TEMPORAL DATASET SPLITTING

## PURPOSE
To create evaluation sets that accurately simulate predicting the future, preventing random split contamination in time-series data.

## WHEN TO USE
After leakage-detection passes, before model training.

## INPUTS
- Engineered dataset (cleared of leakage).

## OUTPUTS
- train_dataset
- validation_dataset
- test_dataset
- split_report (exact date ranges for each split)

## RESPONSIBILITIES
- Sort dataset purely by chronological DateTime.
- Partition data temporally (e.g., OLDER DATA -> Train, MORE RECENT -> Validation, LATEST -> Final Test).

## RULES
- NEVER use random shuffling (`train_test_split(shuffle=True)`) for splitting the dataset globally.
- The final test set MUST remain entirely untouched until the model-selection phase.

## VALIDATION
- Max timestamp in Train <= Min timestamp in Validation <= Min timestamp in Test.

## FAILURE CONDITIONS
- Insufficient data volume to create meaningful temporal splits (e.g., only 2 days of data collected).

## EDGE CASES
- Overlapping timestamps: Resolve by maintaining stable sort on secondary identifiers (e.g., Lecture Number).

## EXAMPLES
- Split point calculation: `split_idx = int(len(df) * 0.8); train, val = df.iloc[:split_idx], df.iloc[split_idx:]`

## DOWNSTREAM DEPENDENCIES
- model-training, cross-validation
