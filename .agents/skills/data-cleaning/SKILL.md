---
name: data-cleaning
description: >-
  Applies validated transformations to raw data to handle missing values, duplicates, and inconsistent formats.
---

# DATA CLEANING

## PURPOSE
To produce a sanitized, standardized dataset ready for exploratory analysis and feature engineering, ensuring no loss of chronological integrity.

## WHEN TO USE
After dataset-validation passes or outputs warnings that can be safely mitigated.

## INPUTS
- Raw attendance dataset (CSV).
- validation_report.

## OUTPUTS
- Cleaned CSV.
- Cleaning log detailing all transformations.

## RESPONSIBILITIES
- Handle missing values appropriately.
- Remove duplicate records.
- Normalize categorical values (e.g., case folding).
- Standardize datetime formatting.

## RULES
- NEVER overwrite or modify the raw CSV file on disk.
- Every transformation MUST be reproducible (set random seeds if stochastic imputation is used, prefer deterministic methods).
- Document exact row reductions (e.g., "Dropped 5 rows due to missing target").

## VALIDATION
- Compare row counts before and after. 
- Ensure no missing values remain in critical columns (Target, Date).

## FAILURE CONDITIONS
- Imputation introduces data leakage (e.g., filling historical missing values using future dataset means).
- Dropping >10% of the dataset due to errors.

## EDGE CASES
- Chronologically inconsistent data: Sort by Date/Time before returning.

## EXAMPLES
- Normalizing subjects: `df['Subject'].str.lower().str.strip()`

## DOWNSTREAM DEPENDENCIES
- exploratory-data-analysis, feature-engineering
