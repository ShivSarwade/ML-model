---
name: feature-engineering
description: >-
  Creates advanced temporal and historical features from the cleaned dataset as specified by the project requirements.
---

# FEATURE ENGINEERING

## PURPOSE
To construct meaningful predictive signals (features) that models can use to predict future attendance, strictly using past information.

## WHEN TO USE
After EDA is complete, to prepare the dataset for model training.

## INPUTS
- Cleaned CSV.

## OUTPUTS
- Engineered CSV dataset.
- Feature dictionary (name, definition, formula, data source, availability_time, leakage_risk).

## RESPONSIBILITIES
- Implement required PDF features: Day of semester, Week number, Days since holiday, Consecutive lecture count, Previous lecture attendance, Rolling average (3 lectures), Monthly average, Time-of-day clusters, Week-before-exam flag.
- Ensure strict temporal causality.

## RULES
- A feature MUST ONLY use information available strictly BEFORE the prediction event time.
- Categorical features must be appropriately encoded (e.g., One-Hot).

## VALIDATION
- Check that 'Previous lecture attendance' actually shifts data chronologically and grouped by the correct cohort (Subject/Section).

## FAILURE CONDITIONS
- 'Previous lecture' matches 'Current lecture' exactly (shift error).

## EDGE CASES
- First lecture of semester / First lecture after gap: Impute previous attendance safely (e.g., use global mean or domain default) and document it.
- No previous 3 lectures available: Use a growing window or fallback value.

## EXAMPLES
- Rolling average: `df.groupby(['Subject', 'Section'])['Attendance'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())`

## DOWNSTREAM DEPENDENCIES
- leakage-detection, temporal-dataset-splitting
