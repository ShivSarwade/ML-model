---
name: dataset-validation
description: >-
  Validates the raw attendance dataset for required columns, logical constraints, and data quality issues.
---

# DATASET VALIDATION

## PURPOSE
To assure the fundamental quality, schema, and logical integrity of the raw attendance dataset before any cleaning or modeling occurs.

## WHEN TO USE
Immediately after data collection is complete and the raw CSV is provided, or when a new version of the raw dataset is uploaded.

## INPUTS
- Raw attendance dataset (CSV).
- Expected data dictionary from requirement-analysis.

## OUTPUTS
- validation_report
- data_quality_report
- critical_errors
- warnings

## RESPONSIBILITIES
- Validate column existence and data types.
- Check logical constraints (e.g., 0 <= Students Present <= Total Enrolled Students).
- Detect impossible records, duplicates, or extreme outliers.

## RULES
- Do NOT modify the raw dataset. Read-only validation.
- Attendance percentage must be calculated correctly if present, or verified against Present/Enrolled columns.

## VALIDATION
- Output report must clearly list pass/fail for every expected column.

## FAILURE CONDITIONS
- Missing mandatory columns (e.g., Date, Subject, Students Present).
- Impossible logical bounds (Students Present > Enrolled) exceeding an acceptable noise threshold.

## EDGE CASES
- Missing enrolled student counts: Warn and suggest imputation or failure based on severity.
- Zero enrolled students: Flag as critical error (division by zero risk).

## EXAMPLES
- Checking bounds: `df['Students Present'] <= df['Total Enrolled']`

## DOWNSTREAM DEPENDENCIES
- data-cleaning
