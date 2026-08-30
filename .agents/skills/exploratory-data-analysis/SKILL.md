---
name: exploratory-data-analysis
description: >-
  Performs EDA on the cleaned attendance dataset, analyzing distributions, trends, and relationships.
---

# EXPLORATORY DATA ANALYSIS (EDA)

## PURPOSE
To automatically discover patterns, trends, and potential predictive signals in the cleaned historical attendance data.

## WHEN TO USE
After data-cleaning is complete, before formal feature engineering.

## INPUTS
- Cleaned CSV.

## OUTPUTS
- Summary statistics.
- Visualizations (Attendance distribution, day/time trends, subject trends).
- EDA findings log (question, analysis, result, evidence, possible_implication).

## RESPONSIBILITIES
- Analyze attendance distribution and detect class imbalance or skew.
- Analyze test-week and holiday proximity effects.
- Analyze variations across day, time, and subject.

## RULES
- Do NOT claim causation from correlation.
- Limit analysis strictly to variables relevant to the project specification.

## VALIDATION
- Ensure all generated visualizations have appropriate labels, titles, and legends.

## FAILURE CONDITIONS
- Dataset lacks sufficient variance (e.g., attendance is exactly 100% for all records).

## EDGE CASES
- Extremely small datasets: Skip complex distribution plotting, use simple bar charts.

## EXAMPLES
- Question: Do 8 AM classes have lower attendance? -> Groupby Time, calculate mean, plot bar chart.

## DOWNSTREAM DEPENDENCIES
- feature-engineering, project-documentation
