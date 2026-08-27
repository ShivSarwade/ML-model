# 03 - Feature Engineering, Leakage Detection & Splitting

## Objective
This folder is the most critical gate in the entire Machine Learning pipeline. It builds predictive signals and guarantees we aren't cheating by peeking into the future.

## What This Achieves
1. **Feature Engineering**: Derives advanced historical features such as the rolling average of the last 3 lectures, and the number of days since the previous lecture.
2. **Leakage Detection**: Mathematically verifies that no engineered feature has an illegal 1.0 correlation with our target variable, ensuring we don't accidentally train on future information.
3. **Temporal Splitting**: Strictly sorts the dataset by time and chronologically splits it into Train, Validation, and Final Test sets, completely avoiding random shuffling to maintain time-series integrity.
