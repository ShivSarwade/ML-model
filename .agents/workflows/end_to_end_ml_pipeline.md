# Classroom Attendance Prediction: End-to-End Execution Workflow

This workflow document orchestrates the 19 specialized ML skills for this capstone project. When executing this project, you MUST follow these phases and rely on the detailed instructions within each skill located in `.agents/skills/`.

## PHASE 1: Initialization & Data Quality
1. Execute `requirement-analysis` to map constraints.
2. Execute `dataset-validation` on the raw CSV. 
   - **DATA GATE:** Halt if the dataset fails critical checks (e.g., Present > Enrolled).
3. Execute `data-cleaning` to sanitize the records.

## PHASE 2: Engineering & Leakage Prevention
4. Execute `exploratory-data-analysis` to identify time-series and categorical trends.
5. Execute `feature-engineering` to construct temporal/historical signals.
6. Execute `leakage-detection`. 
   - **FEATURE GATE:** Halt immediately if future information or the target variable leaks into the features.
7. Execute `temporal-dataset-splitting` to partition data while strictly preserving chronological order.

## PHASE 3: Modeling & Evaluation
8. Execute `model-training` to build the required baseline models.
9. Execute `cross-validation` using time-aware split techniques.
10. Execute `hyperparameter-optimization` to search for optimal configurations.
11. Execute `model-evaluation` to extract the precise metrics required by the PDF.

## PHASE 4: Iterative Improvement
12. Execute `error-analysis` to find systematic failures (e.g., poor prediction on 8 AM classes).
13. Execute `feature-improvement` to hypothesize and construct new features targeting the identified failures.
14. **Loop:** After adding new features, loop back to Step 6 (`leakage-detection`) and proceed downward. Break the loop when validation performance plateaues.

## PHASE 5: Finalization & Delivery
15. Execute `model-selection` to lock in the final candidate objectively.
   - **MODEL GATE:** Ensure selection is based on validation stability, not overfit training scores.
16. Execute `experiment-tracking` to ensure all trials are permanently recorded for the final report.
17. Execute `prediction-pipeline` to map raw inputs to predictions utilizing the exact same preprocessing logic.
18. Execute `final-model-validation` to evaluate the model on the untouched test set.
   - **FINAL MODEL GATE:** Block deployment if test performance collapses compared to validation performance.
19. Execute `streamlit-deployment` to build the user interface.
20. Execute `project-documentation` to compile the final submission deliverables.
