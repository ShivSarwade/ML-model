---
name: prediction-pipeline
description: >-
  Creates a production-style inference pipeline to securely map raw user inputs to a final model prediction.
---

# PREDICTION PIPELINE

## PURPOSE
To encapsulate data preprocessing, feature engineering, and model inference into a single, cohesive, production-ready function/class.

## WHEN TO USE
After the final model is selected, to prepare for dashboard deployment.

## INPUTS
- Raw lecture information (from user or system).
- Pickled final model and preprocessors.

## OUTPUTS
- Predicted attendance (value or category).

## RESPONSIBILITIES
- Validate input schemas.
- Re-apply EXACTLY the same preprocessing steps used in training.
- Generate dynamic features (e.g., mapping a raw time to a 'Morning' cluster).
- Handle prediction errors.

## RULES
- Training and inference feature transformations MUST remain 100% consistent.
- Handle missing inputs gracefully during inference if possible.

## VALIDATION
- Pass a known training sample through the inference pipeline and verify the output exactly matches the training prediction.

## FAILURE CONDITIONS
- Feature schema mismatch (e.g., inference input is missing a one-hot encoded column expected by the model).

## EDGE CASES
- Unseen categorical values (New Subject / New Faculty): Ensure the pipeline handles this without crashing (e.g., mapping to a generic/unknown category).

## EXAMPLES
- `def predict_attendance(raw_json): processed = preprocess(raw_json); return model.predict(processed)`

## DOWNSTREAM DEPENDENCIES
- final-model-validation, streamlit-deployment
