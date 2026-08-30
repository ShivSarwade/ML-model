---
name: model-training
description: >-
  Trains the suite of specified baseline Regression and Classification algorithms.
---

# MODEL TRAINING

## PURPOSE
To train multiple baseline ML algorithms strictly defined in the project specification to establish performance benchmarks.

## WHEN TO USE
After temporal dataset splitting provides the training dataset.

## INPUTS
- train_dataset
- validation_dataset (optional, for early stopping)
- ML problem formulation (Regression or Classification)

## OUTPUTS
- Fitted model objects.
- Baseline configuration records.

## RESPONSIBILITIES
- Apply correct preprocessing pipelines (Imputation, Scaling, Encoding).
- Train Regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost/CatBoost).
- Train Classification models (Logistic Regression, Decision Tree, Random Forest, SVM, k-NN, Naïve Bayes, XGBoost).

## RULES
- Preprocessing objects (Scalers, Encoders) MUST be fit ONLY on the `train_dataset`.
- Do not attempt to train models not explicitly authorized by the project specification without user permission.
- Make training reproducible by explicitly setting `random_state`.

## VALIDATION
- Ensure models can generate a valid prediction shape on a dummy sample.

## FAILURE CONDITIONS
- Missing values leak into models requiring dense data (e.g., SVM).
- Target variable contains unseen categorical labels.

## EDGE CASES
- Imbalanced classification classes: Apply class weights if necessary.

## EXAMPLES
- Pipeline setup: `Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))])`

## DOWNSTREAM DEPENDENCIES
- model-evaluation, hyperparameter-optimization
