# Requirement Analysis & Deliverables Checklist

This document extracts and tracks all requirements based on the `requirement.md` project specification.

## 1. Technologies & Stack

### MANDATORY
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebooks
- Streamlit OR Power BI (for deployment)

### OPTIONAL
- XGBoost (for advanced algorithmic training)
- CatBoost (for advanced algorithmic training)

### NOT SPECIFIED (PROHIBITED)
- React, Next.js, Node.js, FastAPI, Django
- MongoDB, SQL databases
- Docker, Kubernetes, AWS
- TensorFlow, PyTorch, Deep Learning
- LangChain, RAG, LLMs

## 2. Algorithms

### MANDATORY REGRESSION ALGORITHMS
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting

### MANDATORY CLASSIFICATION ALGORITHMS
- Logistic Regression
- Decision Tree & Random Forest
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- Naïve Bayes

### OPTIONAL ALGORITHMS
- XGBoost (Regression & Classification)
- CatBoost (Regression)

## 3. Evaluation Metrics

### MANDATORY REGRESSION METRICS
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R2 Score

### MANDATORY CLASSIFICATION METRICS
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## 4. Final Deliverables (Section 7)

- [ ] **Handwritten data-collection logbook:** Physical proof of the primary data collection effort.
- [ ] **Raw and cleaned CSV datasets:** Both iterations of the data to reproduce the engineering pipeline.
- [ ] **Source code:** Modularized Jupyter Notebooks detailing EDA, feature engineering, and model training.
- [ ] **Experiment table:** A matrix detailing the configurations and outcomes of all tested algorithms.
- [ ] **Deployment files:** All requisite code for the Streamlit dashboard or Power BI file.
- [ ] **Working application demonstration:** A functional presentation of the user interface executing predictions.
- [ ] **Final report:** A cohesive document synthesizing the analytical findings, optimal scheduling recommendations, and deployment strategy.

## 5. Optional Extensions (Section 6.2)
- [ ] Compare attendance prediction efficacy across entirely different engineering departments.
- [ ] Develop individualized absenteeism predictions for specific core courses.
- [ ] Build an automated recommendation engine that suggests optimal lecture timings based on historical peak attendance hours.
- [ ] Analyze and visualize the direct statistical influence of weather and examinations on cohort behavior.
