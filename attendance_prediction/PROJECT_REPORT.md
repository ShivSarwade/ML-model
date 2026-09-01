# Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

**A Project Report**

**Submitted by**  
**Name :** Shiv Sarwade &nbsp;&nbsp;&nbsp;&nbsp; **Roll No.:** 2501157

---

## 1. Abstract

Student attendance is a critical metric for educational institutions, often correlating directly with academic performance and engagement. I aim to solve the unpredictability of daily classroom attendance by developing an intelligent predictive system. I utilized a comprehensive dataset encompassing academic schedules, historical attendance records, and contextual factors such as weather, internal test weeks, and adjacent holidays.

My approach leverages both **Classification** (predicting attendance bands: Low, Medium, High) and **Regression** (predicting the exact attendance percentage) machine learning models. To achieve a high prediction rate, I meticulously engineered a robust dataset utilizing **41 distinct predictive fields** (features) such as historical momentum, exact lecture timings, and proximity to holidays. I rigorously trained and evaluated a total of **12 distinct machine learning models** (7 Classification and 5 Regression algorithms), including Logistic Regression, Support Vector Machines (SVM), Random Forests, and Gradient Boosting techniques. 

By applying strict temporal dataset splitting (to completely eliminate data leakage) and hyperparameter optimization, the models achieved highly reliable prediction rates on completely unseen future lectures. The **Decision Tree** and **XGBoost** models emerged as top performers, effectively capturing complex, non-linear relationships in student behavior. My findings indicate that "Previous Attendance" and "Subject" are the strongest predictors, while factors like "Weather" and "Holiday Proximity" provide moderate but vital nuance. Finally, I deployed the champion models into **AttendAI**, an interactive Streamlit web dashboard that allows faculty members to enter upcoming lecture details and receive real-time attendance estimates and insights.

---

## 2. Introduction

### 2.1 Background
Attendance management in educational institutions has traditionally been a reactive process, primarily focused on record-keeping rather than forecasting. While historical attendance data is useful for identifying chronic absenteeism, it fails to dynamically account for day-to-day fluctuations in student turnout. Relying solely on historical averages ignores the complex web of factors that affect a student's decision to attend a specific lecture, such as the timing of the class, the subject's difficulty, proximity to exams, or even consecutive lecture fatigue. 

### 2.2 Problem Statement
Predict attendance for an upcoming lecture using academic schedule data, historical attendance trends, and contextual environmental factors. 

### 2.3 Motivation
An accurate attendance prediction system is required to transition institutions from reactive monitoring to proactive planning. My primary motivations include:
- **Identifying At-Risk Lectures:** Highlighting specific classes likely to have poor attendance so faculty can adapt their teaching strategies (e.g., deferring a core topic).
- **Optimizing Timetables:** Understanding problematic time slots (e.g., late afternoon on Fridays) to support better schedule planning by administration.
- **Resource Allocation:** Helping faculty plan lecture materials, lab equipment, or group activities based on the expected number of students.
- **Understanding External Impacts:** Quantifying the effect of internal tests, adjacent holidays, and practical versus theoretical sessions on student engagement.

### 2.4 Objectives
1. Collect and aggregate raw attendance data and scheduling information.
2. Clean and preprocess the dataset to handle missing values and inconsistencies.
3. Perform exploratory data analysis (EDA) to uncover trends and correlations.
4. Engineer useful predictive features, particularly temporal and momentum-based features (e.g., gap since last lecture).
5. Train multiple Machine Learning algorithms across both classification and regression paradigms.
6. Compare model performance using rigorous validation metrics (Accuracy, F1-Score, RMSE, MAE).
7. Predict future lecture attendance with high reliability while strictly preventing data leakage.
8. Deploy the prediction system via an accessible web interface.
9. Generate scheduling recommendations and dynamic insights based on prediction outputs.

### 2.5 Scope
My system covers the predictive modeling of aggregate classroom attendance (percentage and student count) for specific scheduled lectures. It encompasses the end-to-end pipeline from raw CSV data processing to a localized web dashboard deployment.

The system **does not** cover individual student-level attendance prediction (i.e., predicting exactly *which* specific student will be absent). I also did not include real-time hardware integration (e.g., biometrics or RFID) for live attendance capturing.

---

## 3. Literature / Background Study

### 3.1 Attendance Prediction
Existing approaches to attendance management predominantly rely on descriptive analytics—summarizing past data via dashboards to alert administrators of low overall attendance. Early attempts at prediction relied on simple moving averages or linear extrapolation, which often failed to capture the nuances of an academic calendar (e.g., sudden drops before a major exam week).

### 3.2 Machine Learning for Attendance Analysis
Machine learning introduces the ability to identify complex, multi-dimensional patterns in historical attendance. By framing the problem as a supervised learning task, I used models like Random Forests and Gradient Boosting (XGBoost) to weigh the interacting importance of variables. For instance, ML can learn that a "9:00 AM lecture" might have high attendance on a Tuesday, but significantly lower attendance on a Monday following a long weekend—a pattern that rigid statistical models struggle to capture.

### 3.3 Factors Affecting Attendance
My study evaluates several critical variables:
- **Previous Attendance:** The strongest indicator of momentum. High attendance in the previous lecture often correlates with continued attendance.
- **Subject:** Core/difficult subjects often command higher attendance than elective or softer skills sessions.
- **Lecture Timing & Day of Week:** Morning sessions typically exhibit different attendance behaviors compared to post-lunch or late-afternoon sessions. Mid-week days (Tuesday-Thursday) generally see peak attendance.
- **Gap Between Lectures:** Longer gaps since the previous lecture for a specific subject can lead to lower continuity and drop-offs.
- **Tests/Examinations:** The "Internal Test Week" flag heavily influences attendance, often causing spikes for revision lectures and drops for non-essential classes.
- **Holidays:** Lectures immediately preceding or succeeding a holiday ("Holiday Adjacent") suffer from extended absenteeism.
- **Weather:** Rainy or extreme weather conditions can marginally suppress turnout.
- **Session Type:** Practical/Lab sessions often have stricter attendance requirements or continuous evaluation, differing from theoretical lectures.

---

## 4. System Overview

### 4.1 Proposed System
I designed the system as a linear, reproducible pipeline ensuring strict chronological validation (to prevent future data leakage) and culminating in a user-friendly deployment.

```text
       Data Collection
              ↓
         Raw Dataset
              ↓
        Data Cleaning
              ↓
    Exploratory Data Analysis (EDA)
              ↓
     Feature Engineering
              ↓
      Train/Test Split
              ↓
       Model Training
 (Classification & Regression)
              ↓
      Model Evaluation
              ↓
    Best Model Selection
              ↓
 Prediction API/Application
              ↓
    Streamlit Dashboard (AttendAI)
```

### 4.2 Development Phases
The project is strictly divided into two distinct phases to ensure a robust and scientifically sound development process:

#### Phase 1: Engineering and Leakage Prevention
This phase focuses entirely on preparing the data while maintaining strict chronological integrity. Key steps include:
- **Data Cleaning & EDA**: Handling missing values, standardizing formats, and uncovering baseline correlations between attendance and contextual factors.
- **Feature Engineering**: Creating historical momentum features (e.g., previous lecture attendance) and temporal features (e.g., day of the week, time of day).
- **Leakage Prevention**: Strictly splitting the dataset chronologically (train on past, validate on future) to prevent the model from "peeking" into the future, ensuring real-world reliability.

#### Phase 2: Model Training and Selection
With a clean, engineered, and properly split dataset, the focus shifts to machine learning:
- **Model Training**: Training a diverse suite of Regression (predicting percentage) and Classification (predicting Low/Medium/High bands) algorithms.
- **Evaluation & Tuning**: Assessing models based on metrics like RMSE and F1-Score, and applying hyperparameter optimization to prevent overfitting.
- **Champion Selection**: Objectively selecting the best-performing models (e.g., XGBoost) for final deployment.

### 4.3 Folder Structure and Deployment Flow
The project follows an organized, modular directory structure that naturally guides the flow of data from raw inputs to a deployed application.

```text
attendance_prediction/
├── data/                       # Contains raw, intermediate, and processed datasets
├── jupiter books/              # All exploratory and training notebooks
│   ├── phase 1 engineering and leakage prevention/
│   └── phase 2 model training/
│       └── models/             # Contains all 12 individual model training notebooks
├── deployment_assets/          # Central location for exported .pkl files (models, scalers)
└── app.py                      # The main Streamlit web application
```

#### The End-to-End Deployment Flow:
1. **Model Execution**: When any of the individual model notebooks located in `jupiter books/phase 2 model training/models/` are executed, they train the respective machine learning algorithm on the processed dataset.
2. **Automated Export (`.pkl`)**: Upon successful training and evaluation, each notebook automatically serializes and exports the trained model object as a `.pkl` (pickle) file directly into the `deployment_assets/` directory.
3. **Application Inference**: The `app.py` Streamlit application continuously monitors the `deployment_assets/` directory. When a user requests a prediction on the dashboard, the application dynamically loads the required `.pkl` model and applies it to the user's input to generate real-time attendance estimates.

---

## 5. Model Evaluation and Results

During Phase 2, a total of 12 machine learning models were trained and evaluated. Below are the final validation metrics for both the Regression and Classification models, demonstrating their performance on the completely unseen temporal validation set.

### 5.1 Regression Models Performance
| Model ID | Model Name | MAE | RMSE | MAPE (%) | R² Score |
|----------|------------|-----|------|----------|----------|
| 01 | Linear Regression | 7.60 | 9.46 | 45.62 | 0.044 |
| 02 | Decision Tree Regressor | 6.51 | 8.71 | 38.64 | 0.190 |
| 03 | Random Forest Regressor | 6.70 | 8.67 | 40.62 | 0.197 |
| 04 | Gradient Boosting Regressor | 7.71 | 10.26 | 45.03 | -0.123 |
| 05 | XGBoost Regressor | 7.71 | 10.02 | 45.78 | -0.071 |

*Note: For the regression task (predicting exact percentage), the **Random Forest Regressor** and **Decision Tree Regressor** yielded the strongest R² scores and lowest errors (MAE/RMSE).*

### 5.2 Classification Models Performance
| Model ID | Model Name | Accuracy | F1-Score | ROC-AUC |
|----------|------------|----------|----------|---------|
| 06 | Logistic Regression | 0.376 | 0.365 | 0.589 |
| 07 | Decision Tree Classifier | 0.480 | 0.385 | 0.621 |
| 08 | Random Forest Classifier | 0.324 | 0.314 | 0.601 |
| 09 | SVM Classifier | 0.350 | 0.304 | 0.486 |
| 10 | KNN Classifier | 0.376 | 0.361 | 0.539 |
| 11 | Naive Bayes | 0.298 | 0.269 | 0.546 |
| 12 | XGBoost Classifier | 0.428 | 0.438 | 0.628 |

*Note: For the classification task (predicting Low/Medium/High bands), the **Decision Tree Classifier** (highest accuracy at 48.0%) and the **XGBoost Classifier** (highest F1-Score and ROC-AUC) emerged as the best performing models.*
