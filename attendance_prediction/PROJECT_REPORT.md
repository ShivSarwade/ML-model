# Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

---

## 1. Abstract

Student attendance is a critical metric for educational institutions, often correlating directly with academic performance and engagement. This project aims to solve the unpredictability of daily classroom attendance by developing an intelligent predictive system. We utilized a comprehensive dataset encompassing academic schedules, historical attendance records, and contextual factors such as weather, internal test weeks, and adjacent holidays.

Our approach leverages both **Classification** (predicting attendance bands: Low, Medium, High) and **Regression** (predicting the exact attendance percentage) machine learning models. We experimented with a suite of 12 algorithms, including Logistic Regression, Support Vector Machines (SVM), Random Forests, and Gradient Boosting techniques. 

The **XGBoost Classifier** emerged as a top performer for categorical prediction, effectively capturing non-linear relationships in student behavior. Key findings indicate that "Previous Attendance" and "Subject" are the strongest predictors, while factors like "Weather" and "Holiday Proximity" provide moderate but vital nuance. Finally, the selected models were deployed into **AttendAI**, an interactive Streamlit web dashboard that allows faculty members to enter upcoming lecture details and receive real-time attendance estimates and insights.

---

## 2. Introduction

### 2.1 Background
Attendance management in educational institutions has traditionally been a reactive process, primarily focused on record-keeping rather than forecasting. While historical attendance data is useful for identifying chronic absenteeism, it fails to dynamically account for day-to-day fluctuations in student turnout. Relying solely on historical averages ignores the complex web of factors that affect a student's decision to attend a specific lecture, such as the timing of the class, the subject's difficulty, proximity to exams, or even consecutive lecture fatigue. 

### 2.2 Problem Statement
Predict attendance for an upcoming lecture using academic schedule data, historical attendance trends, and contextual environmental factors. 

### 2.3 Motivation
An accurate attendance prediction system is required to transition institutions from reactive monitoring to proactive planning. The primary motivations include:
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
This system covers the predictive modeling of aggregate classroom attendance (percentage and student count) for specific scheduled lectures. It encompasses the end-to-end pipeline from raw CSV data processing to a localized web dashboard deployment.

The system **does not** cover individual student-level attendance prediction (i.e., predicting exactly *which* specific student will be absent). It also does not include real-time hardware integration (e.g., biometrics or RFID) for live attendance capturing.

---

## 3. Literature / Background Study

### 3.1 Attendance Prediction
Existing approaches to attendance management predominantly rely on descriptive analytics—summarizing past data via dashboards to alert administrators of low overall attendance. Early attempts at prediction relied on simple moving averages or linear extrapolation, which often failed to capture the nuances of an academic calendar (e.g., sudden drops before a major exam week).

### 3.2 Machine Learning for Attendance Analysis
Machine learning introduces the ability to identify complex, multi-dimensional patterns in historical attendance. By framing the problem as a supervised learning task, models like Random Forests and Gradient Boosting (XGBoost) can weigh the interacting importance of variables. For instance, ML can learn that a "9:00 AM lecture" might have high attendance on a Tuesday, but significantly lower attendance on a Monday following a long weekend—a pattern that rigid statistical models struggle to capture.

### 3.3 Factors Affecting Attendance
Our study evaluates several critical variables:
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
The system is designed as a linear, reproducible pipeline ensuring strict chronological validation (to prevent future data leakage) and culminating in a user-friendly deployment.

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
