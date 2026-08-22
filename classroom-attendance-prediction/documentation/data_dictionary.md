# Data Dictionary

This document defines the schema for the custom Classroom Attendance Dataset.

## Raw Dataset (`data/raw/attendance_raw.csv`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Date` | String (DD-MM-YYYY) | The exact date of the scheduled lecture. |
| `Day_of_Week` | String | e.g., 'Monday', 'Tuesday'. |
| `Subject` | String | Code or Name of the academic subject (e.g., 'CS101'). |
| `Faculty_ID` | String | Unique identifier for the professor. |
| `Start_Time` | String | e.g., '9.15 AM'. |
| `End_Time` | String | e.g., '10.15 AM'. |
| `Lecture_Number` | Integer | Sequence of the lecture on that particular date. |
| `Section` | String | Cohort mapping (e.g., 'A', 'B'). |
| `Total_Enrolled` | Integer | Maximum potential class size. |
| `Students_Present` | Integer | Raw count of attendance. |
| `Internal_Test_Week` | String ('Yes'/'No') | Academic calendar flag. |
| `Holiday_Before_After` | String ('Yes'/'No') | Proximity to a holiday. |
| `Weather` | String | Environmental condition flag. |

## Processed & Engineered Features

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Attendance_Percentage` | Float | **(Target Variable for Regression)** `(Students_Present / Total_Enrolled) * 100`. |
| `Attendance_Class` | String | **(Target Variable for Classification)** Binned into Low, Medium, High. |
| `Rolling_Avg_3` | Float | The historical momentum (average attendance of the previous 3 lectures for this specific cohort). Strict `.shift(1)` logic is applied. |
| `Week_Number` | Integer | Academic calendar week. |
| `Day_of_Semester` | Integer | Integer days elapsed since the start of the semester. |
| `Time_of_Day` | String | Categorical clustering (Morning/Afternoon). |
| `Days_Since_Last_Holiday` | Integer | Running counter identifying post-holiday slumps. |
| `Week_Before_Exam` | Integer (0/1) | Boolean indicator if an exam is imminent. |
| `Consecutive_Lecture_Count` | Integer | Cumulative lectures for this cohort on the current date. |
| `Monthly_Avg_Attendance` | Float | Expanding mean historical trend. |
