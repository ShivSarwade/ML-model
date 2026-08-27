# 01 - Data Cleaning

## Objective
This folder is dedicated to sanitizing the raw attendance dataset (`raw_attendance.csv`) before any modeling begins. 

## What This Achieves
- Converts raw text dates and times into machine-readable `datetime` formats.
- Checks for and resolves missing values or duplicate records.
- Enforces logical bounds (e.g., ensuring `Students_Present` <= `Total_Enrolled`).
- Calculates the true `Attendance_Percentage` safely.
- Outputs the sanitized `attendance_cleaned.csv` for the next phases.

This step guarantees that our downstream algorithms won't crash due to unexpected text formats or NaN errors.
