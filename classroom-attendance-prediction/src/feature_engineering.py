import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting Feature Engineering...")
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Temporal Features
    df['Week_Number'] = df['Date'].dt.isocalendar().week
    df['Day_of_Semester'] = (df['Date'] - df['Date'].min()).dt.days
    
    def get_time_of_day(time_str):
        try:
            # e.g., '9.15 AM', '1.30 PM'
            if 'PM' in str(time_str) and not str(time_str).startswith('12'):
                return 'Afternoon'
            return 'Morning'
        except:
            return 'Morning'
            
    df['Time_of_Day'] = df['Start_Time'].apply(get_time_of_day)
    
    # Week before exam (Assuming we know the academic calendar, we can check if the next week is an exam)
    # We will just map it based on if the current Week_Number + 1 has any 'Yes' for Internal_Test_Week
    test_weeks = df[df['Internal_Test_Week'].str.lower() == 'yes']['Week_Number'].unique()
    df['Week_Before_Exam'] = df['Week_Number'].apply(lambda w: 1 if (w + 1) in test_weeks else 0)
    
    # Days since last holiday
    # We identify holiday dates and calculate days since
    # Assuming 'Holiday_Before_After' means a holiday is adjacent. We'll just create a running counter
    # as a simple proxy if we don't have exact holiday dates.
    df['Is_Holiday_Adjacent'] = df['Holiday_Before_After'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    # This is a bit tricky without a real calendar, we'll proxy it by counting days since the last time Is_Holiday_Adjacent was 1
    # Sort by date first
    df = df.sort_values(by=['Date', 'Lecture_Number']).reset_index(drop=True)
    
    last_holiday_date = df['Date'].min()
    days_since_list = []
    for idx, row in df.iterrows():
        if row['Is_Holiday_Adjacent'] == 1:
            last_holiday_date = row['Date']
        days_since = (row['Date'] - last_holiday_date).days
        days_since_list.append(max(0, days_since))
    df['Days_Since_Last_Holiday'] = days_since_list

    # Consecutive lecture count for the student cohort (Section) on that day
    df['Consecutive_Lecture_Count'] = df.groupby(['Date', 'Section']).cumcount() + 1
    
    # 2. Historical & Macro Trends
    # Ensure data is sorted by Date and Time (Lecture_Number)
    df = df.sort_values(by=['Date', 'Lecture_Number']).reset_index(drop=True)
    
    # Calculate rolling average on shifted attendance (to avoid leakage)
    df['Rolling_Avg_3'] = df.groupby(['Subject', 'Section'])['Attendance_Percentage'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Monthly average attendance and macro-attendance trends
    df['Month'] = df['Date'].dt.month
    df['Monthly_Avg_Attendance'] = df.groupby(['Month', 'Subject'])['Attendance_Percentage'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Fill NaN for first lectures with the global mean or 0
    global_mean = df['Attendance_Percentage'].mean()
    df['Rolling_Avg_3'] = df['Rolling_Avg_3'].fillna(global_mean)
    df['Monthly_Avg_Attendance'] = df['Monthly_Avg_Attendance'].fillna(global_mean)
    
    # 3. Target Variable creation (Classification)
    def classify_attendance(perc):
        if perc < 14.0:
            return 'Low'
        elif perc <= 25.0:
            return 'Medium'
        else:
            return 'High'
            
    df['Attendance_Class'] = df['Attendance_Percentage'].apply(classify_attendance)
    
    logging.info("Feature Engineering complete.")
    return df

def check_leakage(df: pd.DataFrame):
    """Basic check to ensure we drop target-leaking columns."""
    logging.info("Performing Leakage Detection...")
    leaky_cols = ['Students_Present', 'Attendance_Percentage', 'Total_Enrolled']
    
    for col in leaky_cols:
        if col in df.columns:
            logging.info(f"Target column '{col}' is present. It must be dropped before training.")
            
    logging.info("Leakage check complete.")

def split_chronological(df: pd.DataFrame, output_dir: str):
    logging.info("Splitting dataset chronologically...")
    
    # Drop leaky columns from the output features, except we might need them for reference.
    # We will save the full dataset, but models must drop them.
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logging.info(f"Train set: {len(train_df)} records")
    logging.info(f"Val set: {len(val_df)} records")
    logging.info(f"Test set: {len(test_df)} records")
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    logging.info("Splits saved successfully.")

if __name__ == "__main__":
    processed_path = "../data/processed/attendance_cleaned.csv"
    output_dir = "../data/processed/"
    
    try:
        df = pd.read_csv(processed_path)
        df_engineered = engineer_features(df)
        check_leakage(df_engineered)
        split_chronological(df_engineered, output_dir)
    except Exception as e:
        logging.error(f"Feature Engineering pipeline failed: {e}")
