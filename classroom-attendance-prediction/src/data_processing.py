import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_data(df: pd.DataFrame) -> bool:
    """Validates the raw attendance dataset for critical requirements."""
    logging.info("Starting dataset validation...")
    
    # 1. Check required columns
    required_cols = [
        'Date', 'Subject', 'Total_Enrolled', 'Students_Present', 
        'Start_Time', 'Internal_Test_Week'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return False
        
    # 2. Check Logical Bounds
    invalid_bounds = df[df['Students_Present'] > df['Total_Enrolled']]
    if not invalid_bounds.empty:
        logging.warning(f"Found {len(invalid_bounds)} records where Students Present > Enrolled.")
        
    zero_enrolled = df[df['Total_Enrolled'] <= 0]
    if not zero_enrolled.empty:
        logging.error(f"Found {len(zero_enrolled)} records with zero or negative Total_Enrolled.")
        return False
        
    # 3. Check for Nulls in critical columns
    nulls = df[['Date', 'Subject', 'Students_Present', 'Total_Enrolled']].isnull().sum()
    if nulls.any():
        logging.warning(f"Null values found in critical columns:\n{nulls[nulls > 0]}")
        
    logging.info("Validation Complete.")
    return True

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and formats the raw attendance dataset."""
    logging.info("Starting data cleaning...")
    
    # Copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # 1. Handle Missing/Invalid rows
    # Drop rows where target is missing
    df_clean = df_clean.dropna(subset=['Students_Present', 'Total_Enrolled'])
    
    # Cap Students_Present at Total_Enrolled just in case
    df_clean['Students_Present'] = np.minimum(df_clean['Students_Present'], df_clean['Total_Enrolled'])
    
    # 2. Formatting
    # Convert Date to datetime format explicitly using dayfirst=True since format is DD-MM-YYYY
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d-%m-%Y', errors='coerce')
    
    # Clean string columns
    str_cols = ['Day_of_Week', 'Subject', 'Faculty_ID', 'Internal_Test_Week', 'Weather']
    for col in str_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
    # Calculate/Recalculate Attendance Percentage to ensure consistency
    df_clean['Attendance_Percentage'] = (df_clean['Students_Present'] / df_clean['Total_Enrolled']) * 100
    
    # Sort chronologically (Crucial for later Temporal Splitting and feature engineering)
    df_clean = df_clean.sort_values(by=['Date', 'Lecture_Number']).reset_index(drop=True)
    
    logging.info(f"Data cleaning complete. Retained {len(df_clean)} records.")
    return df_clean

if __name__ == "__main__":
    raw_path = "../data/raw/attendance_raw.csv"
    processed_path = "../data/processed/attendance_cleaned.csv"
    
    try:
        df_raw = pd.read_csv(raw_path)
        is_valid = validate_data(df_raw)
        
        # Even if there are warnings, we proceed with cleaning
        df_cleaned = clean_data(df_raw)
        
        df_cleaned.to_csv(processed_path, index=False)
        logging.info(f"Saved cleaned data to {processed_path}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
