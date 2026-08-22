import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def perform_eda(df_path: str, output_dir: str):
    logging.info("Starting EDA...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(df_path)
    
    # 1. Attendance Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Attendance_Percentage'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Attendance Percentage')
    plt.xlabel('Attendance %')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attendance_distribution.png'))
    plt.close()
    
    # 2. Attendance by Day of Week
    # Reorder days for plotting
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Day_of_Week', y='Attendance_Percentage', data=df, order=days_order, palette='viridis')
    plt.title('Attendance by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Attendance %')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attendance_by_day.png'))
    plt.close()
    
    # 3. Attendance by Time
    plt.figure(figsize=(12, 6))
    # Assuming Start_Time is categorical or string format
    sns.boxplot(x='Start_Time', y='Attendance_Percentage', data=df, palette='magma')
    plt.title('Attendance by Start Time')
    plt.xlabel('Start Time')
    plt.ylabel('Attendance %')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attendance_by_time.png'))
    plt.close()
    
    # 4. Attendance by Subject
    plt.figure(figsize=(12, 8))
    sns.boxplot(y='Subject', x='Attendance_Percentage', data=df, palette='Set2')
    plt.title('Attendance by Subject')
    plt.xlabel('Attendance %')
    plt.ylabel('Subject')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attendance_by_subject.png'))
    plt.close()
    
    logging.info(f"EDA visualizations saved to {output_dir}")

if __name__ == "__main__":
    processed_path = "../data/processed/attendance_cleaned.csv"
    viz_dir = "../visualizations/"
    
    try:
        perform_eda(processed_path, viz_dir)
    except Exception as e:
        logging.error(f"EDA pipeline failed: {e}")
