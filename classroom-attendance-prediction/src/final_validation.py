import pandas as pd
import logging
import joblib
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def final_validation(test_path: str, model_path: str):
    logging.info("Starting Final Model Validation on Test Set...")
    
    test_df = pd.read_csv(test_path)
    
    leaky_cols = ['Students_Present', 'Attendance_Percentage', 'Total_Enrolled']
    non_features = ['Date', 'Attendance_Class']
    cols_to_drop = leaky_cols + non_features
    
    X_test = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns])
    y_test_raw = test_df['Attendance_Class']
    
    # Load model artifacts
    artifacts = joblib.load(model_path)
    pipeline = artifacts['pipeline']
    le = artifacts['label_encoder']
    
    y_test = le.transform(y_test_raw)
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    logging.info(f"Final Test Accuracy: {acc:.4f}")
    logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

if __name__ == "__main__":
    test_path = "../data/processed/test.csv"
    model_path = "../models/final_model.pkl"
    
    try:
        final_validation(test_path, model_path)
    except Exception as e:
        logging.error(f"Final validation failed: {e}")
