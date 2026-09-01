import os
import joblib
import pandas as pd
import numpy as np

# Load assets globally so they are cached in memory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "deployment_assets")

# Shared assets loaded once
scaler = joblib.load(os.path.join(ASSETS_DIR, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(ASSETS_DIR, 'label_encoder.pkl'))
feature_columns = joblib.load(os.path.join(ASSETS_DIR, 'feature_columns.pkl'))

# All available models mapped to their .pkl filenames
AVAILABLE_MODELS = {
    "Classification": {
        "Logistic Regression": "06_logistic_regression.pkl",
        "Decision Tree Classifier": "07_decision_tree_classifier.pkl",
        "Random Forest Classifier": "08_random_forest_classifier.pkl",
        "SVM Classifier": "09_svm_classifier.pkl",
        "k-NN Classifier": "10_knn_classifier.pkl",
        "Naive Bayes": "11_naive_bayes.pkl",
        "XGBoost Classifier": "12_xgboost_classifier.pkl",
    },
    "Regression": {
        "Linear Regression": "01_linear_regression.pkl",
        "Decision Tree Regressor": "02_decision_tree_regressor.pkl",
        "Random Forest Regressor": "03_random_forest_regressor.pkl",
        "Gradient Boosting Regressor": "04_gradient_boosting_regressor.pkl",
        "XGBoost Regressor": "05_xgboost_regressor.pkl",
    }
}

# Cache loaded models to avoid reloading on every prediction
_model_cache = {}


def _load_model(model_name):
    """Load and cache a model by its display name."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    for category in AVAILABLE_MODELS.values():
        if model_name in category:
            pkl_file = category[model_name]
            break
    else:
        raise ValueError(f"Unknown model: {model_name}")

    path = os.path.join(ASSETS_DIR, pkl_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {pkl_file}")

    loaded = joblib.load(path)
    _model_cache[model_name] = loaded
    return loaded


def _get_model_type(model_name):
    """Return 'Classification' or 'Regression' for a given model name."""
    for mtype, models in AVAILABLE_MODELS.items():
        if model_name in models:
            return mtype
    return None


def prepare_features(input_data):
    """
    Transforms the raw UI input dictionary into a scaled DataFrame
    matching the exact feature schema used during training.
    """
    df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Numerical / direct features

    date_val = pd.to_datetime(input_data.get('Date'))
    semester_start = pd.to_datetime('2026-04-10')

    df['Day_of_Semester'] = max((date_val - semester_start).days, 0)
    df['Week_Number'] = date_val.isocalendar().week
    df['Month'] = date_val.month

    # Time features
    start_time = input_data.get('Start_Time')
    try:
        hour = int(start_time.split(':')[0])
        if 'PM' in start_time.upper() and hour != 12:
            hour += 12
        if 'AM' in start_time.upper() and hour == 12:
            hour = 0
    except Exception:
        hour = 9

    df['Time_of_Day_Cluster_Morning'] = 1 if hour < 12 else 0
    df['Is_Post_Lunch_Class'] = 1 if 13 <= hour <= 14 else 0

    # Attendance momentum features
    prev_att = input_data.get('Previous_Lecture_Attendance_Pct', 0.0)
    df['Previous_Lecture_Attendance_Pct'] = prev_att
    df['Gap_Since_Previous_Lecture_Days'] = input_data.get('Gap_Since_Previous_Lecture_Days', 0)

    df['Rolling_Avg_3_Lectures'] = prev_att
    df['Monthly_Expanding_Mean'] = prev_att

    gap = df['Gap_Since_Previous_Lecture_Days'][0]
    df['Is_Consecutive'] = 1 if gap <= 2 else 0
    df['Consecutive_Lecture_Count'] = 1 if gap <= 2 else 0

    # Binary flags
    df['Is_Holiday_Adjacent'] = 1 if input_data.get('Holiday_Before_After') == 'Yes' else 0
    df['Week_Before_Exam_Flag'] = 1 if input_data.get('Internal_Test_Week') == 'Yes' else 0

    # One-hot encoded columns
    day_of_week = date_val.day_name()
    day_col = f'Day_of_Week_{day_of_week}'
    if day_col in df.columns:
        df[day_col] = 1

    subject = input_data.get('Subject')
    sub_col = f'Subject_{subject}'
    if sub_col in df.columns:
        df[sub_col] = 1

    weather = input_data.get('Weather')
    weather_col = f'Weather_{weather}'
    if weather_col in df.columns:
        df[weather_col] = 1

    prac_theory = input_data.get('Session_Type')
    pt_col = f'Practical_Theory_{prac_theory}'
    if pt_col in df.columns:
        df[pt_col] = 1

    # Scale and preserve feature names
    X_scaled = pd.DataFrame(scaler.transform(df), columns=feature_columns)
    return X_scaled


def predict_attendance(input_data, model_name="XGBoost Classifier"):
    """
    Returns the predicted attendance using the specified model.
    Handles both Classification and Regression models.
    """
    model = _load_model(model_name)
    X_scaled = prepare_features(input_data)
    model_type = _get_model_type(model_name)
    enrolled = input_data.get('Total_Enrolled', 60)

    if model_type == "Classification":
        pred_idx = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]
        category = label_encoder.inverse_transform([pred_idx])[0]

        confidence = probs[pred_idx]
        if category == 'High':
            estimated_pct = 76 + (confidence * 20)
        elif category == 'Medium':
            estimated_pct = 50 + (confidence * 25)
        else:
            estimated_pct = 20 + (confidence * 29)

        estimated_pct = min(max(estimated_pct, 0), 100)
        expected_students = int((estimated_pct / 100.0) * enrolled)

        return {
            'model_type': 'Classification',
            'category': category.upper(),
            'percentage': round(estimated_pct),
            'expected_students': expected_students,
            'probability': round(confidence * 100, 1),
            'all_probs': {
                label_encoder.inverse_transform([i])[0]: round(p * 100, 1)
                for i, p in enumerate(probs)
            }
        }

    else:  # Regression
        raw_prediction = model.predict(X_scaled)[0]
        estimated_pct = min(max(float(raw_prediction), 0), 100)
        expected_students = int((estimated_pct / 100.0) * enrolled)

        # Derive a category from the predicted percentage
        if estimated_pct >= 75:
            category = "HIGH"
        elif estimated_pct >= 50:
            category = "MEDIUM"
        else:
            category = "LOW"

        return {
            'model_type': 'Regression',
            'category': category,
            'percentage': round(estimated_pct, 1),
            'expected_students': expected_students,
            'raw_value': round(float(raw_prediction), 2),
        }
