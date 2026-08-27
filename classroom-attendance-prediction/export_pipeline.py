import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

print("Starting Pipeline Export...")

base_dir = r"d:\coding\ML model\classroom-attendance-prediction"
data_dir = os.path.join(base_dir, "data", "processed")
export_dir = os.path.join(base_dir, "deployment_assets")
os.makedirs(export_dir, exist_ok=True)

# 1. Load the raw chronologically split training data to rebuild the scaler and weights
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))

# One-hot encode categorical variables for both
train_df = pd.get_dummies(train_df, columns=['Day_of_Week', 'Subject', 'Weather', 'Time_of_Day', 'Practical_Theory'])
val_df = pd.get_dummies(val_df, columns=['Day_of_Week', 'Subject', 'Weather', 'Time_of_Day', 'Practical_Theory'])

# Align columns
val_df = val_df.reindex(columns=train_df.columns, fill_value=0)

global_mean = train_df['Attendance_Percentage'].mean()

def calculate_momentum(df):
    return (0.40 * df['Monthly_Avg_Attendance']) + (0.60 * df['Rolling_Avg_3'])

train_df['Base_Momentum'] = calculate_momentum(train_df)
binary_cols = ['Is_Post_Lunch_Class', 'Is_Holiday_Adjacent', 'Week_Before_Exam'] + [c for c in train_df.columns if c.startswith(('Weather_', 'Subject_', 'Day_of_Week_', 'Practical_Theory_', 'Time_of_Day_'))]

affection_rates = {}
for col in binary_cols:
    if col in train_df.columns:
        mean_when_present = train_df[train_df[col] == 1]['Attendance_Percentage'].mean()
        weight = mean_when_present - global_mean
        affection_rates[col] = weight if not pd.isna(weight) else 0.0

def calculate_expected_and_residual(df, rates):
    expected = calculate_momentum(df)
    for col in rates.keys():
        if col in df.columns:
            expected += (df[col] * rates[col])
    expected = expected.clip(0, 100)
    residual = df['Attendance_Percentage'] - expected
    return expected, residual

train_df['Expected_Attendance'], train_df['Residual'] = calculate_expected_and_residual(train_df, affection_rates)
train_df['Attendance_Class'], residual_bins = pd.qcut(train_df['Residual'], q=3, labels=['Low', 'Medium', 'High'], retbins=True)

targets_to_drop = ['Attendance_Percentage', 'Attendance_Class', 'Expected_Attendance', 'Residual', 'Base_Momentum']
y_train_raw = train_df['Attendance_Class']
X_train = train_df.drop(columns=[col for col in targets_to_drop if col in train_df.columns])

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# Save column names to ensure Streamlit orders features correctly
feature_columns = X_train.columns.tolist()

# Dump metadata
joblib.dump(scaler, os.path.join(export_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(export_dir, 'label_encoder.pkl'))
joblib.dump(affection_rates, os.path.join(export_dir, 'affection_rates.pkl'))
joblib.dump(feature_columns, os.path.join(export_dir, 'feature_columns.pkl'))
joblib.dump(residual_bins, os.path.join(export_dir, 'residual_bins.pkl'))
joblib.dump(global_mean, os.path.join(export_dir, 'global_mean.pkl'))

print("Metadata, Scaler, and Weights exported.")

# 2. Train and Export all Classification Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='mlogloss', n_jobs=-1)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    filename = name.replace(" ", "_").replace("-", "").lower() + ".pkl"
    joblib.dump(model, os.path.join(export_dir, filename))
    print(f"Exported {filename}")

print("✅ Pipeline export complete. Ready for Streamlit deployment.")
