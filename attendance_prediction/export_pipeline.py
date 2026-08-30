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

base_dir = r"d:\coding\ML model\attendance_prediction"
# If data isn't here, fallback to the other directory just in case
data_dir = os.path.join(base_dir, "data", "processed")
if not os.path.exists(data_dir):
    data_dir = r"d:\coding\ML model\classroom-attendance-prediction\data\processed"

export_dir = os.path.join(base_dir, "deployment_assets")
os.makedirs(export_dir, exist_ok=True)

# 1. Load the raw chronologically split training data to rebuild the scaler and weights
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))

# Columns are already one-hot encoded in train.csv

global_mean = train_df['Attendance_Percentage'].mean()

train_df['Attendance_Class'], attendance_bins = pd.qcut(train_df['Attendance_Percentage'], q=3, labels=['Low', 'Medium', 'High'], retbins=True)

cols_to_drop = [
    'Date', 'Start_Time', 'End_Time', 'Faculty_ID', 'Semester', 'Branch', 
    'Section', 'Classroom', 'Attendance_Percentage', 'Attendance_Class', 
    'Special_Event', 'Assignment_Due', 'Assignment_Due_Flag', 'Holiday_Before_After', 'Internal_Test_Week',
    'Students_Present'
]
y_train_raw = train_df['Attendance_Class']
X_train = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])

# Fill any NaNs with the global mean
X_train = X_train.fillna(global_mean)

# Also drop any remaining 'object' columns to be absolutely safe
X_train = X_train.select_dtypes(exclude=['object', 'string'])

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# Save column names to ensure Streamlit orders features correctly
feature_columns = X_train.columns.tolist()

# Dump metadata
joblib.dump(scaler, os.path.join(export_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(export_dir, 'label_encoder.pkl'))
joblib.dump(feature_columns, os.path.join(export_dir, 'feature_columns.pkl'))
joblib.dump(attendance_bins, os.path.join(export_dir, 'attendance_bins.pkl'))
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
    print(f"Training {name} (Classification)...")
    model.fit(X_train_scaled, y_train)
    filename = name.replace(" ", "_").replace("-", "").lower() + ".pkl"
    joblib.dump(model, os.path.join(export_dir, filename))
    print(f"Exported {filename}")

# 3. Train and Export all Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

y_train_reg = train_df['Attendance_Percentage']

reg_models = {
    "linear_regression_reg": LinearRegression(),
    "decision_tree_reg": DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42),
    "random_forest_reg": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "gradient_boosting_reg": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "xgboost_reg": XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
}

for name, model in reg_models.items():
    print(f"Training {name} (Regression)...")
    model.fit(X_train_scaled, y_train_reg)
    filename = name + ".pkl"
    joblib.dump(model, os.path.join(export_dir, filename))
    print(f"Exported {filename}")

print("Pipeline export complete. Ready for Streamlit deployment.")
