import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(BASE_DIR, "data", "processed", "train.csv")
DEFAULT_VAL = os.path.join(BASE_DIR, "data", "processed", "val.csv")

def get_classification_data(train_path=DEFAULT_TRAIN, val_path=DEFAULT_VAL):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    leaky_cols = ['Students_Present', 'Attendance_Percentage', 'Total_Enrolled']
    non_features = ['Date', 'Attendance_Class']
    cols_to_drop = leaky_cols + non_features
    
    X_train = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])
    y_train_raw = train_df['Attendance_Class']
    
    X_val = val_df.drop(columns=[col for col in cols_to_drop if col in val_df.columns])
    y_val_raw = val_df['Attendance_Class']
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    return X_train, y_train, X_val, y_val, preprocessor, le

def get_regression_data(train_path=DEFAULT_TRAIN, val_path=DEFAULT_VAL):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    target = 'Attendance_Percentage'
    
    leaky_cols = ['Students_Present', 'Attendance_Class', 'Total_Enrolled']
    non_features = ['Date']
    cols_to_drop = leaky_cols + non_features + [target]
    
    X_train = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])
    y_train = train_df[target]
    
    X_val = val_df.drop(columns=[col for col in cols_to_drop if col in val_df.columns])
    y_val = val_df[target]
    
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    return X_train, y_train, X_val, y_val, preprocessor
