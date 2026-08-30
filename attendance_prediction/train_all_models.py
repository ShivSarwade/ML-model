import pandas as pd
import numpy as np
import os
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("Loading preprocessed datasets...")
data_dir = r"d:\coding\ML model\attendance_prediction\data\processed"

X_train = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
X_val = pd.read_csv(os.path.join(data_dir, 'X_val_scaled.csv'))

y_train_reg = pd.read_csv(os.path.join(data_dir, 'y_train_reg.csv')).squeeze()
y_val_reg = pd.read_csv(os.path.join(data_dir, 'y_val_reg.csv')).squeeze()

y_train_class_raw = pd.read_csv(os.path.join(data_dir, 'y_train_class.csv')).squeeze()
y_val_class_raw = pd.read_csv(os.path.join(data_dir, 'y_val_class.csv')).squeeze()

le = LabelEncoder()
y_train_class = le.fit_transform(y_train_class_raw)
y_val_class = le.transform(y_val_class_raw)

results = []

def log_result(model_id, model_type, val_mae=np.nan, val_rmse=np.nan, val_mape=np.nan, val_r2=np.nan, val_acc=np.nan, val_f1=np.nan, val_roc=np.nan):
    results.append({
        'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Model_ID': model_id,
        'Model_Type': model_type,
        'Val_MAE': val_mae,
        'Val_RMSE': val_rmse,
        'Val_MAPE': val_mape,
        'Val_R2': val_r2,
        'Val_Accuracy': val_acc,
        'Val_F1': val_f1,
        'Val_ROCAUC': val_roc
    })

print("Training Regression Models...")
# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train_reg)
y_pred = lr.predict(X_val)
log_result("01_linear_regression", "Regression", mean_absolute_error(y_val_reg, y_pred), np.sqrt(mean_squared_error(y_val_reg, y_pred)), mean_absolute_percentage_error(y_val_reg, y_pred), r2_score(y_val_reg, y_pred))

# 2. Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
dt.fit(X_train, y_train_reg)
y_pred = dt.predict(X_val)
log_result("02_decision_tree_regressor", "Regression", mean_absolute_error(y_val_reg, y_pred), np.sqrt(mean_squared_error(y_val_reg, y_pred)), mean_absolute_percentage_error(y_val_reg, y_pred), r2_score(y_val_reg, y_pred))

# 3. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_reg)
y_pred = rf.predict(X_val)
log_result("03_random_forest_regressor", "Regression", mean_absolute_error(y_val_reg, y_pred), np.sqrt(mean_squared_error(y_val_reg, y_pred)), mean_absolute_percentage_error(y_val_reg, y_pred), r2_score(y_val_reg, y_pred))

# 4. Gradient Boosting Regressor
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train_reg)
y_pred = gb.predict(X_val)
log_result("04_gradient_boosting_regressor", "Regression", mean_absolute_error(y_val_reg, y_pred), np.sqrt(mean_squared_error(y_val_reg, y_pred)), mean_absolute_percentage_error(y_val_reg, y_pred), r2_score(y_val_reg, y_pred))

# 5. XGBoost Regressor
xgb_r = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)
xgb_r.fit(X_train, y_train_reg)
y_pred = xgb_r.predict(X_val)
log_result("05_xgboost_regressor", "Regression", mean_absolute_error(y_val_reg, y_pred), np.sqrt(mean_squared_error(y_val_reg, y_pred)), mean_absolute_percentage_error(y_val_reg, y_pred), r2_score(y_val_reg, y_pred))

print("Training Classification Models...")
# 6. Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train_class)
y_pred = log_reg.predict(X_val)
y_prob = log_reg.predict_proba(X_val)
log_result("06_logistic_regression", "Classification", val_acc=accuracy_score(y_val_class, y_pred), val_f1=f1_score(y_val_class, y_pred, average='macro'), val_roc=roc_auc_score(y_val_class, y_prob, multi_class='ovr'))

# 7. Decision Tree Classifier
dt_c = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_c.fit(X_train, y_train_class)
y_pred = dt_c.predict(X_val)
y_prob = dt_c.predict_proba(X_val)
log_result("07_decision_tree_classifier", "Classification", val_acc=accuracy_score(y_val_class, y_pred), val_f1=f1_score(y_val_class, y_pred, average='macro'), val_roc=roc_auc_score(y_val_class, y_prob, multi_class='ovr'))

# 8. Random Forest Classifier
rf_c = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_c.fit(X_train, y_train_class)
y_pred = rf_c.predict(X_val)
y_prob = rf_c.predict_proba(X_val)
log_result("08_random_forest_classifier", "Classification", val_acc=accuracy_score(y_val_class, y_pred), val_f1=f1_score(y_val_class, y_pred, average='macro'), val_roc=roc_auc_score(y_val_class, y_prob, multi_class='ovr'))

# 9. SVM
svm_c = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_c.fit(X_train, y_train_class)
y_pred = svm_c.predict(X_val)
y_prob = svm_c.predict_proba(X_val)
log_result("09_svm_classifier", "Classification", val_acc=accuracy_score(y_val_class, y_pred), val_f1=f1_score(y_val_class, y_pred, average='macro'), val_roc=roc_auc_score(y_val_class, y_prob, multi_class='ovr'))

# 10. k-NN Classifier
knn_c = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_c.fit(X_train, y_train_class)
y_pred = knn_c.predict(X_val)
y_prob = knn_c.predict_proba(X_val)
log_result("10_knn_classifier", "Classification", val_acc=accuracy_score(y_val_class, y_pred), val_f1=f1_score(y_val_class, y_pred, average='macro'), val_roc=roc_auc_score(y_val_class, y_prob, multi_class='ovr'))

# 11. Naive Bayes
nb_c = GaussianNB()
nb_c.fit(X_train, y_train_class)
y_pred = nb_c.predict(X_val)
y_prob = nb_c.predict_proba(X_val)
log_result("11_naive_bayes", "Classification", val_acc=accuracy_score(y_val_class, y_pred), val_f1=f1_score(y_val_class, y_pred, average='macro'), val_roc=roc_auc_score(y_val_class, y_prob, multi_class='ovr'))

# 12. XGBoost Classifier
xgb_c = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
xgb_c.fit(X_train, y_train_class)
y_pred = xgb_c.predict(X_val)
y_prob = xgb_c.predict_proba(X_val)
log_result("12_xgboost_classifier", "Classification", val_acc=accuracy_score(y_val_class, y_pred), val_f1=f1_score(y_val_class, y_pred, average='macro'), val_roc=roc_auc_score(y_val_class, y_prob, multi_class='ovr'))

print("Saving results to exp.csv...")
results_df = pd.DataFrame(results)
exp_file = os.path.join(data_dir, 'exp.csv')
results_df.to_csv(exp_file, index=False)

print(f"Results successfully saved to {exp_file}")
