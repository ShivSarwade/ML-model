import os

class_template = """import sys
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_classification_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
{imports}

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, preprocessor, le = get_classification_data()
    model = {model_init}
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    
    try:
        y_prob = pipeline.predict_proba(X_val)
        roc = roc_auc_score(y_val, y_prob, multi_class='ovr')
    except (AttributeError, ValueError):
        roc = 0.0
    
    print(f"[{model_name}]")
    print(f"Validation Accuracy: {{acc:.4f}}")
    print(f"Validation F1 Score: {{f1:.4f}}")
    print(f"Validation Precision: {{prec:.4f}}")
    print(f"Validation Recall: {{rec:.4f}}")
    print(f"Validation ROC-AUC: {{roc:.4f}}")
    
    artifacts = {{
        'pipeline': pipeline,
        'label_encoder': le
    }}
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', f"{{'{model_name}'.replace(' ', '_').lower()}}_class.pkl")
    joblib.dump(artifacts, save_path)
"""

reg_template = """import sys
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_regression_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
{imports}

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, preprocessor = get_regression_data()
    model = {model_init}
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"[{model_name}]")
    print(f"Validation MAE: {{mae:.4f}}")
    print(f"Validation R2 Score: {{r2:.4f}}")
    
    artifacts = {{
        'pipeline': pipeline
    }}
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', f"{{'{model_name}'.replace(' ', '_').lower()}}_reg.pkl")
    joblib.dump(artifacts, save_path)
"""

class_models = [
    ("logistic_regression", "Logistic Regression", "from sklearn.linear_model import LogisticRegression", "LogisticRegression(max_iter=1000, random_state=42)"),
    ("decision_tree", "Decision Tree", "from sklearn.tree import DecisionTreeClassifier", "DecisionTreeClassifier(random_state=42)"),
    ("random_forest", "Random Forest", "from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier(random_state=42)"),
    ("svm", "SVM", "from sklearn.svm import SVC", "SVC(random_state=42, probability=True)"),
    ("knn", "KNN", "from sklearn.neighbors import KNeighborsClassifier", "KNeighborsClassifier()"),
    ("naive_bayes", "Naive Bayes", "from sklearn.naive_bayes import GaussianNB", "GaussianNB()"),
    ("xgboost_classifier", "XGBoost Classifier", "from xgboost import XGBClassifier", "XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')")
]

reg_models = [
    ("linear_regression", "Linear Regression", "from sklearn.linear_model import LinearRegression", "LinearRegression()"),
    ("decision_tree_regressor", "Decision Tree Regressor", "from sklearn.tree import DecisionTreeRegressor", "DecisionTreeRegressor(random_state=42)"),
    ("random_forest_regressor", "Random Forest Regressor", "from sklearn.ensemble import RandomForestRegressor", "RandomForestRegressor(random_state=42)"),
    ("gradient_boosting", "Gradient Boosting", "from sklearn.ensemble import GradientBoostingRegressor", "GradientBoostingRegressor(random_state=42)"),
    ("xgboost_regressor", "XGBoost Regressor", "from xgboost import XGBRegressor", "XGBRegressor(random_state=42)")
]

for fname, mname, imp, init in class_models:
    with open(f"classification/{fname}.py", "w", encoding="utf-8") as f:
        f.write(class_template.format(imports=imp, model_init=init, model_name=mname))

for fname, mname, imp, init in reg_models:
    with open(f"regression/{fname}.py", "w", encoding="utf-8") as f:
        f.write(reg_template.format(imports=imp, model_init=init, model_name=mname))

print("All scripts regenerated with joblib.dump logic.")
