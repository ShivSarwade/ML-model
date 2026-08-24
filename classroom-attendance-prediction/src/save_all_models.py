import os
import joblib
import logging
from sklearn.pipeline import Pipeline
from models.utils import get_classification_data, get_regression_data

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def save_all_models():
    logging.info("Training and aggregating all models for the dashboard...")
    
    all_models = {
        'Classification': {},
        'Regression': {},
        'label_encoder': None
    }
    
    # Classification
    X_train_c, y_train_c, _, _, prep_c, le = get_classification_data()
    all_models['label_encoder'] = le
    
    class_algs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost Classifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    for name, model in class_algs.items():
        pipe = Pipeline(steps=[('preprocessor', prep_c), ('classifier', model)])
        pipe.fit(X_train_c, y_train_c)
        all_models['Classification'][name] = pipe
        
    # Regression
    X_train_r, y_train_r, _, _, prep_r = get_regression_data()
    
    reg_algs = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost Regressor': XGBRegressor(random_state=42)
    }
    
    for name, model in reg_algs.items():
        pipe = Pipeline(steps=[('preprocessor', prep_r), ('regressor', model)])
        pipe.fit(X_train_r, y_train_r)
        all_models['Regression'][name] = pipe
        
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'all_models.pkl')
    joblib.dump(all_models, save_path)
    logging.info(f"Successfully saved 12 models to {save_path}")

if __name__ == "__main__":
    save_all_models()
