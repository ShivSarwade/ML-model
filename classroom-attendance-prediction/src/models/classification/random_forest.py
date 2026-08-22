import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_classification_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, preprocessor, le = get_classification_data()
    model = RandomForestClassifier(random_state=42)
    
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
    
    print(f"[Random Forest]")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Precision: {prec:.4f}")
    print(f"Validation Recall: {rec:.4f}")
    print(f"Validation ROC-AUC: {roc:.4f}")
