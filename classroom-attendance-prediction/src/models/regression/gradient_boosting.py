import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_regression_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, preprocessor = get_regression_data()
    model = GradientBoostingRegressor(random_state=42)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"[Gradient Boosting]")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R2 Score: {r2:.4f}")
