import os
import subprocess
import glob
import sys

base_dir = r"d:\coding\ML model\attendance_prediction\jupiter books"
notebooks = [
    r"phase 1 engineering and leakage prevention\01_data_cleaning\01_data_cleaning.ipynb",
    r"phase 1 engineering and leakage prevention\02_exploratory_data_analysis\02_exploratory_data_analysis.ipynb",
    r"phase 1 engineering and leakage prevention\03_feature_engineering_and_splitting\03_feature_engineering_and_splitting.ipynb",
    r"phase 2 model training\00_baseline_preprocessing_and_metrics\00_baseline_setup.ipynb",
    r"phase 2 model training\models\regression\01_linear_regression\01_linear_regression.ipynb",
    r"phase 2 model training\models\regression\02_decision_tree_regressor\02_decision_tree_regressor.ipynb",
    r"phase 2 model training\models\regression\03_random_forest_regressor\03_random_forest_regressor.ipynb",
    r"phase 2 model training\models\regression\04_gradient_boosting_regressor\04_gradient_boosting_regressor.ipynb",
    r"phase 2 model training\models\regression\05_xgboost_regressor\05_xgboost_regressor.ipynb",
    r"phase 2 model training\models\classification\06_logistic_regression\06_logistic_regression.ipynb",
    r"phase 2 model training\models\classification\07_decision_tree_classifier\07_decision_tree_classifier.ipynb",
    r"phase 2 model training\models\classification\08_random_forest_classifier\08_random_forest_classifier.ipynb",
    r"phase 2 model training\models\classification\09_svm_classifier\09_svm_classifier.ipynb",
    r"phase 2 model training\models\classification\10_knn_classifier\10_knn_classifier.ipynb",
    r"phase 2 model training\models\classification\11_naive_bayes\11_naive_bayes.ipynb",
    r"phase 2 model training\models\classification\12_xgboost_classifier\12_xgboost_classifier.ipynb",
    r"phase 2 model training\13_champion_selection_and_tuning\13_champion_selection_and_tuning.ipynb"
]

for rel_path in notebooks:
    full_path = os.path.join(base_dir, rel_path)
    print(f"Executing {full_path}...")
    cmd = [
        "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", full_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running {rel_path}!")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"SUCCESS: {rel_path}")

print("ALL NOTEBOOKS EXECUTED SUCCESSFULLY!")

print("Executing export_pipeline.py...")
export_script = os.path.join(base_dir, "..", "export_pipeline.py")
result = subprocess.run([sys.executable, export_script], capture_output=True, text=True)
if result.returncode != 0:
    print("ERROR running export_pipeline.py!")
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)
else:
    print("SUCCESS: export_pipeline.py")

print("PIPELINE FULLY COMPLETE!")
