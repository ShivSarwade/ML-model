import os
import subprocess
import sys
import shutil

project_dir = r"d:\coding\ML model\attendance_prediction"

def clean_old_files():
    print("0. Cleaning old evidences and assets...")
    import glob
    
    # Delete evidence and checkpoint folders globally
    dirs_to_delete = glob.glob(r'd:\coding\ML model\**\evidence', recursive=True) + \
                     glob.glob(r'd:\coding\ML model\**\.ipynb_checkpoints', recursive=True)
                     
    for d in dirs_to_delete:
        shutil.rmtree(d, ignore_errors=True)
            
    # Delete deployment assets
    assets_dir = os.path.join(project_dir, "deployment_assets")
    if os.path.exists(assets_dir):
        shutil.rmtree(assets_dir, ignore_errors=True)
        
    # Delete processed data
    processed_dir = os.path.join(project_dir, "data", "processed")
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir, ignore_errors=True)

clean_old_files()

venv_dir = os.path.join(project_dir, "venv")
req_file = os.path.join(project_dir, "requirements.txt")

print("1. Virtual Environment Setup...")
if not os.path.exists(venv_dir):
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
jupyter_exe = os.path.join(venv_dir, "Scripts", "jupyter.exe")

subprocess.run([pip_exe, "install", "-r", req_file], check=True)
subprocess.run([pip_exe, "install", "nbconvert", "ipykernel"], check=True)

notebooks_in_order = [
    r"jupiter books\phase 1 engineering and leakage prevention\01_data_cleaning\01_data_cleaning.ipynb",
    r"jupiter books\phase 1 engineering and leakage prevention\02_exploratory_data_analysis\02_exploratory_data_analysis.ipynb",
    r"jupiter books\phase 1 engineering and leakage prevention\03_feature_engineering_and_splitting\03_feature_engineering_and_splitting.ipynb",
    r"jupiter books\phase 2 model training\00_baseline_preprocessing_and_metrics\00_baseline_setup.ipynb",
    r"jupiter books\phase 2 model training\models\regression\01_linear_regression\01_linear_regression.ipynb",
    r"jupiter books\phase 2 model training\models\regression\02_decision_tree_regressor\02_decision_tree_regressor.ipynb",
    r"jupiter books\phase 2 model training\models\regression\03_random_forest_regressor\03_random_forest_regressor.ipynb",
    r"jupiter books\phase 2 model training\models\regression\04_gradient_boosting_regressor\04_gradient_boosting_regressor.ipynb",
    r"jupiter books\phase 2 model training\models\regression\05_xgboost_regressor\05_xgboost_regressor.ipynb",
    r"jupiter books\phase 2 model training\models\classification\06_logistic_regression\06_logistic_regression.ipynb",
    r"jupiter books\phase 2 model training\models\classification\07_decision_tree_classifier\07_decision_tree_classifier.ipynb",
    r"jupiter books\phase 2 model training\models\classification\08_random_forest_classifier\08_random_forest_classifier.ipynb",
    r"jupiter books\phase 2 model training\models\classification\09_svm_classifier\09_svm_classifier.ipynb",
    r"jupiter books\phase 2 model training\models\classification\10_knn_classifier\10_knn_classifier.ipynb",
    r"jupiter books\phase 2 model training\models\classification\11_naive_bayes\11_naive_bayes.ipynb",
    r"jupiter books\phase 2 model training\models\classification\12_xgboost_classifier\12_xgboost_classifier.ipynb",
    r"jupiter books\phase 2 model training\13_champion_selection_and_tuning\13_champion_selection_and_tuning.ipynb"
]

print("2. Executing ALL Notebooks (The notebooks now handle everything natively!)...")
for nb_rel_path in notebooks_in_order:
    nb_full_path = os.path.join(project_dir, nb_rel_path)
    if not os.path.exists(nb_full_path):
        continue
    subprocess.run([jupyter_exe, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_full_path], check=True, cwd=project_dir)

print("Pipeline Finished Successfully! Every single .pkl comes from a notebook now.")
