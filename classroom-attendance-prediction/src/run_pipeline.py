import os
import subprocess
import sys
import glob

def run_script(script_path, description):
    print(f"\n{'='*60}")
    print(f"Executing: {description}")
    print(f"File: {script_path}")
    print(f"{'='*60}")
    
    try:
        # Use sys.executable to ensure we use the same Python interpreter
        result = subprocess.run([sys.executable, script_path], check=True, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout.strip())
        print(f"\n[SUCCESS] {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to execute {script_path}.")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        sys.exit(1)

def run_models_in_directory(directory_path, task_name):
    print(f"\n{'#'*60}")
    print(f"Executing {task_name} Models")
    print(f"{'#'*60}")
    
    # Get all python files in the directory
    pattern = os.path.join(directory_path, "*.py")
    model_scripts = glob.glob(pattern)
    
    for script in model_scripts:
        basename = os.path.basename(script)
        if basename == "__init__.py":
            continue
            
        print(f"\n--- Running {basename} ---")
        try:
            result = subprocess.run([sys.executable, script], check=True, text=True, capture_output=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {basename} failed.")
            if e.stderr:
                print(e.stderr)
            sys.exit(1)

def main():
    print("Starting End-to-End Classroom Attendance Prediction Pipeline")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Phase 1: Data Processing
    run_script(os.path.join(base_dir, "data_processing.py"), "Phase 1: Data Cleaning & Validation")
    
    # Phase 2: Exploratory Data Analysis
    run_script(os.path.join(base_dir, "eda.py"), "Phase 2: Exploratory Data Analysis (EDA)")
    
    # Phase 3: Feature Engineering
    run_script(os.path.join(base_dir, "feature_engineering.py"), "Phase 3: Feature Engineering & Data Splitting")
    
    # Phase 4: Model Training (Classification)
    class_dir = os.path.join(base_dir, "models", "classification")
    run_models_in_directory(class_dir, "Classification")
    
    # Phase 4: Model Training (Regression)
    reg_dir = os.path.join(base_dir, "models", "regression")
    run_models_in_directory(reg_dir, "Regression")
    
    # Phase 5: Final Evaluation on Test Set
    run_script(os.path.join(base_dir, "final_validation.py"), "Phase 5: Final Validation on Test Set")
    
    # Phase 6: Aggregate and Save All Models for Dashboard
    run_script(os.path.join(base_dir, "save_all_models.py"), "Phase 6: Saving All Models for Dashboard Inference")
    
    print(f"\n{'*'*60}")
    print("[SUCCESS] PIPELINE EXECUTION COMPLETE!")
    print("All models successfully trained and evaluated.")
    print("The final dashboard is ready to launch in the 'dashboard' folder.")
    print(f"{'*'*60}\n")

if __name__ == "__main__":
    main()
