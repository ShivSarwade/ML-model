import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime

st.set_page_config(page_title="Attendance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Classroom Attendance Predictor")
st.markdown("Predict the expected attendance risk (High, Medium, Low) for an upcoming lecture using Machine Learning.")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "deployment_assets")
    
    scaler = joblib.load(os.path.join(assets_dir, 'scaler.pkl'))
    le = joblib.load(os.path.join(assets_dir, 'label_encoder.pkl'))
    rates = joblib.load(os.path.join(assets_dir, 'affection_rates.pkl'))
    cols = joblib.load(os.path.join(assets_dir, 'feature_columns.pkl'))
    bins = joblib.load(os.path.join(assets_dir, 'residual_bins.pkl'))
    g_mean = joblib.load(os.path.join(assets_dir, 'global_mean.pkl'))
    
    # Load Models
    models = {
        "XGBoost (Champion)": joblib.load(os.path.join(assets_dir, 'xgboost.pkl')),
        "Random Forest": joblib.load(os.path.join(assets_dir, 'random_forest.pkl')),
        "Decision Tree": joblib.load(os.path.join(assets_dir, 'decision_tree.pkl')),
        "Support Vector Machine": joblib.load(os.path.join(assets_dir, 'svm.pkl')),
        "k-Nearest Neighbors": joblib.load(os.path.join(assets_dir, 'knn.pkl')),
        "Naive Bayes": joblib.load(os.path.join(assets_dir, 'naive_bayes.pkl')),
        "Logistic Regression": joblib.load(os.path.join(assets_dir, 'logistic_regression.pkl'))
    }
    return scaler, le, rates, cols, bins, g_mean, models

try:
    scaler, le, affection_rates, feature_cols, residual_bins, global_mean, models = load_assets()
except Exception as e:
    st.error(f"Error loading backend models: {e}. Please ensure you ran the export_pipeline.py script.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("📝 Lecture Details")

# 1. Date
sel_date = st.sidebar.date_input("Select Date", datetime.date.today())
# Calculate derived date features
day_name = sel_date.strftime("%A")
month_name = sel_date.strftime("%B")
# Simple holiday adjacency logic (weekends)
is_holiday_adj = 1 if day_name in ["Monday", "Friday"] else 0

# 2. Time
time_slots = [
    "8.30 AM - 9.15 AM", "9.15 AM - 10.00 AM", "10.00 AM - 10.45 AM",
    "11.00 AM - 11.45 AM", "11.45 AM - 12.30 PM", "12.30 PM - 1.15 PM",
    "1.30 PM - 2.15 PM", "2.15 PM - 3.00 PM", "3.00 PM - 3.45 PM",
    "4.00 PM - 4.45 PM", "4.45 PM - 5.30 PM", "5.30 PM - 6.15 PM"
]
sel_time = st.sidebar.selectbox("Lecture Timing", time_slots)

# Derived time logic (simplified for dashboard)
lecture_num = time_slots.index(sel_time) + 1
if lecture_num <= 3:
    time_cluster = "Morning"
elif lecture_num <= 8:
    time_cluster = "Afternoon"
else:
    time_cluster = "Late Evening"
is_post_lunch = 1 if lecture_num in [7,8] else 0

# 3. Subject
subject_list = [
    "Mobile Application Development", "Internet of Things", "Digital Marketing",
    "Block Chain", "Project Phase- I", "Java Programming", "Information Security",
    "Data Science", "Agile Methodology", "Machine Learning"
]
sel_subject = st.sidebar.selectbox("Subject", subject_list)

# 4. Theory/Practical (Conditional)
subjects_with_practical = ["Mobile Application Development", "Internet of Things", "Java Programming", "Data Science", "Machine Learning"]
is_practical = 0
if sel_subject in subjects_with_practical:
    ttype = st.sidebar.radio("Class Type", ["Theory", "Practical"])
    is_practical = 1 if ttype == "Practical" else 0

# 5. Model Selector
st.sidebar.divider()
st.sidebar.header("🤖 Engine Settings")
selected_model_name = st.sidebar.selectbox("Prediction Algorithm", list(models.keys()))

st.sidebar.info(f"💡 Day ({day_name}) and Time Cluster ({time_cluster}) were calculated automatically!")

# --- Backend Processing ---
# Construct the input row exactly as the scaler expects
input_data = pd.DataFrame(columns=feature_cols)
input_data.loc[0] = 0 # Initialize with 0s

# Populate known fields
if 'Lecture_Number' in input_data.columns: input_data['Lecture_Number'] = lecture_num
if 'Is_Post_Lunch_Class' in input_data.columns: input_data['Is_Post_Lunch_Class'] = is_post_lunch
if 'Is_Holiday_Adjacent' in input_data.columns: input_data['Is_Holiday_Adjacent'] = is_holiday_adj
if 'Week_Before_Exam_Flag' in input_data.columns: input_data['Week_Before_Exam_Flag'] = 0 # Assume False for normal prediction

# Populate dummy variables (Weather, Subject, Day, etc.)
# We will assume "Clear" weather for a basic prediction
if f'Weather_Clear' in input_data.columns: input_data[f'Weather_Clear'] = 1
if f'Day_of_Week_{day_name}' in input_data.columns: input_data[f'Day_of_Week_{day_name}'] = 1
if f'Subject_{sel_subject}' in input_data.columns: input_data[f'Subject_{sel_subject}'] = 1
if f'Time_of_Day_Cluster_{time_cluster}' in input_data.columns: input_data[f'Time_of_Day_Cluster_{time_cluster}'] = 1
if is_practical and 'Practical_Theory_Practical' in input_data.columns: input_data['Practical_Theory_Practical'] = 1

# Calculate Base Momentum and Expected Attendance
input_data['Monthly_Expanding_Mean'] = global_mean # Assume global average for new prediction
input_data['Rolling_Avg_3_Lectures'] = global_mean

base_momentum = (0.40 * global_mean) + (0.60 * global_mean)

# Calculate expected attendance by adding affection rates
expected = base_momentum
for col, weight in affection_rates.items():
    if col in input_data.columns and input_data[col].iloc[0] == 1:
        expected += weight
expected = min(max(expected, 0), 100) # Clip 0-100

# Scale
input_scaled = scaler.transform(input_data)

# --- Main UI ---
tab1, tab2 = st.tabs(["🎯 Prediction Engine", "📊 Model Evaluation Matrix"])

with tab1:
    st.subheader(f"Predicting with: {selected_model_name}")
    
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        model = models[selected_model_name]
        
        # Some models return probability if requested, but predict() returns class directly
        pred_encoded = model.predict(input_scaled)
        pred_class = le.inverse_transform(pred_encoded)[0]
        
        # Color coding
        if pred_class == "High":
            st.success("🟢 **HIGH ATTENDANCE EXPECTED** (Overperformance Band)")
        elif pred_class == "Medium":
            st.warning("🟡 **MEDIUM ATTENDANCE EXPECTED** (As Expected Band)")
        else:
            st.error("🔴 **LOW ATTENDANCE EXPECTED** (Underperformance Band)")
            
        st.write("---")
        st.write("### 🧠 Backend Calculations")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Momentum", f"{base_momentum:.2f}%")
        with col2:
            st.metric("Expected Attendance (With Weights)", f"{expected:.2f}%")
            
        st.write("**Active Affection Weights Applied (Learned from Phase 1):**")
        active_weights = {k: round(v, 2) for k, v in affection_rates.items() if k in input_data.columns and input_data[k].iloc[0] == 1}
        st.json(active_weights)

with tab2:
    st.subheader("Automated Experiment Tracking Matrix")
    st.markdown("This table contains the validation scores from our rigorous Phase 3 training block. It fulfills the project deliverable requirement for an Experiment Matrix.")
    
    # Check both directories just in case
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_file = os.path.join(base_dir, "data", "processed", "experiment_results.csv")
    if not os.path.exists(exp_file):
        exp_file = r"d:\coding\ML model\classroom-attendance-prediction\data\processed\experiment_results.csv"
        
    if os.path.exists(exp_file):
        df_exp = pd.read_csv(exp_file)
        # Drop NaN columns to make it cleaner for the dashboard (like Regression metrics for Classification models)
        df_clean = df_exp.dropna(axis=1, how='all')
        st.dataframe(df_clean, use_container_width=True)
    else:
        st.info("Experiment results not found. Please run the training notebooks.")
