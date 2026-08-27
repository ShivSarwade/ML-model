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
        "XGBoost Classifier": joblib.load(os.path.join(assets_dir, 'xgboost.pkl')),
        "Random Forest Classifier": joblib.load(os.path.join(assets_dir, 'random_forest.pkl')),
        "Decision Tree Classifier": joblib.load(os.path.join(assets_dir, 'decision_tree.pkl')),
        "Support Vector Machine": joblib.load(os.path.join(assets_dir, 'svm.pkl')),
        "k-Nearest Neighbors": joblib.load(os.path.join(assets_dir, 'knn.pkl')),
        "Naive Bayes": joblib.load(os.path.join(assets_dir, 'naive_bayes.pkl')),
        "Logistic Regression": joblib.load(os.path.join(assets_dir, 'logistic_regression.pkl'))
    }
    
    reg_models = {
        "XGBoost Regressor": joblib.load(os.path.join(assets_dir, 'xgboost_regressor_reg.pkl')),
        "Random Forest Regressor": joblib.load(os.path.join(assets_dir, 'random_forest_regressor_reg.pkl')),
        "Decision Tree Regressor": joblib.load(os.path.join(assets_dir, 'decision_tree_regressor_reg.pkl')),
        "Gradient Boosting Regressor": joblib.load(os.path.join(assets_dir, 'gradient_boosting_regressor_reg.pkl')),
        "Linear Regression": joblib.load(os.path.join(assets_dir, 'linear_regression_reg.pkl'))
    }
    return scaler, le, rates, cols, bins, g_mean, models, reg_models

try:
    scaler, le, affection_rates, feature_cols, residual_bins, global_mean, models, reg_models = load_assets()
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

# 5. Advanced Metrics
st.sidebar.divider()
st.sidebar.header("🔬 Raw Constraints (Simulation)")
sel_weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Cloudy", "Rainy"])
is_exam_prox = st.sidebar.checkbox("Is it the week before an exam?", value=False)
prev_pct = st.sidebar.slider("Previous Lecture Attendance (%)", 0.0, 100.0, float(global_mean))
rolling_avg_3 = st.sidebar.slider("Last 3 Lectures Average (%)", 0.0, 100.0, float(global_mean))
gap_days = st.sidebar.slider("Gap Since Previous Lecture (Days)", 1, 14, 2)

# 6. Model Selector
st.sidebar.divider()
st.sidebar.header("🤖 Engine Settings")
selected_cls_model = st.sidebar.selectbox("Classification Engine (Band)", list(models.keys()), index=0)
selected_reg_model = st.sidebar.selectbox("Regression Engine (Headcount)", list(reg_models.keys()), index=0)

st.sidebar.info(f"💡 Day ({day_name}) and Time Cluster ({time_cluster}) were calculated automatically!")

# --- Backend Processing ---
# Construct the input row exactly as the scaler expects
input_data = pd.DataFrame(columns=feature_cols)
input_data.loc[0] = 0 # Initialize with 0s

# Populate known fields
if 'Lecture_Number' in input_data.columns: input_data['Lecture_Number'] = lecture_num
if 'Is_Post_Lunch_Class' in input_data.columns: input_data['Is_Post_Lunch_Class'] = is_post_lunch
if 'Is_Holiday_Adjacent' in input_data.columns: input_data['Is_Holiday_Adjacent'] = is_holiday_adj
if 'Month' in input_data.columns: input_data['Month'] = sel_date.month
if 'Week_Number' in input_data.columns: input_data['Week_Number'] = sel_date.isocalendar()[1]
if 'Day_of_Semester' in input_data.columns: input_data['Day_of_Semester'] = 45 # Assuming middle of semester
if 'Week_Before_Exam_Flag' in input_data.columns: input_data['Week_Before_Exam_Flag'] = 1 if is_exam_prox else 0
if 'Previous_Lecture_Attendance_Pct' in input_data.columns: input_data['Previous_Lecture_Attendance_Pct'] = prev_pct
if 'Gap_Since_Previous_Lecture_Days' in input_data.columns: input_data['Gap_Since_Previous_Lecture_Days'] = gap_days

# Populate dummy variables (Weather, Subject, Day, etc.)
if f'Weather_{sel_weather}' in input_data.columns: input_data[f'Weather_{sel_weather}'] = 1
if f'Day_of_Week_{day_name}' in input_data.columns: input_data[f'Day_of_Week_{day_name}'] = 1
if f'Subject_{sel_subject}' in input_data.columns: input_data[f'Subject_{sel_subject}'] = 1
if f'Time_of_Day_Cluster_{time_cluster}' in input_data.columns: input_data[f'Time_of_Day_Cluster_{time_cluster}'] = 1
if is_practical and 'Practical_Theory_Practical' in input_data.columns: input_data['Practical_Theory_Practical'] = 1

# Calculate Base Momentum and Expected Attendance
input_data['Monthly_Expanding_Mean'] = global_mean # Assume global average for new prediction
input_data['Rolling_Avg_3_Lectures'] = rolling_avg_3

base_momentum = (0.40 * global_mean) + (0.60 * rolling_avg_3)

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
    st.subheader(f"Ensemble Prediction Engine")
    
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        # Predict Class
        cls_model = models[selected_cls_model]
        pred_encoded = cls_model.predict(input_scaled)
        pred_class = le.inverse_transform(pred_encoded)[0]
        
        # Predict Pct
        reg_model = reg_models[selected_reg_model]
        pred_pct = reg_model.predict(input_scaled)[0]
        pred_pct = min(max(pred_pct, 0), 100) # bounds
        
        # Calculate Headcount
        headcount = int(round((pred_pct / 100) * 204))
        
        # Calculate dynamic threshold percentages based on Phase 2 training trend (Residual Bins)
        low_thresh = expected + residual_bins[1]
        high_thresh = expected + residual_bins[2]
        
        # Color coding UI with dynamic bounds
        if pred_class == "High":
            st.success(f"🟢 **RISK BAND:** {pred_class.upper()} ATTENDANCE EXPECTED (Trend Threshold: > {high_thresh:.1f}%)")
        elif pred_class == "Medium":
            st.warning(f"🟡 **RISK BAND:** {pred_class.upper()} ATTENDANCE EXPECTED (Trend Threshold: {low_thresh:.1f}% to {high_thresh:.1f}%)")
        else:
            st.error(f"🔴 **RISK BAND:** {pred_class.upper()} ATTENDANCE EXPECTED (Trend Threshold: < {low_thresh:.1f}%)")
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Percentage", f"{pred_pct:.1f}%")
        with col2:
            st.metric("Predicted Headcount", f"{headcount} / 204 Students")
            
        st.write("---")
        st.write("### 🧠 AI Reasoning (Explainability)")
        
        # Dynamic English Reason
        reason = f"**Why did the AI predict this?**\n\nThe base momentum of attendance right now is **{base_momentum:.1f}%**. "
        reason += f"Because the weather is **{sel_weather}**, the lecture is scheduled in the **{time_cluster}**, and the last 3 lectures rollover average is **{rolling_avg_3:.1f}%**, "
        reason += f"the structural schedule penalties push the expected percentage to **{expected:.1f}%**. "
        reason += f"The {selected_cls_model} grouped this into a **{pred_class}** risk band, while the {selected_reg_model} precisely estimated exactly **{headcount} students** would show up."
        
        st.info(reason)
        
        st.write("### 🧮 Detailed Calculation Weights")
        active_weights = {k: round(v, 2) for k, v in affection_rates.items() if k in input_data.columns and input_data[k].iloc[0] == 1}
        
        weight_df = pd.DataFrame(list(active_weights.items()), columns=["Feature Activated", "Percentage Impact"])
        weight_df.loc[-1] = ["Base Momentum (Starting Point)", round(base_momentum, 2)]
        weight_df.index = weight_df.index + 1
        weight_df = weight_df.sort_index()
        weight_df.loc[len(weight_df)] = ["TOTAL EXPECTED (Clipped)", round(expected, 2)]
        
        st.table(weight_df)

with tab2:
    st.subheader("Automated Experiment Tracking Matrix")
    st.markdown("This table contains the validation scores from our rigorous Phase 3 training block. It fulfills the project deliverable requirement for an Experiment Matrix.")
    
    exp_file = r"d:\coding\ML model\attendance_prediction\data\processed\experiment_results.csv"
    if os.path.exists(exp_file):
        df_exp = pd.read_csv(exp_file)
        # Separate Classification and Regression to rank them properly
        df_class = df_exp[df_exp['Model_Type'] == 'Classification'].dropna(axis=1, how='all')
        df_reg = df_exp[df_exp['Model_Type'] == 'Regression'].dropna(axis=1, how='all')
        
        st.write("### 🥇 Classification Leaderboard (Predicting Risk Band)")
        if not df_class.empty and 'Val_Accuracy' in df_class.columns:
            df_class = df_class.sort_values(by='Val_Accuracy', ascending=False).reset_index(drop=True)
            st.dataframe(df_class, use_container_width=True)
            
        st.write("### 🥇 Regression Leaderboard (Predicting Headcount)")
        if not df_reg.empty and 'Val_R2' in df_reg.columns:
            df_reg = df_reg.sort_values(by='Val_R2', ascending=False).reset_index(drop=True)
            st.dataframe(df_reg, use_container_width=True)
    else:
        st.info("Experiment results not found. Please run the training notebooks.")
