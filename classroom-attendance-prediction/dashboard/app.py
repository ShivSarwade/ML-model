import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Classroom Attendance Predictor", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0A0A0A;
        color: #E2E2E2;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 500;
        letter-spacing: -0.02em;
    }
    .stProgress .st-bo {
        background-color: #5E6AD2;
    }
    .prediction-box {
        padding: 24px;
        border-radius: 6px;
        background-color: #121212;
        border: 1px solid #2A2A2A;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #5E6AD2;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #4B55C3;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), "../models/all_models.pkl")
    return joblib.load(model_path)

all_models = load_models()
le = all_models['label_encoder']

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Predict Attendance", "Historical Insights"])

if page == "Dashboard Overview":
    st.title("Classroom Attendance Predictor")
    st.markdown("""
    Welcome to the **Attendance Prediction System**. 
    
    This analytical tool leverages historical scheduling data, temporal dynamics, and environmental variables to forecast whether a specific upcoming class will experience Low, Medium, or High attendance, or to output an exact percentage estimate.
    
    ### System Architecture
    - **Inference Engine**: Employs robust ensemble and linear models, with XGBoost serving as the primary predictor.
    - **Temporal Analysis**: Incorporates strict time-series integrity to measure cohort momentum and schedule clustering.
    - **Historical Metrics**: Access exact statistical trends detailing cohort engagement profiles.
    """)
    
    st.info("Navigate using the sidebar to generate a prediction or view data insights.")

elif page == "Historical Insights":
    st.title("Historical Insights")
    st.markdown("These visualizations were generated during the Exploratory Data Analysis phase on our historical data.")
    
    viz_dir = os.path.join(os.path.dirname(__file__), "../visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attendance Distribution")
        if os.path.exists(os.path.join(viz_dir, "attendance_distribution.png")):
            st.image(os.path.join(viz_dir, "attendance_distribution.png"), use_container_width=True)
            
        st.subheader("Attendance by Time Slot")
        if os.path.exists(os.path.join(viz_dir, "attendance_by_time.png")):
            st.image(os.path.join(viz_dir, "attendance_by_time.png"), use_container_width=True)
            
    with col2:
        st.subheader("Attendance by Day of Week")
        if os.path.exists(os.path.join(viz_dir, "attendance_by_day.png")):
            st.image(os.path.join(viz_dir, "attendance_by_day.png"), use_container_width=True)
            
        st.subheader("Attendance by Subject")
        if os.path.exists(os.path.join(viz_dir, "attendance_by_subject.png")):
            st.image(os.path.join(viz_dir, "attendance_by_subject.png"), use_container_width=True)

elif page == "Predict Attendance":
    st.title("Predict Upcoming Attendance")
    st.markdown("Configure the parameters of an upcoming lecture to execute inference.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            subject = st.selectbox("Subject", [
                'Mobile Application Development', 
                'Innovation and Entrepreneurship Development',
                'STQA Practical',
                'Principles of Cloud Management and Security',
                'Data Science & Machine Learning'
            ])
            section = st.selectbox("Section", ['A&B', 'A', 'B'])
            faculty = st.text_input("Faculty ID", "SSP")
            task_type = st.radio("Prediction Type", ["Classification (Band)", "Regression (Exact %)"])
            
            if task_type == "Classification (Band)":
                model_choice = st.selectbox("Select Model", list(all_models['Classification'].keys()), index=6) # Default to XGBoost
            else:
                model_choice = st.selectbox("Select Model", list(all_models['Regression'].keys()), index=4)
            
        with col2:
            day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
            week = st.number_input("Week of Semester", min_value=1, max_value=20, value=5)
            start_time = st.selectbox("Start Time", ['8.30 AM', '9.15 AM', '10.15 AM', '11.15 AM', '1.30 PM', '2.30 PM'])
            
            # New Temporal Features
            day_of_semester = st.number_input("Day of Semester", min_value=1, max_value=150, value=30)
            time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon"])
            days_since_hol = st.number_input("Days Since Last Holiday", min_value=0, max_value=100, value=5)
            
        with col3:
            rolling_avg = st.slider("Recent Momentum (Rolling Avg %)", 0.0, 100.0, 20.0)
            internal_test = st.selectbox("Internal Test Week?", ['No', 'Yes'])
            practical = st.selectbox("Theory or Practical?", ['Theory', 'Practical'])
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy'])
            
            # New Additional Features
            week_before_exam = st.selectbox("Week Before Exam?", [0, 1])
            consecutive_lectures = st.number_input("Consecutive Lecture Count", min_value=1, max_value=5, value=1)
            monthly_avg = st.slider("Monthly Average Trend (%)", 0.0, 100.0, 45.0)
            
        submit = st.form_submit_button("Generate Prediction")
        
    if submit:
        # Construct DataFrame matching the pipeline input
        input_data = {
            'Subject': [subject],
            'Section': [section],
            'Faculty_ID': [faculty],
            'Day_of_Week': [day],
            'Week_Number': [week],
            'Start_Time': [start_time],
            'Rolling_Avg_3': [rolling_avg],
            'Internal_Test_Week': [internal_test],
            'Practical_Theory': [practical],
            'Weather': [weather],
            'Day_of_Semester': [day_of_semester],
            'Time_of_Day': [time_of_day],
            'Days_Since_Last_Holiday': [days_since_hol],
            'Week_Before_Exam': [week_before_exam],
            'Consecutive_Lecture_Count': [consecutive_lectures],
            'Monthly_Avg_Attendance': [monthly_avg],
            # Mock defaults for the rest
            'Lecture_Number': [1],
            'End_Time': ['10:00 AM'],
            'Semester': [3],
            'Branch': ['MCA'],
            'Classroom': ['Computer Lab'],
            'Previous_Lecture_Attendance': [30],
            'Gap_Since_Previous_Lecture': ['1 Day'],
            'Assignment_Due': ['No'],
            'Holiday_Before_After': ['No'],
            'Special_Event': ['No']
        }
        
        input_df = pd.DataFrame(input_data)
        
        st.markdown("---")
        st.subheader(f"Prediction Results: {model_choice}")
        
        with st.spinner("Analyzing..."):
            if task_type == "Classification (Band)":
                pipeline = all_models['Classification'][model_choice]
                pred_encoded = pipeline.predict(input_df)[0]
                pred_class = le.inverse_transform([pred_encoded])[0]
                
                color = "#FF4B4B" if pred_class == 'Low' else "#FACA2B" if pred_class == 'Medium' else "#00C246"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Expected Attendance: <span style="color:{color};">{pred_class.upper()}</span></h2>
                </div>
                <br>
                """, unsafe_allow_html=True)
                
                try:
                    probs = pipeline.predict_proba(input_df)[0]
                    st.write("Confidence Probabilities:")
                    for idx, cls in enumerate(le.classes_):
                        st.write(f"{cls}: {probs[idx]:.1%}")
                        st.progress(float(probs[idx]))
                except Exception:
                    st.write("Confidence probabilities not supported by this algorithm.")
                    
            else:
                pipeline = all_models['Regression'][model_choice]
                pred_val = pipeline.predict(input_df)[0]
                
                color = "#FF4B4B" if pred_val < 30 else "#FACA2B" if pred_val < 65 else "#00C246"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Exact Expected Attendance: <span style="color:{color};">{pred_val:.1f}%</span></h2>
                </div>
                <br>
                """, unsafe_allow_html=True)
