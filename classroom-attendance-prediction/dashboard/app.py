import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Classroom Attendance Predictor", page_icon="🏫", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #00ADB5;
    }
    .stProgress .st-bo {
        background-color: #00ADB5;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1E2329;
        text-align: center;
        border-left: 5px solid #00ADB5;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../models/final_model.pkl")
    return joblib.load(model_path)

artifacts = load_model()
pipeline = artifacts['pipeline']
le = artifacts['label_encoder']

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Predict Attendance", "Historical Insights"])

if page == "Dashboard Overview":
    st.title("🏫 Classroom Attendance Predictor")
    st.markdown("""
    Welcome to the proactive **Attendance Prediction System**. 
    
    This AI-driven tool leverages historical lecture data, temporal features, and environmental contexts to forecast whether a specific upcoming class will experience **Low**, **Medium**, or **High** attendance.
    
    ### Key Features
    - 🧠 **XGBoost Inference**: Powered by a robust gradient boosting model.
    - ⏱️ **Temporal Awareness**: Accounts for class momentum and day-of-week trends.
    - 📊 **Historical Insights**: Visualize exact trends driving student engagement.
    """)
    
    st.info("Navigate using the sidebar to make a prediction or view insights.")

elif page == "Historical Insights":
    st.title("📊 Historical Insights")
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
    st.title("🔮 Predict Upcoming Attendance")
    st.markdown("Input the context of an upcoming lecture to forecast expected attendance.")
    
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
            
        with col2:
            day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
            week = st.number_input("Week of Semester", min_value=1, max_value=20, value=5)
            start_time = st.selectbox("Start Time", ['8.30 AM', '9.15 AM', '10.15 AM', '11.15 AM', '1.30 PM', '2.30 PM'])
            
        with col3:
            rolling_avg = st.slider("Recent Momentum (Rolling Avg %)", 0.0, 100.0, 20.0)
            internal_test = st.selectbox("Internal Test Week?", ['No', 'Yes'])
            practical = st.selectbox("Theory or Practical?", ['Theory', 'Practical'])
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy'])
            
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
        
        with st.spinner("Analyzing..."):
            pred_encoded = pipeline.predict(input_df)[0]
            pred_class = le.inverse_transform([pred_encoded])[0]
            
            # Get probabilities
            probs = pipeline.predict_proba(input_df)[0]
            
        st.markdown("---")
        st.subheader("Prediction Results")
        
        color = "#FF4B4B" if pred_class == 'Low' else "#FACA2B" if pred_class == 'Medium' else "#00C246"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Expected Attendance: <span style="color:{color};">{pred_class.upper()}</span></h2>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.write("Confidence Probabilities:")
        for idx, cls in enumerate(le.classes_):
            st.write(f"{cls}: {probs[idx]:.1%}")
            st.progress(float(probs[idx]))
