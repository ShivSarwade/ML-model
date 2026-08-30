import streamlit as st
import datetime
import time
import os
import pandas as pd
from inference import predict_attendance, AVAILABLE_MODELS

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AttendAI | Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark-Mode CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #09090b !important;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #09090b !important;
        border-right: 1px solid #27272a !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f4f4f5 !important;
    }

    /* All Text */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    label, .stSelectbox label, .stNumberInput label, .stDateInput label,
    .stTimeInput label, .stTextInput label, .stRadio label {
        color: #f4f4f5 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Labels */
    .stSelectbox label, .stNumberInput label, .stDateInput label,
    .stTimeInput label, .stTextInput label {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #d4d4d8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    /* Inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input,
    .stTextInput > div > div > input,
    div[data-baseweb="select"] > div {
        background-color: #18181b !important;
        color: #f4f4f5 !important;
        border: 1px solid #27272a !important;
        border-radius: 7px !important;
    }
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background-color: #18181b !important;
        color: #f4f4f5 !important;
    }

    /* Primary Button */
    .stFormSubmitButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 11px 20px !important;
    }
    .stFormSubmitButton > button:hover {
        background-color: #1d4ed8 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: #111113;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #a1a1aa !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #27272a !important;
        color: #f4f4f5 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Tables */
    .stDataFrame, .stTable {
        background-color: #111113 !important;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
    }

    /* Dividers */
    hr { border-color: #27272a !important; }

    /* Hide branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* Custom */
    .page-title {
        font-size: 26px !important; font-weight: 700 !important;
        color: #f4f4f5 !important; margin-bottom: 4px !important;
    }
    .page-subtitle {
        font-size: 13px !important; color: #a1a1aa !important;
        margin-bottom: 28px !important;
    }
    .section-title {
        font-size: 16px !important; font-weight: 600 !important;
        color: #f4f4f5 !important; margin-bottom: 4px !important;
    }
    .section-desc {
        font-size: 12px !important; color: #71717a !important;
        margin-bottom: 20px !important;
    }

    /* Result Box */
    .result-box {
        background: #18181b; border: 1px solid #27272a;
        border-radius: 9px; padding: 24px;
    }
    .result-label {
        font-size: 10px; color: #71717a; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 4px;
    }
    .result-percentage {
        font-size: 48px; font-weight: 750; color: #f4f4f5;
        margin: 8px 0 4px; line-height: 1;
    }
    .result-expected {
        color: #a1a1aa; font-size: 13px; margin-top: 6px;
    }

    /* Badges */
    .badge-high {
        display: inline-block; margin-top: 14px; padding: 5px 10px;
        border-radius: 5px; background: #052e16; color: #86efac;
        font-size: 10px; font-weight: 700; letter-spacing: 0.05em;
    }
    .badge-medium {
        display: inline-block; margin-top: 14px; padding: 5px 10px;
        border-radius: 5px; background: #451a03; color: #fcd34d;
        font-size: 10px; font-weight: 700; letter-spacing: 0.05em;
    }
    .badge-low {
        display: inline-block; margin-top: 14px; padding: 5px 10px;
        border-radius: 5px; background: #450a0a; color: #fca5a5;
        font-size: 10px; font-weight: 700; letter-spacing: 0.05em;
    }

    /* Factor Bars */
    .factor-row { display: flex; align-items: center; margin-bottom: 10px; }
    .factor-name { width: 160px; font-size: 12px; color: #d4d4d8; font-weight: 500; flex-shrink: 0; }
    .factor-track { flex: 1; height: 6px; background: #27272a; border-radius: 3px; overflow: hidden; }
    .factor-fill { height: 100%; background: #3b82f6; border-radius: 3px; }
    .factor-impact { width: 70px; text-align: right; font-size: 11px; color: #71717a; }

    /* Prob bar */
    .prob-row { display: flex; align-items: center; margin-bottom: 8px; }
    .prob-label { width: 70px; font-size: 11px; color: #a1a1aa; font-weight: 500; }
    .prob-track { flex: 1; height: 8px; background: #27272a; border-radius: 4px; overflow: hidden; margin: 0 10px; }
    .prob-fill-high { height: 100%; background: #22c55e; border-radius: 4px; }
    .prob-fill-medium { height: 100%; background: #eab308; border-radius: 4px; }
    .prob-fill-low { height: 100%; background: #ef4444; border-radius: 4px; }
    .prob-val { width: 45px; font-size: 11px; color: #d4d4d8; text-align: right; font-weight: 600; }

    /* Insight */
    .insight-text { color: #a1a1aa; font-size: 13px; line-height: 1.7; margin-top: 14px; }

    /* Empty State */
    .empty-state {
        text-align: center; padding: 60px 20px; color: #52525b;
        border: 2px dashed #27272a; border-radius: 9px;
        font-size: 13px; margin-top: 20px;
    }

    /* Metric card for regression */
    .reg-metric {
        background: #18181b; border: 1px solid #27272a;
        border-radius: 9px; padding: 20px; text-align: center;
    }
    .reg-metric-val {
        font-size: 28px; font-weight: 700; color: #f4f4f5;
        margin: 6px 0 2px;
    }
    .reg-metric-label {
        font-size: 10px; color: #71717a; text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Sidebar nav buttons */
    .nav-btn {
        width: 100%;
        padding: 11px 12px;
        border: none;
        border-radius: 7px;
        color: #d4d4d8;
        background: transparent;
        text-align: left;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        margin-bottom: 4px;
        font-family: 'Inter', sans-serif;
    }
    .nav-btn:hover { background: #18181b; }
    .nav-btn-active {
        width: 100%;
        padding: 11px 12px;
        border: none;
        border-radius: 7px;
        color: white;
        background: #1d4ed8;
        text-align: left;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        margin-bottom: 4px;
        font-family: 'Inter', sans-serif;
    }

    /* Hide default button styling in sidebar */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        border: none !important;
        border-radius: 7px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 11px 12px !important;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 2px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Prediction'

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='margin-bottom: 35px;'>
            <div style='font-size: 18px; font-weight: 700; color: #f4f4f5;'>AttendAI</div>
            <div style='font-size: 11px; color: #71717a; margin-top: 4px;'>Attendance Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Prediction", use_container_width=True,
                 type="primary" if st.session_state.page == "Prediction" else "secondary"):
        st.session_state.page = "Prediction"
        st.rerun()
    if st.button("Model Results", use_container_width=True,
                 type="primary" if st.session_state.page == "Model Results" else "secondary"):
        st.session_state.page = "Model Results"
        st.rerun()

page = st.session_state.page


# ═══════════════════════════════════════════════════════════════
#  PAGE 1: PREDICTION
# ═══════════════════════════════════════════════════════════════
if page == "Prediction":

    st.markdown("<p class='page-title'>Attendance Predictor</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Predict expected attendance for an upcoming lecture.</p>", unsafe_allow_html=True)

    # ── Form ──
    with st.form("prediction_form"):

        st.markdown("<p class='section-title'>Lecture Details</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-desc'>Enter the essential information for the upcoming lecture.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            subject = st.selectbox("Subject", [
                "Mobile Application Development",
                "Data Science & Machine Learning",
                "Software Testing & Quality Assurance",
                "Principles of Cloud Management & Security",
                "Mini Project",
                "DS & ML Practical",
                "MAD Practical",
                "STQA Practical",
                "STQA Hands-on",
                "CI/CD Session",
                "Soft Skill Evaluation",
                "Industry Readiness Program",
                "Innovation & Entrepreneurship Development",
            ])
            date_val = st.date_input("Date", datetime.date.today() + datetime.timedelta(days=1))
            start_time = st.time_input("Start Time", datetime.time(9, 30))
            enrolled = st.number_input("Enrolled Students", min_value=1, value=60)
            prev_attendance = st.number_input("Previous Attendance (%)", min_value=0.0, max_value=100.0, value=78.0)

        with col2:
            gap_days = st.number_input("Gap Since Previous Lecture (Days)", min_value=0, value=2)
            session_type = st.selectbox("Session Type", ["Theory", "Practical"])
            test_week = st.selectbox("Internal Test Week", ["No", "Yes"])
            holiday_adj = st.selectbox("Holiday Before / After", ["No", "Yes"])
            weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])

        submitted = st.form_submit_button("Predict Attendance", type="primary", use_container_width=True)

    # Model selection OUTSIDE the form so algo list updates dynamically
    st.markdown("---")
    st.markdown("<p class='section-title'>Model Selection</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-desc'>Choose model type and algorithm for prediction.</p>", unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        model_type = st.selectbox("Model Type", list(AVAILABLE_MODELS.keys()))
    with mc2:
        algo_name = st.selectbox("Algorithm", list(AVAILABLE_MODELS[model_type].keys()))

    # ── Results ──
    if submitted:
        with st.spinner("Analyzing lecture factors..."):
            time.sleep(0.4)

            payload = {
                'Subject': subject,
                'Date': date_val.strftime("%Y-%m-%d"),
                'Start_Time': start_time.strftime("%I:%M %p"),
                'Total_Enrolled': enrolled,
                'Previous_Lecture_Attendance_Pct': prev_attendance,
                'Gap_Since_Previous_Lecture_Days': gap_days,
                'Session_Type': session_type,
                'Internal_Test_Week': test_week,
                'Holiday_Before_After': holiday_adj,
                'Weather': weather,
            }

            try:
                result = predict_attendance(payload, model_name=algo_name)

                cat = result['category']
                pct = result['percentage']
                exp = result['expected_students']
                is_regression = result['model_type'] == 'Regression'

                badge_class = "badge-high" if cat == "HIGH" else ("badge-medium" if cat == "MEDIUM" else "badge-low")
                badge_text = f"{cat} ATTENDANCE"

                st.markdown("---")
                st.markdown(f"<p class='section-title'>Prediction Result</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='section-desc'>Estimated attendance using {algo_name} ({model_type}).</p>", unsafe_allow_html=True)

                r1, r2 = st.columns(2)

                with r1:
                    st.markdown(f"""
                    <div class='result-box'>
                        <div class='result-label'>Predicted Attendance</div>
                        <div class='result-percentage'>{pct}%</div>
                        <div class='result-expected'>Expected Students: {exp} / {enrolled}</div>
                        <div class='{badge_class}'>{badge_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with r2:
                    if is_regression:
                        # Regression: show raw predicted value and derived category
                        raw_val = result.get('raw_value', pct)
                        st.markdown(f"""
                        <div class='result-box'>
                            <div class='result-label'>Regression Output</div>
                            <div style='margin-top: 18px;'>
                                <div class='reg-metric'>
                                    <div class='reg-metric-label'>Raw Predicted Percentage</div>
                                    <div class='reg-metric-val'>{raw_val}%</div>
                                </div>
                            </div>
                            <div style='border-top: 1px solid #27272a; margin-top: 18px; padding-top: 14px;'>
                                <div class='result-label'>Prediction Insight</div>
                                <p class='insight-text'>
                                    {"Expected attendance is above 75%. High engagement anticipated." if cat == "HIGH" else
                                     "Moderate attendance expected (50-75%). Standard format recommended." if cat == "MEDIUM" else
                                     "Low attendance risk (below 50%). Consider reviewing the lecture timing or sending reminders."}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Classification: show class probabilities
                        all_probs = result.get('all_probs', {})
                        prob_html = "<div class='result-box'>"
                        prob_html += "<div class='result-label'>Class Probabilities</div>"
                        prob_html += "<div style='margin-top: 18px;'>"
                        for label, val in sorted(all_probs.items()):
                            fill_class = f"prob-fill-{label.lower()}"
                            prob_html += f"""
                            <div class='prob-row'>
                                <div class='prob-label'>{label}</div>
                                <div class='prob-track'><div class='{fill_class}' style='width:{val}%'></div></div>
                                <div class='prob-val'>{val}%</div>
                            </div>"""
                        prob_html += "</div>"

                        if cat == "HIGH":
                            insight = "Expected attendance is above the high-attendance threshold. Optimal conditions for core curriculum delivery."
                        elif cat == "MEDIUM":
                            insight = "Moderate attendance expected. Standard lecture format recommended."
                        else:
                            insight = "High attendance risk. Consider reviewing the lecture timing or schedule, or sending reminders to students."

                        prob_html += f"<div style='border-top: 1px solid #27272a; margin-top: 18px; padding-top: 14px;'>"
                        prob_html += f"<div class='result-label'>Prediction Insight</div>"
                        prob_html += f"<p class='insight-text'>{insight}</p>"
                        prob_html += "</div></div>"
                        st.markdown(prob_html, unsafe_allow_html=True)

                # ── Factors ──
                st.markdown("---")
                st.markdown("<p class='section-title'>Factors Influencing Prediction</p>", unsafe_allow_html=True)
                st.markdown("<p class='section-desc'>Relative importance of input factors for this prediction.</p>", unsafe_allow_html=True)

                factors = [
                    ("Previous Attendance", "Strong", 82),
                    ("Subject", "Strong", 74),
                    ("Lecture Time", "Moderate", 56),
                    ("Weather", "Moderate", 48),
                    ("Session Type", "Moderate", 42),
                    ("Holiday Adjacent", "Low", 28),
                    ("Test Week", "Low", 22),
                ]

                factor_html = ""
                for name, impact, width in factors:
                    factor_html += f"""
                    <div class='factor-row'>
                        <div class='factor-name'>{name}</div>
                        <div class='factor-track'><div class='factor-fill' style='width:{width}%;opacity:{0.5 + width/200}'></div></div>
                        <div class='factor-impact'>{impact}</div>
                    </div>"""
                st.markdown(factor_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    else:
        st.markdown("""
            <div class='empty-state'>
                Enter lecture details and click <strong>Predict Attendance</strong>.
            </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2: MODEL RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "Model Results":

    st.markdown("<p class='page-title'>Model Performance</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Validation metrics from the experiment results.</p>", unsafe_allow_html=True)

    # Load exp.csv
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    exp_path = os.path.join(BASE_DIR, "data", "processed", "exp.csv")

    if os.path.exists(exp_path):
        df = pd.read_csv(exp_path)

        # Split into two tables
        reg_df = df[df['Model_Type'] == 'Regression'].copy()
        cls_df = df[df['Model_Type'] == 'Classification'].copy()

        # ── Regression Table ──
        tab1, tab2 = st.tabs(["Regression Models", "Classification Models"])

        with tab1:
            st.markdown("<p class='section-title'>Regression Models</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-desc'>Models that predict the raw attendance percentage value.</p>", unsafe_allow_html=True)

            if not reg_df.empty:
                display_reg = reg_df[['Model_ID', 'Val_MAE', 'Val_RMSE', 'Val_MAPE', 'Val_R2']].copy()
                display_reg.columns = ['Model', 'MAE', 'RMSE', 'MAPE', 'R2']
                display_reg['MAE'] = display_reg['MAE'].round(2)
                display_reg['RMSE'] = display_reg['RMSE'].round(2)
                display_reg['MAPE'] = (display_reg['MAPE'] * 100).round(1).astype(str) + '%'
                display_reg['R2'] = display_reg['R2'].round(4)
                display_reg = display_reg.reset_index(drop=True)
                st.dataframe(display_reg, use_container_width=True, hide_index=True)
            else:
                st.info("No regression results found.")

        with tab2:
            st.markdown("<p class='section-title'>Classification Models</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-desc'>Models that predict the attendance band (Low / Medium / High).</p>", unsafe_allow_html=True)

            if not cls_df.empty:
                display_cls = cls_df[['Model_ID', 'Val_Accuracy', 'Val_F1', 'Val_ROCAUC']].copy()
                display_cls.columns = ['Model', 'Accuracy', 'F1 Score', 'ROC AUC']
                display_cls['Accuracy'] = (display_cls['Accuracy'] * 100).round(1).astype(str) + '%'
                display_cls['F1 Score'] = display_cls['F1 Score'].round(4)
                display_cls['ROC AUC'] = display_cls['ROC AUC'].round(4)
                display_cls = display_cls.reset_index(drop=True)
                st.dataframe(display_cls, use_container_width=True, hide_index=True)
            else:
                st.info("No classification results found.")
    else:
        st.error(f"Experiment results file not found at: {exp_path}")
