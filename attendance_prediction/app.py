import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime

st.set_page_config(page_title="Attendance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Classroom Attendance Predictor")
st.markdown("Predict the expected attendance risk **(High / Medium / Low)**, exact percentage, and student headcount for any upcoming lecture.")

# --- Load ALL Assets from training pipeline (zero hardcoding) ---
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "deployment_assets")

    scaler      = joblib.load(os.path.join(assets_dir, 'scaler.pkl'))
    le          = joblib.load(os.path.join(assets_dir, 'label_encoder.pkl'))
    rates       = joblib.load(os.path.join(assets_dir, 'affection_rates.pkl'))
    cols        = joblib.load(os.path.join(assets_dir, 'feature_columns.pkl'))
    bins        = joblib.load(os.path.join(assets_dir, 'residual_bins.pkl'))
    g_mean      = joblib.load(os.path.join(assets_dir, 'global_mean.pkl'))

    # Dynamically derive dropdown options from feature columns (no hardcoding!)
    subjects  = sorted([c.replace("Subject_", "") for c in cols if c.startswith("Subject_")])
    weathers  = sorted([c.replace("Weather_", "") for c in cols if c.startswith("Weather_")])
    time_opts = sorted([c.replace("Time_of_Day_", "") for c in cols if c.startswith("Time_of_Day_")])
    day_opts  = sorted([c.replace("Day_of_Week_", "") for c in cols if c.startswith("Day_of_Week_")])

    # Load Models
    models = {}
    model_files = {
        "XGBoost (Champion)":    "xgboost.pkl",
        "Random Forest":         "random_forest.pkl",
        "Decision Tree":         "decision_tree.pkl",
        "Support Vector Machine":"svm.pkl",
        "k-Nearest Neighbors":   "knn.pkl",
        "Naive Bayes":           "naive_bayes.pkl",
        "Logistic Regression":   "logistic_regression.pkl",
    }
    for display_name, fname in model_files.items():
        fpath = os.path.join(assets_dir, fname)
        if os.path.exists(fpath):
            models[display_name] = joblib.load(fpath)

    return scaler, le, rates, cols, bins, g_mean, models, subjects, weathers, time_opts, day_opts

try:
    scaler, le, affection_rates, feature_cols, residual_bins, global_mean, models, \
        subjects, weathers, time_clusters, day_names = load_assets()
except Exception as e:
    st.error(f"Error loading backend models: {e}. Please run the export_pipeline.py script first.")
    st.stop()

# ──────────────────────────────────────────────
# SIDEBAR  (everything comes from training data)
# ──────────────────────────────────────────────
st.sidebar.header("📝 Lecture Details")

# 1. Date → auto-derive Day and Month
sel_date   = st.sidebar.date_input("Select Date", datetime.date.today())
day_name   = sel_date.strftime("%A")   # e.g. "Thursday"
month_num  = sel_date.month
week_num   = sel_date.isocalendar()[1]
is_holiday_adj = 1 if day_name in ["Monday", "Friday"] else 0

# Warn if day_name is not in the trained data's days
if day_name not in day_names:
    st.sidebar.warning(f"⚠️ {day_name} was not in the training data — prediction may be less accurate.")

# 2. Time of Day — options pulled from feature columns
sel_time_cluster = st.sidebar.selectbox("Time of Day", time_clusters, index=0)
is_post_lunch    = 1 if sel_time_cluster == "Afternoon" else 0

# 3. Subject — options pulled from feature columns
sel_subject = st.sidebar.selectbox("Subject", subjects, index=0)

# 4. Theory / Practical — detected from feature columns
practical_col   = "Practical_Theory_Practical"
theory_col      = "Practical_Theory_Theory"
has_practical   = practical_col in feature_cols or theory_col in feature_cols
is_practical    = 0
if has_practical:
    ttype        = st.sidebar.radio("Class Type", ["Theory", "Practical"])
    is_practical = 1 if ttype == "Practical" else 0

# 5. Raw simulation constraints
st.sidebar.divider()
st.sidebar.header("🔬 Simulation Inputs")
sel_weather   = st.sidebar.selectbox("Weather Condition", weathers, index=0)
is_exam_prox  = st.sidebar.checkbox("Week Before Exam?", value=False)
prev_pct      = st.sidebar.slider("Previous Lecture Attendance (%)", 0.0, 100.0, float(global_mean))
rolling_avg_3 = st.sidebar.slider("Last 3 Lectures Average (%)", 0.0, 100.0, float(global_mean))
gap_days      = st.sidebar.slider("Gap Since Last Lecture (Days)", 1, 14, 2)
lecture_num   = st.sidebar.slider("Lecture Number (of semester)", 1, 90, 45)

# 6. Model Selector
st.sidebar.divider()
st.sidebar.header("🤖 Engine Settings")
selected_model_name = st.sidebar.selectbox("Prediction Algorithm", list(models.keys()))

st.sidebar.info(f"📅 Day **{day_name}** • Month **{month_num}** • Week **{week_num}** auto-calculated from date.")

# ──────────────────────────────────────────────
# BACKEND: Build input row exactly as trained
# ──────────────────────────────────────────────
input_data          = pd.DataFrame(columns=feature_cols)
input_data.loc[0]   = 0  # init all to 0

# Numeric features
def _set(col, val):
    if col in input_data.columns:
        input_data[col] = val

_set('Lecture_Number',               lecture_num)
_set('Is_Post_Lunch_Class',          is_post_lunch)
_set('Is_Holiday_Adjacent',          is_holiday_adj)
_set('Week_Before_Exam',             1 if is_exam_prox else 0)
_set('Week_Before_Exam_Flag',        1 if is_exam_prox else 0)   # both naming conventions
_set('Previous_Lecture_Attendance',  prev_pct)
_set('Previous_Lecture_Attendance_Pct', prev_pct)
_set('Rolling_Avg_3',                rolling_avg_3)
_set('Rolling_Avg_3_Lectures',       rolling_avg_3)
_set('Monthly_Avg_Attendance',       global_mean)
_set('Monthly_Expanding_Mean',       global_mean)
_set('Month',                        month_num)
_set('Week_Number',                  week_num)
_set('Day_of_Semester',             lecture_num)   # proxy
_set('Gap_Since_Previous_Lecture_Days', gap_days)

# Dummy / one-hot features (set matched column to 1)
def _flag(prefix, value):
    col = f"{prefix}_{value}"
    if col in input_data.columns:
        input_data[col] = 1

_flag('Weather',       sel_weather)
_flag('Day_of_Week',   day_name)
_flag('Subject',       sel_subject)
_flag('Time_of_Day',   sel_time_cluster)
_flag('Time_of_Day_Cluster', sel_time_cluster)  # both naming conventions

if is_practical:
    _flag('Practical_Theory', 'Practical')
else:
    _flag('Practical_Theory', 'Theory')

# ──────────────────────────────────────────────
# Expected Attendance (data-driven momentum)
# ──────────────────────────────────────────────
base_momentum = (0.40 * global_mean) + (0.60 * rolling_avg_3)

expected = base_momentum
for col, weight in affection_rates.items():
    if col in input_data.columns and input_data[col].iloc[0] == 1:
        expected += weight
expected = float(np.clip(expected, 0, 100))

# ──────────────────────────────────────────────
# Scale
# ──────────────────────────────────────────────
input_scaled = scaler.transform(input_data)

# ──────────────────────────────────────────────
# DATA-DRIVEN BAND THRESHOLDS from Phase 2 qcut
# residual_bins = [-inf, low_bound, high_bound, +inf]
# Low:    residual < bins[1]
# Medium: bins[1] <= residual < bins[2]
# High:   residual >= bins[2]
# In terms of absolute %:
# ──────────────────────────────────────────────
low_abs_thresh  = expected + residual_bins[1]   # below this → Low
high_abs_thresh = expected + residual_bins[2]   # above this → High

# ──────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Prediction Engine", "📊 Model Leaderboard"])

with tab1:
    st.subheader(f"Prediction Engine — {selected_model_name}")

    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        model       = models[selected_model_name]
        pred_enc    = model.predict(input_scaled)
        pred_class  = le.inverse_transform(pred_enc)[0]

        # Predicted percentage = expected + small offset based on band
        # We use expected as the regression proxy since regressors are optional here
        pred_pct     = expected
        headcount    = int(round((pred_pct / 100) * 204))

        # ── 1. Risk Band ──
        st.markdown("### 🎯 Risk Band")
        if pred_class == "High":
            st.success(
                f"🟢 **HIGH ATTENDANCE** — Overperformance Band\n\n"
                f"Attendance predicted to exceed **{high_abs_thresh:.1f}%** "
                f"(trained threshold: residual > +{residual_bins[2]:.2f})"
            )
        elif pred_class == "Medium":
            st.warning(
                f"🟡 **MEDIUM ATTENDANCE** — As-Expected Band\n\n"
                f"Attendance predicted between **{low_abs_thresh:.1f}% – {high_abs_thresh:.1f}%** "
                f"(trained residual window: {residual_bins[1]:.2f} to {residual_bins[2]:.2f})"
            )
        else:
            st.error(
                f"🔴 **LOW ATTENDANCE** — Underperformance Band\n\n"
                f"Attendance predicted to fall below **{low_abs_thresh:.1f}%** "
                f"(trained threshold: residual < {residual_bins[1]:.2f})"
            )

        # ── 2. Numeric outputs ──
        st.markdown("### 📊 Numeric Estimates")
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Attendance %",  f"{pred_pct:.1f}%")
        c2.metric("Predicted Headcount",    f"{headcount} students")
        c3.metric("Out of",                 "204 enrolled")

        st.divider()

        # ── 3. AI Reasoning ──
        st.markdown("### 🧠 Why this prediction?")
        exam_txt   = "**YES — exam week pressure**" if is_exam_prox else "no exam this week"
        weather_wt = affection_rates.get(f"Weather_{sel_weather}", 0)
        day_wt     = affection_rates.get(f"Day_of_Week_{day_name}", 0)
        subj_wt    = affection_rates.get(f"Subject_{sel_subject}", 0)
        time_wt    = affection_rates.get(f"Time_of_Day_{sel_time_cluster}", 0)

        reason = (
            f"The **Base Momentum** is **{base_momentum:.1f}%** "
            f"(40% of historical mean {global_mean:.1f}% + 60% of last-3-lecture avg {rolling_avg_3:.1f}%). "
            f"Today is **{day_name}** (day weight: **{day_wt:+.2f}%**), "
            f"weather is **{sel_weather}** (weather weight: **{weather_wt:+.2f}%**), "
            f"subject is **{sel_subject}** (subject weight: **{subj_wt:+.2f}%**), "
            f"time cluster is **{sel_time_cluster}** (time weight: **{time_wt:+.2f}%**). "
            f"Exam proximity: {exam_txt}. "
            f"After all factors, expected attendance is **{expected:.1f}%** → "
            f"the **{pred_class}** risk band (from Phase 2 residual-qcut thresholds)."
        )
        st.info(reason)

        # ── 4. Calculation weight table ──
        st.markdown("### 🧮 Detailed Calculation Table")
        rows = [{"Factor": "Base Momentum (Starting Point)", "Weight Applied (%)": round(base_momentum, 2)}]
        active = {k: round(v, 2) for k, v in affection_rates.items()
                  if k in input_data.columns and input_data[k].iloc[0] == 1}
        for factor, w in active.items():
            rows.append({"Factor": factor, "Weight Applied (%)": w})
        rows.append({"Factor": "━━━ TOTAL EXPECTED (clipped 0–100) ━━━", "Weight Applied (%)": round(expected, 2)})
        rows.append({"Factor": "Low Band Threshold (from Phase 2 training)", "Weight Applied (%)": round(low_abs_thresh, 2)})
        rows.append({"Factor": "High Band Threshold (from Phase 2 training)", "Weight Applied (%)": round(high_abs_thresh, 2)})

        weight_df = pd.DataFrame(rows)
        st.table(weight_df)

with tab2:
    st.subheader("Model Leaderboard — Experiment Results")
    st.markdown("Scores from Phase 3 chronological validation. Ranked by validation accuracy.")

    exp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "experiment_results.csv")
    if not os.path.exists(exp_file):
        # fallback
        exp_file = r"d:\coding\ML model\classroom-attendance-prediction\data\processed\experiment_results.csv"

    if os.path.exists(exp_file):
        df_exp   = pd.read_csv(exp_file)
        df_class = df_exp[df_exp['Model_Type'] == 'Classification'].dropna(axis=1, how='all')
        df_reg   = df_exp[df_exp['Model_Type'] == 'Regression'].dropna(axis=1, how='all')

        st.markdown("#### 🥇 Classification Models (Predicting Risk Band)")
        if not df_class.empty and 'Val_Accuracy' in df_class.columns:
            df_class = df_class.sort_values('Val_Accuracy', ascending=False).reset_index(drop=True)
            st.dataframe(df_class, use_container_width=True)

        st.markdown("#### 📈 Regression Models (Predicting Headcount)")
        if not df_reg.empty and 'Val_R2' in df_reg.columns:
            df_reg = df_reg.sort_values('Val_R2', ascending=False).reset_index(drop=True)
            st.dataframe(df_reg, use_container_width=True)
    else:
        st.info("Experiment results not found. Run the training notebooks first.")
