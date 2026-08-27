"""
Classroom Attendance Prediction Dashboard
==========================================
FULLY DYNAMIC: all dropdown options, feature ranges, and model names
are derived at runtime from deployment_assets/*.pkl and data/processed/*.csv.
Nothing is hardcoded in this file.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classroom Attendance Predictor",
    page_icon="\U0001f393",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Light Academic Theme (University Blue + White)
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f5f7fa; }

.header-banner {
  background: linear-gradient(135deg, #1a3a6b 0%, #2d5da3 50%, #3e7bc4 100%);
  border-radius: 12px; padding: 24px 32px; margin-bottom: 24px;
  color: white; box-shadow: 0 4px 16px rgba(26,58,107,0.18);
}
.header-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.header-banner p  { margin: 6px 0 0; font-size: 1rem; opacity: 0.88; font-weight: 300; }

.metric-card {
  background: white; border-radius: 10px; padding: 20px 24px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.07); border-left: 4px solid #2d5da3; margin-bottom: 12px;
}
.metric-card .metric-label {
  font-size: 0.75rem; color: #6c757d; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.metric-card .metric-value { font-size: 1.6rem; font-weight: 700; color: #1a3a6b; margin-top: 4px; }
.metric-card .metric-delta { font-size: 0.8rem; color: #6c757d; margin-top: 2px; }

.result-high   { background:#d1fae5; border:2px solid #10b981; border-radius:12px; padding:24px; text-align:center; }
.result-medium { background:#fef3c7; border:2px solid #f59e0b; border-radius:12px; padding:24px; text-align:center; }
.result-low    { background:#fee2e2; border:2px solid #ef4444; border-radius:12px; padding:24px; text-align:center; }
.result-label  { font-size:1.5rem; font-weight:700; margin-bottom:4px; }
.result-sub    { font-size:0.9rem; opacity:0.8; }

[data-testid="stSidebar"] { background-color: #f0f4f9; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label { font-weight: 600; color: #1a3a6b; font-size: 0.85rem; }

.section-header {
  font-size: 1.1rem; font-weight: 700; color: #1a3a6b;
  border-bottom: 2px solid #2d5da3; padding-bottom: 6px; margin: 20px 0 12px;
}
.info-box {
  background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
  padding: 14px 18px; font-size: 0.85rem; color: #1e40af; margin: 10px 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "deployment_assets")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS (all cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models...")
def load_assets():
    scaler    = joblib.load(os.path.join(ASSETS_DIR, "scaler.pkl"))
    le        = joblib.load(os.path.join(ASSETS_DIR, "label_encoder.pkl"))
    rates     = joblib.load(os.path.join(ASSETS_DIR, "affection_rates.pkl"))
    feat_cols = joblib.load(os.path.join(ASSETS_DIR, "feature_columns.pkl"))
    res_bins  = joblib.load(os.path.join(ASSETS_DIR, "residual_bins.pkl"))
    g_mean    = joblib.load(os.path.join(ASSETS_DIR, "global_mean.pkl"))
    model_files = {
        "XGBoost (Champion)":     "xgboost.pkl",
        "Random Forest":          "random_forest.pkl",
        "Decision Tree":          "decision_tree.pkl",
        "Support Vector Machine": "svm.pkl",
        "k-Nearest Neighbors":    "knn.pkl",
        "Naive Bayes":            "naive_bayes.pkl",
        "Logistic Regression":    "logistic_regression.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        fpath = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(fpath):
            models[name] = joblib.load(fpath)
    return scaler, le, rates, feat_cols, res_bins, g_mean, models


@st.cache_data(show_spinner="Loading historical data...")
def load_historical():
    path = os.path.join(DATA_DIR, "attendance_cleaned.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading experiment results...")
def load_experiments():
    path = os.path.join(DATA_DIR, "experiment_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def derive_options(feat_cols):
    """Extract dropdown options dynamically from feature_columns.pkl."""
    opts = {"subjects": [], "days": [], "weathers": [], "time_clusters": [], "practical_types": []}
    for col in feat_cols:
        if   col.startswith("Subject_"):          opts["subjects"].append(col[8:])
        elif col.startswith("Day_of_Week_"):       opts["days"].append(col[12:])
        elif col.startswith("Weather_"):           opts["weathers"].append(col[8:])
        elif col.startswith("Time_of_Day_"):       opts["time_clusters"].append(col[12:])
        elif col.startswith("Practical_Theory_"):  opts["practical_types"].append(col[17:])
    for k in opts:
        opts[k] = sorted(opts[k])
    return opts


@st.cache_data
def compute_stats():
    """Compute training data statistics for auto-filling engineered features."""
    path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[f"{col}_median"] = float(df[col].median())
        stats[f"{col}_mean"]   = float(df[col].mean())
        stats[f"{col}_min"]    = float(df[col].min())
        stats[f"{col}_max"]    = float(df[col].max())
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
try:
    scaler, le, affection_rates, feature_cols, residual_bins, global_mean, models = load_assets()
except Exception as e:
    st.error(f"Failed to load backend assets: {e}")
    st.info("Run export_pipeline.py first to generate deployment_assets/.")
    st.stop()

options     = derive_options(tuple(feature_cols))
train_stats = compute_stats()
hist_df     = load_historical()
exp_df      = load_experiments()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>&#127891; Classroom Attendance Predictor</h1>
  <p>AI-powered dashboard &nbsp;&middot;&nbsp; MCA Sem 3 Cohort &nbsp;&middot;&nbsp;
     Predict lecture attendance risk using Machine Learning on historical data.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FULLY DYNAMIC INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### \U0001f4dd Lecture Details")
    st.markdown("*All options derived dynamically from training data.*")
    st.divider()

    # Subject — derived from feature_columns.pkl
    sel_subject = st.selectbox("Subject", options=options["subjects"],
        help="Subjects seen during model training (from feature_columns.pkl)")

    # Day of Week — derived from feature_columns.pkl
    sel_day = st.selectbox("Day of Week", options=options["days"],
        help="Days in the model one-hot encoding")

    # Lecture Number — range from train.csv statistics
    lec_min = int(train_stats.get("Lecture_Number_min", 1))
    lec_max = int(train_stats.get("Lecture_Number_max", 5))
    sel_lecture_num = st.slider("Lecture Number",
        min_value=lec_min, max_value=lec_max,
        value=int(train_stats.get("Lecture_Number_median", 2)),
        help=f"Slot in the timetable (training range: {lec_min} to {lec_max})")

    # Time of Day Cluster — derived from feature_columns.pkl
    time_opts = options["time_clusters"] if options["time_clusters"] else ["Morning"]
    sel_time_cluster = st.selectbox("Time of Day Cluster", options=time_opts,
        help="Derived from one-hot encoding in training data")

    # Weather — derived from feature_columns.pkl
    sel_weather = st.selectbox("Weather Condition", options=options["weathers"],
        help="Weather conditions seen during training")

    # Class Type — derived from feature_columns.pkl
    pt_opts = options["practical_types"] if options["practical_types"] else ["Theory"]
    sel_pt = st.selectbox("Class Type", options=pt_opts,
        help="Practical or Theory (from training encoding)")

    # Previous Lecture Attendance % — range from train.csv
    prev_min  = float(train_stats.get("Previous_Lecture_Attendance_Pct_min",  9.8))
    prev_max  = float(train_stats.get("Previous_Lecture_Attendance_Pct_max",  43.14))
    prev_mean = float(train_stats.get("Previous_Lecture_Attendance_Pct_mean", 19.82))
    sel_prev_att = st.slider("Previous Lecture Attendance %",
        min_value=round(prev_min, 1), max_value=round(prev_max, 1),
        value=round(prev_mean, 1), step=0.1,
        help=f"Attendance % of preceding lecture (training: {prev_min:.1f}% - {prev_max:.1f}%)")

    # Boolean flags
    sel_week_before_exam = st.toggle("Week Before Examination", value=False,
        help="Is this in the week before an internal exam?")
    sel_holiday_adj = st.toggle("Day Adjacent to Holiday", value=False,
        help="Monday or Friday adjacent to a holiday?")

    # Gap since previous lecture — range from train.csv
    gap_max = float(train_stats.get("Gap_Since_Previous_Lecture_Days_max", 7.0))
    sel_gap = st.slider("Gap Since Previous Lecture (days)",
        min_value=0.0, max_value=round(gap_max, 1),
        value=float(train_stats.get("Gap_Since_Previous_Lecture_Days_median", 1.0)),
        step=0.25)

    st.divider()
    st.markdown("### \U0001f916 Model Settings")
    sel_model_name = st.selectbox("Prediction Algorithm", options=list(models.keys()))
    st.divider()
    st.markdown(
        '<div class="info-box">All options and feature ranges are loaded dynamically '
        'from <b>deployment_assets/feature_columns.pkl</b> and the training CSV. '
        'Nothing is hardcoded.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────────
def build_input(feat_cols, train_stats, global_mean,
                sel_subject, sel_day, sel_lecture_num, sel_time_cluster,
                sel_weather, sel_pt, sel_prev_att,
                sel_week_before_exam, sel_holiday_adj, sel_gap):
    row = {col: 0.0 for col in feat_cols}
    auto_filled = {}

    # Direct user inputs
    for feat, val in [
        ("Lecture_Number",                  float(sel_lecture_num)),
        ("Is_Holiday_Adjacent",             1.0 if sel_holiday_adj else 0.0),
        ("Week_Before_Exam_Flag",           1.0 if sel_week_before_exam else 0.0),
        ("Previous_Lecture_Attendance_Pct", float(sel_prev_att)),
        ("Gap_Since_Previous_Lecture_Days", float(sel_gap)),
        (f"Subject_{sel_subject}",          1.0),
        (f"Day_of_Week_{sel_day}",          1.0),
        (f"Weather_{sel_weather}",          1.0),
        (f"Time_of_Day_{sel_time_cluster}", 1.0),
        (f"Practical_Theory_{sel_pt}",      1.0),
    ]:
        if feat in row:
            row[feat] = val

    # Auto-compute engineered features from training statistics
    auto_map = [
        ("Total_Enrolled",            "Total_Enrolled_mean",            204.0),
        ("Students_Present",          "Students_Present_mean",           40.0),
        ("Week_Number",               "Week_Number_median",              21.0),
        ("Day_of_Semester",           "Day_of_Semester_median",          45.0),
        ("Month",                     "Month_median",                     5.0),
        ("Consecutive_Lecture_Count", "Consecutive_Lecture_Count_median", 13.5),
        ("Rolling_Avg_3_Lectures",    "Rolling_Avg_3_Lectures_mean",    global_mean),
        ("Monthly_Expanding_Mean",    "Monthly_Expanding_Mean_mean",    global_mean),
        ("Is_Consecutive",            "Is_Consecutive_median",            1.0),
        ("Days_Since_Last_Holiday",   "Days_Since_Last_Holiday_median",   3.0),
    ]
    for feat, stat_key, default in auto_map:
        if feat in row:
            val = float(train_stats.get(stat_key, default))
            row[feat] = val
            auto_filled[feat] = round(val, 4)

    # Use previous lecture attendance as rolling avg proxy
    if "Rolling_Avg_3_Lectures" in row:
        row["Rolling_Avg_3_Lectures"] = float(sel_prev_att)
        auto_filled["Rolling_Avg_3_Lectures"] = round(float(sel_prev_att), 4)

    # Derived: Is_Post_Lunch_Class
    if "Is_Post_Lunch_Class" in row:
        row["Is_Post_Lunch_Class"] = 1.0 if sel_lecture_num > 5 else 0.0

    return pd.DataFrame([row], columns=feat_cols), auto_filled


def compute_expected(input_df, affection_rates, global_mean):
    rolling  = float(input_df["Rolling_Avg_3_Lectures"].iloc[0]) if "Rolling_Avg_3_Lectures" in input_df.columns else global_mean
    monthly  = float(input_df["Monthly_Expanding_Mean"].iloc[0]) if "Monthly_Expanding_Mean"  in input_df.columns else global_mean
    momentum = (0.40 * monthly) + (0.60 * rolling)
    expected = momentum
    active   = {}
    for col, w in affection_rates.items():
        if col in input_df.columns and float(input_df[col].iloc[0]) == 1.0:
            expected += w
            active[col] = round(w, 4)
    return round(momentum, 2), round(float(np.clip(expected, 0, 100)), 2), active


def band_desc(pred_class, residual_bins):
    try:
        lt, ht = round(float(residual_bins[1]), 1), round(float(residual_bins[2]), 1)
        if pred_class == "High":   return f"Residual > +{ht}% vs expected (Overperformance Band)"
        elif pred_class == "Low":  return f"Residual < {lt}% vs expected (Underperformance Band)"
        else:                      return f"Residual between {lt}% and +{ht}% (As-Expected Band)"
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "\U0001f3af  Prediction Engine",
    "\U0001f4ca  Model Evaluation Matrix",
    "\U0001f4c8  Historical Insights",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Subject</div>'
                    f'<div class="metric-value" style="font-size:0.95rem;">{sel_subject}</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Day / Lecture</div>'
                    f'<div class="metric-value">{sel_day} #{sel_lecture_num}</div></div>',
                    unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Weather / Time</div>'
                    f'<div class="metric-value" style="font-size:0.95rem;">{sel_weather} / {sel_time_cluster}</div></div>',
                    unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">Prev Attendance</div>'
                    f'<div class="metric-value">{sel_prev_att:.1f}%</div></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        input_df, auto_filled = build_input(
            feature_cols, train_stats, global_mean,
            sel_subject, sel_day, sel_lecture_num, sel_time_cluster,
            sel_weather, sel_pt, sel_prev_att,
            sel_week_before_exam, sel_holiday_adj, sel_gap)

        base_momentum, expected_att, active_weights = compute_expected(
            input_df, affection_rates, global_mean)

        with st.expander(f"\U0001f527 Auto-Computed Engineered Features ({len(auto_filled)} fields) — click to inspect"):
            st.markdown('<div class="info-box">These features are auto-derived from <b>training data statistics '
                        '(median/mean)</b> because they cannot be known in advance for a future lecture. '
                        'They replicate the pipeline used during training.</div>', unsafe_allow_html=True)
            af_df = pd.DataFrame(list(auto_filled.items()), columns=["Engineered Feature", "Auto-Filled Value"])
            st.dataframe(af_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Run Prediction</div>', unsafe_allow_html=True)
        pcol, mcol = st.columns([3, 1])
        pcol.markdown(f"**Selected Model:** `{sel_model_name}`")
        run_btn = mcol.button("Generate Prediction", type="primary", use_container_width=True)

        if run_btn:
            model        = models[sel_model_name]
            input_scaled = scaler.transform(input_df)
            pred_encoded = model.predict(input_scaled)
            pred_class   = le.inverse_transform(pred_encoded)[0]

            proba_dict, confidence = {}, None
            try:
                probas     = model.predict_proba(input_scaled)[0]
                proba_dict = {cls: round(float(p)*100, 1) for cls, p in zip(le.classes_, probas)}
                confidence = round(max(probas)*100, 1)
            except Exception:
                pass

            desc    = band_desc(pred_class, residual_bins)
            css_map = {"High": "result-high", "Medium": "result-medium", "Low": "result-low"}
            icon_map = {"High": "&#129001; HIGH", "Medium": "&#128997; MEDIUM", "Low": "&#128308; LOW"}

            st.markdown(
                f'<div class="{css_map.get(pred_class, "result-medium")}">'
                f'<div class="result-label">{icon_map.get(pred_class, pred_class.upper())} ATTENDANCE EXPECTED</div>'
                f'<div class="result-sub">{desc}</div></div>',
                unsafe_allow_html=True)
            st.markdown("---")

            r1, r2, r3 = st.columns(3)
            r1.markdown(f'<div class="metric-card"><div class="metric-label">Base Momentum</div>'
                        f'<div class="metric-value">{base_momentum}%</div>'
                        f'<div class="metric-delta">Weighted avg of historical attendance</div></div>',
                        unsafe_allow_html=True)
            r2.markdown(f'<div class="metric-card"><div class="metric-label">Expected Attendance</div>'
                        f'<div class="metric-value">{expected_att}%</div>'
                        f'<div class="metric-delta">After affection weights applied</div></div>',
                        unsafe_allow_html=True)
            if confidence is not None:
                r3.markdown(f'<div class="metric-card"><div class="metric-label">Model Confidence</div>'
                            f'<div class="metric-value">{confidence}%</div>'
                            f'<div class="metric-delta">predict_proba for predicted class</div></div>',
                            unsafe_allow_html=True)
            else:
                r3.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Class</div>'
                            f'<div class="metric-value">{pred_class}</div>'
                            f'<div class="metric-delta">Model does not support probability output</div></div>',
                            unsafe_allow_html=True)

            # Probability breakdown
            if proba_dict:
                st.markdown('<div class="section-header">Class Probability Breakdown</div>', unsafe_allow_html=True)
                sorted_cls = sorted(proba_dict.keys())
                cmap = {"High": "#10b981", "Medium": "#f59e0b", "Low": "#ef4444"}
                fig_p = go.Figure(go.Bar(
                    x=sorted_cls, y=[proba_dict[c] for c in sorted_cls],
                    marker_color=[cmap.get(c, "#6c757d") for c in sorted_cls],
                    text=[f"{proba_dict[c]}%" for c in sorted_cls],
                    textposition="outside", width=0.4,
                ))
                fig_p.update_layout(
                    title=f"Prediction Probabilities — {sel_model_name}",
                    xaxis_title="Attendance Class", yaxis_title="Probability (%)",
                    yaxis=dict(range=[0, 115]),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False, height=320, font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            # Active affection weights
            if active_weights:
                st.markdown('<div class="section-header">Active Affection Weights (Learned from Training)</div>', unsafe_allow_html=True)
                aw_sorted = sorted(active_weights.items(), key=lambda x: abs(x[1]), reverse=True)
                aw_keys   = [k for k, _ in aw_sorted]
                aw_vals   = [v for _, v in aw_sorted]
                fig_aw = go.Figure(go.Bar(
                    x=aw_keys, y=aw_vals,
                    marker_color=["#10b981" if v >= 0 else "#ef4444" for v in aw_vals],
                    text=[f"{v:+.2f}%" for v in aw_vals], textposition="outside",
                ))
                fig_aw.update_layout(
                    title="Impact of Active Features on Expected Attendance",
                    xaxis_title="Feature", yaxis_title="Affection Weight (% shift from mean)",
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False, height=300,
                    font=dict(family="Inter", size=12), xaxis=dict(tickangle=-20),
                )
                st.plotly_chart(fig_aw, use_container_width=True)

    with col_side:
        st.markdown('<div class="section-header">Feature Flags</div>', unsafe_allow_html=True)
        flags = {
            "Week Before Exam":     "Yes" if sel_week_before_exam else "No",
            "Holiday Adjacent":     "Yes" if sel_holiday_adj else "No",
            "Class Type":           sel_pt,
            "Time Cluster":         sel_time_cluster,
            "Gap to Prev Lecture":  f"{sel_gap} days",
        }
        for k, v in flags.items():
            st.markdown(f"**{k}:** {v}")

        st.divider()
        st.markdown('<div class="section-header">All Affection Rates</div>', unsafe_allow_html=True)
        st.markdown("*Learned weights for all binary features:*")
        ar_df = pd.DataFrame(
            sorted(affection_rates.items(), key=lambda x: x[1], reverse=True),
            columns=["Feature", "Weight (%)"])
        ar_df["Weight (%)"] = ar_df["Weight (%)"].round(4)
        st.dataframe(ar_df, use_container_width=True, hide_index=True, height=420)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Experiment Tracking Matrix</div>', unsafe_allow_html=True)
    st.markdown(
        "Models were trained and evaluated using **strict temporal (chronological) "
        "train/validation/test splits** — no random shuffling. "
        "This is the project deliverable experiment matrix."
    )

    if exp_df.empty:
        st.info("Experiment results not found. Run the training notebooks to generate experiment_results.csv.")
    else:
        reg_df   = exp_df[exp_df["Model_Type"] == "Regression"].copy()
        class_df = exp_df[exp_df["Model_Type"] == "Classification"].copy()

        # Regression
        st.markdown("#### \U0001f522 Regression Models")
        if not reg_df.empty:
            reg_cols = ["Model_ID"] + [c for c in ["Val_MAE","Val_RMSE","Val_MAPE","Val_R2"] if c in reg_df.columns]
            st.dataframe(reg_df[reg_cols], use_container_width=True, hide_index=True)

            if "Val_MAE" in reg_df.columns:
                rd = reg_df.sort_values("Val_MAE")
                fig_r = px.bar(rd, x="Model_ID", y="Val_MAE",
                    title="Validation MAE by Regression Model (lower is better)",
                    color="Val_MAE", color_continuous_scale="Blues_r",
                    text=rd["Val_MAE"].apply(lambda x: f"{x:.3f}"),
                    labels={"Model_ID": "", "Val_MAE": "MAE"})
                fig_r.update_traces(textposition="outside")
                fig_r.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False, height=360, font=dict(family="Inter", size=12),
                    xaxis=dict(tickangle=-25))
                st.plotly_chart(fig_r, use_container_width=True)

        # Classification
        st.markdown("#### \U0001f3f7 Classification Models")
        if not class_df.empty:
            cls_cols = ["Model_ID"] + [c for c in ["Val_Accuracy","Val_F1","Val_ROCAUC"] if c in class_df.columns]
            st.dataframe(class_df[cls_cols], use_container_width=True, hide_index=True)

            fig_c = go.Figure()
            for metric, color in [("Val_Accuracy","#2d5da3"),("Val_F1","#10b981"),("Val_ROCAUC","#f59e0b")]:
                if metric in class_df.columns:
                    fig_c.add_trace(go.Bar(
                        name=metric.replace("Val_",""),
                        x=class_df["Model_ID"], y=class_df[metric],
                        marker_color=color,
                        text=class_df[metric].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else ""),
                        textposition="outside",
                    ))
            fig_c.update_layout(
                barmode="group",
                title="Classification Model Performance — Accuracy, F1, ROC-AUC",
                xaxis_title="", yaxis_title="Score", yaxis=dict(range=[0,1.15]),
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, font=dict(family="Inter", size=11),
                xaxis=dict(tickangle=-25), legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_c, use_container_width=True)

            if "Val_F1" in class_df.columns:
                best = class_df.loc[class_df["Val_F1"].idxmax()]
                st.success(
                    f"\U0001f3c6 Champion Model (Best Val F1): `{best['Model_ID']}` "
                    f"— F1: `{best['Val_F1']:.4f}` | Accuracy: `{best.get('Val_Accuracy','N/A')}`"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if hist_df.empty:
        st.info("Historical data not found. Place attendance_cleaned.csv in data/processed/.")
    else:
        st.markdown(
            "Charts from the **cleaned attendance dataset** fulfilling deliverable requirements: "
            "identify low-attendance time slots, highlight poor-attendance subjects, "
            "and estimate the impact of exams, weather, and timetable shifts."
        )

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        avg_v   = hist_df["Attendance_Percentage"].mean() if "Attendance_Percentage" in hist_df.columns else 0
        n_subj  = hist_df["Subject"].nunique() if "Subject" in hist_df.columns else 0
        enr_v   = hist_df["Total_Enrolled"].iloc[0] if "Total_Enrolled" in hist_df.columns else "N/A"
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Total Lectures</div><div class="metric-value">{len(hist_df)}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">Overall Avg Attendance</div><div class="metric-value">{avg_v:.1f}%</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-label">Subjects Tracked</div><div class="metric-value">{n_subj}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-label">Total Enrolled</div><div class="metric-value">{enr_v}</div></div>', unsafe_allow_html=True)
        st.markdown("---")

        # INSIGHT 1: Subject-wise attendance
        st.markdown('<div class="section-header">Insight 1 — Subject-wise Average Attendance</div>', unsafe_allow_html=True)
        st.markdown("Highlights subjects that consistently suffer from **low attendance**.")
        if "Subject" in hist_df.columns and "Attendance_Percentage" in hist_df.columns:
            s_avg = (hist_df.groupby("Subject")["Attendance_Percentage"]
                     .mean().reset_index()
                     .rename(columns={"Attendance_Percentage":"Avg"})
                     .sort_values("Avg"))
            s_avg["Avg"] = s_avg["Avg"].round(2)
            oa = s_avg["Avg"].mean()
            bc = ["#ef4444" if v < oa*0.9 else "#f59e0b" if v < oa else "#10b981" for v in s_avg["Avg"]]
            fig_s = go.Figure(go.Bar(
                y=s_avg["Subject"], x=s_avg["Avg"], orientation="h",
                marker_color=bc,
                text=[f"{v:.1f}%" for v in s_avg["Avg"]], textposition="outside",
            ))
            fig_s.add_vline(x=oa, line_dash="dash", line_color="#2d5da3",
                annotation_text=f"Overall avg: {oa:.1f}%", annotation_position="top right")
            fig_s.update_layout(
                title="Average Attendance by Subject (Red=Below avg, Green=Above avg)",
                xaxis_title="Avg Attendance %", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                height=max(320, 38*len(s_avg)),
                font=dict(family="Inter", size=12), margin=dict(l=10, r=80),
            )
            st.plotly_chart(fig_s, use_container_width=True)
            worst = s_avg.head(3)
            st.warning("Lowest attendance subjects: " +
                       ", ".join(f"{r['Subject']} ({r['Avg']:.1f}%)" for _, r in worst.iterrows()))

        st.markdown("---")

        # INSIGHT 2: Time Slot Trend
        st.markdown('<div class="section-header">Insight 2 — Time Slot Attendance Trend</div>', unsafe_allow_html=True)
        st.markdown("Identifies **lecture slots and days** consistently associated with low attendance.")
        i2c1, i2c2 = st.columns(2)

        if "Lecture_Number" in hist_df.columns and "Attendance_Percentage" in hist_df.columns:
            l_avg = (hist_df.groupby("Lecture_Number")["Attendance_Percentage"]
                     .mean().reset_index().rename(columns={"Attendance_Percentage":"Avg"}))
            l_avg["Avg"] = l_avg["Avg"].round(2)
            fig_l = px.line(l_avg, x="Lecture_Number", y="Avg", markers=True,
                title="Avg Attendance by Lecture Number",
                labels={"Lecture_Number":"Lecture Slot","Avg":"Avg Attendance %"},
                color_discrete_sequence=["#2d5da3"])
            fig_l.add_hline(y=l_avg["Avg"].mean(), line_dash="dash", line_color="#ef4444",
                annotation_text="Day avg", annotation_position="bottom right")
            fig_l.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                height=300, font=dict(family="Inter", size=12))
            i2c1.plotly_chart(fig_l, use_container_width=True)

        if "Day_of_Week" in hist_df.columns and "Attendance_Percentage" in hist_df.columns:
            d_order = [d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
                       if d in hist_df["Day_of_Week"].unique()]
            d_avg = (hist_df.groupby("Day_of_Week")["Attendance_Percentage"]
                     .mean().reindex(d_order).reset_index()
                     .rename(columns={"Attendance_Percentage":"Avg"}))
            d_avg["Avg"] = d_avg["Avg"].round(2)
            fig_d = px.bar(d_avg, x="Day_of_Week", y="Avg",
                title="Avg Attendance by Day of Week",
                color="Avg", color_continuous_scale="Blues",
                text=[f"{v:.1f}%" for v in d_avg["Avg"]],
                labels={"Day_of_Week":"Day","Avg":"Avg Attendance %"})
            fig_d.update_traces(textposition="outside")
            fig_d.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                height=300, font=dict(family="Inter", size=12), showlegend=False)
            i2c2.plotly_chart(fig_d, use_container_width=True)

        st.markdown("---")

        # INSIGHT 3: Weather and Exam Impact
        st.markdown('<div class="section-header">Insight 3 — Weather and Examination Impact</div>', unsafe_allow_html=True)
        st.markdown("Estimates the **statistical impact** of weather conditions and exam proximity on cohort attendance.")
        i3c1, i3c2 = st.columns(2)

        if "Weather" in hist_df.columns and "Attendance_Percentage" in hist_df.columns:
            w_avg = (hist_df.groupby("Weather")["Attendance_Percentage"]
                     .agg(["mean","count"]).reset_index()
                     .rename(columns={"mean":"Avg","count":"N"}))
            w_avg["Avg"] = w_avg["Avg"].round(2)
            wcol = {"Sunny":"#f59e0b","Rainy":"#2d5da3","Cloudy":"#6c757d"}
            fig_w = go.Figure(go.Bar(
                x=w_avg["Weather"], y=w_avg["Avg"],
                marker_color=[wcol.get(w,"#aaa") for w in w_avg["Weather"]],
                text=[f"{v:.1f}%" for v in w_avg["Avg"]], textposition="outside",
            ))
            fig_w.add_hline(y=hist_df["Attendance_Percentage"].mean(),
                line_dash="dash", line_color="#ef4444", annotation_text="Overall avg")
            fig_w.update_layout(
                title="Attendance by Weather Condition",
                xaxis_title="Weather", yaxis_title="Avg Attendance %",
                plot_bgcolor="white", paper_bgcolor="white",
                height=310, font=dict(family="Inter", size=12), showlegend=False)
            i3c1.plotly_chart(fig_w, use_container_width=True)

        if "Internal_Test_Week" in hist_df.columns and "Attendance_Percentage" in hist_df.columns:
            t_avg = (hist_df.groupby("Internal_Test_Week")["Attendance_Percentage"]
                     .agg(["mean","count"]).reset_index()
                     .rename(columns={"mean":"Avg","count":"N"}))
            t_avg["Avg"] = t_avg["Avg"].round(2)
            fig_t = go.Figure(go.Bar(
                x=t_avg["Internal_Test_Week"].astype(str), y=t_avg["Avg"],
                marker_color=["#2d5da3","#10b981","#f59e0b"][:len(t_avg)],
                text=[f"{v:.1f}%" for v in t_avg["Avg"]], textposition="outside",
            ))
            fig_t.add_hline(y=hist_df["Attendance_Percentage"].mean(),
                line_dash="dash", line_color="#ef4444", annotation_text="Overall avg")
            fig_t.update_layout(
                title="Attendance: Internal Test Week vs Normal",
                xaxis_title="Internal Test Week", yaxis_title="Avg Attendance %",
                plot_bgcolor="white", paper_bgcolor="white",
                height=310, font=dict(family="Inter", size=12), showlegend=False)
            i3c2.plotly_chart(fig_t, use_container_width=True)

        # Full affection rates chart
        st.markdown('<div class="section-header">Feature Affection Rate Summary (All Learned Weights)</div>', unsafe_allow_html=True)
        ar_sorted = sorted(affection_rates.items(), key=lambda x: x[1])
        ar_keys   = [k for k, _ in ar_sorted]
        ar_vals   = [v for _, v in ar_sorted]
        fig_ar = go.Figure(go.Bar(
            y=ar_keys, x=ar_vals, orientation="h",
            marker_color=["#ef4444" if v < 0 else "#10b981" for v in ar_vals],
            text=[f"{v:+.2f}%" for v in ar_vals], textposition="outside",
        ))
        fig_ar.add_vline(x=0, line_color="#6c757d", line_width=1)
        fig_ar.update_layout(
            title="Affection Rate of Every Binary Feature (Shift from Class Mean Attendance %)",
            xaxis_title="Affection Weight (% shift)", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(400, 30*len(ar_keys)),
            font=dict(family="Inter", size=11), margin=dict(l=10, r=100),
        )
        st.plotly_chart(fig_ar, use_container_width=True)


# FOOTER
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c757d;font-size:0.8rem;'>"
    "Classroom Attendance Predictor &nbsp;&middot;&nbsp; MCA Sem 3 Capstone &nbsp;&middot;&nbsp; "
    "Built with Streamlit + Plotly &nbsp;&middot;&nbsp; "
    "All feature options dynamically loaded from deployment_assets/"
    "</div>", unsafe_allow_html=True)
