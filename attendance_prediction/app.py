"""
Classroom Attendance Prediction Dashboard
==========================================
ML-powered dashboard — all predictions driven by trained models.
Feature weights are LEARNED from data, not hardcoded.
Fulfills Section 6.1 & 7 deliverables.
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
# CUSTOM CSS — Light Academic Theme
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0f111a; }

.header-banner {
  background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
  border-radius: 12px; padding: 24px 32px; margin-bottom: 24px;
  color: white; box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
.header-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.header-banner p  { margin: 6px 0 0; font-size: 1rem; opacity: 0.88; font-weight: 300; }

.metric-card {
  background: #1e293b; border-radius: 10px; padding: 20px 24px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25); border-left: 4px solid #3b82f6;
  margin-bottom: 12px;
}
.metric-card .metric-label {
  font-size: 0.75rem; color: #94a3b8; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.metric-card .metric-value {
  font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin-top: 4px;
}
.metric-card .metric-delta {
  font-size: 0.8rem; color: #94a3b8; margin-top: 2px;
}

.result-high   { background: rgba(16, 185, 129, 0.1); border:2px solid #10b981; border-radius:12px; padding:24px; text-align:center; color: #34d399; }
.result-medium { background: rgba(245, 158, 11, 0.1); border:2px solid #f59e0b; border-radius:12px; padding:24px; text-align:center; color: #fbbf24; }
.result-low    { background: rgba(239, 68, 68, 0.1); border:2px solid #ef4444; border-radius:12px; padding:24px; text-align:center; color: #f87171; }
.result-label  { font-size:1.5rem; font-weight:700; margin-bottom:4px; }
.result-sub    { font-size:0.9rem; opacity:0.8; }

[data-testid="stSidebar"] { background-color: #151923; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
  font-weight: 600; color: #e2e8f0; font-size: 0.85rem;
}

.section-header {
  font-size: 1.1rem; font-weight: 700; color: #bfdbfe;
  border-bottom: 2px solid #3b82f6; padding-bottom: 6px; margin: 20px 0 12px;
}
.info-box {
  background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; border-radius: 8px;
  padding: 14px 18px; font-size: 0.85rem; color: #93c5fd; margin: 10px 0;
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
    rates     = {} # removed
    feat_cols = joblib.load(os.path.join(ASSETS_DIR, "feature_columns.pkl"))
    att_bins  = joblib.load(os.path.join(ASSETS_DIR, "attendance_bins.pkl"))
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
    return scaler, le, rates, feat_cols, att_bins, g_mean, models


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
    opts = {
        "subjects": [], "days": [], "weathers": [],
        "time_clusters": [], "practical_types": [],
    }
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
def compute_subject_averages():
    """Per-subject average attendance from cleaned historical data."""
    path = os.path.join(DATA_DIR, "attendance_cleaned.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Subject" not in df.columns or "Attendance_Percentage" not in df.columns:
        return {}
    return df.groupby("Subject")["Attendance_Percentage"].mean().to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
try:
    scaler, le, affection_rates, feature_cols, attendance_bins, global_mean, models = load_assets()
except Exception as e:
    st.error(f"Failed to load backend assets: {e}")
    st.info("Run export_pipeline.py first to generate deployment_assets/.")
    st.stop()

options      = derive_options(tuple(feature_cols))
hist_df      = load_historical()
exp_df       = load_experiments()
subject_avgs = compute_subject_averages()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>&#127891; Classroom Attendance Predictor</h1>
  <p>ML-powered dashboard &nbsp;&middot;&nbsp; MCA Sem 3 Cohort &nbsp;&middot;&nbsp;
     Models learn what matters — no hardcoded weights.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FULLY DYNAMIC INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### \U0001f4dd Lecture Details")
    st.markdown("*All options derived dynamically from training data.*")
    st.divider()

    import datetime
    sel_date = st.date_input("Date", value=datetime.date.today(), help="Select the date of the lecture")

    sel_subject = st.selectbox(
        "Subject", options=options["subjects"],
        help="Subjects seen during model training")

    sel_day = st.selectbox(
        "Day of Week", options=options["days"],
        help="Days in the model encoding")

    sel_lecture_num = st.slider(
        "Lecture Number", min_value=1, max_value=5, value=2,
        help="1=First (8:30 AM), 2-4=Mid-day, 5=After lunch (1:30 PM)")

    # Auto-detect time cluster from lecture number
    time_opts = options["time_clusters"] if options["time_clusters"] else ["Morning"]
    auto_time = "Afternoon" if sel_lecture_num == 5 else "Morning"
    if auto_time not in time_opts:
        auto_time = time_opts[0]
    sel_time_cluster = st.selectbox(
        "Time of Day", options=time_opts,
        index=time_opts.index(auto_time),
        help="Auto-detected from lecture number")

    sel_weather = st.selectbox(
        "Weather Condition", options=options["weathers"],
        help="Weather conditions seen during training")

    pt_opts = options["practical_types"] if options["practical_types"] else ["Theory"]
    sel_pt = st.selectbox(
        "Class Type", options=pt_opts,
        help="Practical or Theory")

    sel_prev_att = st.slider(
        "Previous Lecture Attendance %",
        min_value=5.0, max_value=100.0, value=20.0, step=0.5,
        help="Attendance % of the immediately preceding lecture")

    sel_week_before_exam = st.toggle(
        "Week Before Examination", value=False,
        help="Is this in the week before an internal exam?")
    sel_holiday_adj = st.toggle(
        "Day Adjacent to Holiday", value=False,
        help="Monday or Friday adjacent to a holiday?")

    sel_gap = st.slider(
        "Gap Since Previous Lecture (days)",
        min_value=0.0, max_value=7.0, value=1.0, step=0.25)

    st.divider()
    st.markdown("### \U0001f916 Model Settings")
    sel_model_name = st.selectbox(
        "Prediction Algorithm", options=list(models.keys()))
    st.divider()
    st.markdown(
        '<div class="info-box">The model itself decides which features '
        'matter most. No structural weights are imposed — predictions '
        'come purely from learned patterns in the data.</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_subject_rolling_avg(subject_name):
    """Get historical average for a subject; fuzzy-match if exact name differs."""
    if subject_name in subject_avgs:
        return subject_avgs[subject_name]
    for k, v in subject_avgs.items():
        if subject_name.lower() in k.lower() or k.lower() in subject_name.lower():
            return v
    return global_mean


def build_input(feat_cols, global_mean,
                sel_date, sel_subject, sel_day, sel_lecture_num, sel_time_cluster,
                sel_weather, sel_pt, sel_prev_att,
                sel_week_before_exam, sel_holiday_adj, sel_gap):
    """
    Build feature vector matching the trained model's expected columns
    from feature_columns.pkl exactly.
    """
    row = {col: 0.0 for col in feat_cols}
    auto_filled = {}
    subj_avg = get_subject_rolling_avg(sel_subject)

    # ── Direct user inputs ──────────────────────────────────────────────
    direct = {
        "Lecture_Number":               float(sel_lecture_num),
        # Handle both possible column names for previous attendance
        "Previous_Lecture_Attendance":   float(sel_prev_att),
        "Previous_Lecture_Attendance_Pct": float(sel_prev_att),
        "Last_Session_Attendance_Pct": float(sel_prev_att),
        # Handle both possible column names for exam flag
        "Week_Before_Exam":             1.0 if sel_week_before_exam else 0.0,
        "Week_Before_Exam_Flag":        1.0 if sel_week_before_exam else 0.0,
        "Is_Holiday_Adjacent":          1.0 if sel_holiday_adj else 0.0,
        "Gap_Since_Previous_Lecture_Days": float(sel_gap),
        # One-hot encoded categoricals
        f"Subject_{sel_subject}":       1.0,
        f"Day_of_Week_{sel_day}":       1.0,
        f"Weather_{sel_weather}":       1.0,
        f"Time_of_Day_{sel_time_cluster}": 1.0,
        f"Practical_Theory_{sel_pt}":   1.0,
    }
    for feat, val in direct.items():
        if feat in row:
            row[feat] = val

    # ── Auto-fill engineered features ───────────────────────────────────
    semester_start = datetime.date(sel_date.year, 7, 1) if sel_date.month >= 7 else datetime.date(sel_date.year, 1, 1)
    day_of_sem = (sel_date - semester_start).days + 1
    
    auto_defaults = {
        "Total_Enrolled":           204.0,
        "Week_Number":              sel_date.isocalendar()[1],
        "Day_of_Semester":          day_of_sem if day_of_sem > 0 else 1.0,
        "Month":                    sel_date.month,
        "Days_Since_Last_Holiday":  3.0,
        "Consecutive_Lecture_Count": float(sel_lecture_num),
        "Is_Consecutive":           1.0 if sel_lecture_num > 1 else 0.0,
        "Is_Post_Lunch_Class":      1.0 if sel_lecture_num == 5 else 0.0,
        "Semester":                 3.0,
        # Use subject-level rolling avg — the model decides how much to weight it
        "Rolling_Avg_3":            subj_avg,
        "Rolling_Avg_3_Lectures":   subj_avg,
        # Use global mean as monthly baseline
        "Monthly_Avg_Attendance":   global_mean,
        "Monthly_Expanding_Mean":   global_mean,
    }
    for feat, val in auto_defaults.items():
        if feat in row:
            row[feat] = float(val)
            auto_filled[feat] = round(float(val), 4)

    return pd.DataFrame([row], columns=feat_cols), auto_filled


def compute_expected(input_df, affection_rates, global_mean):
    return 0, 0, {}


def band_desc(pred_class, attendance_bins):
    """Describe the predicted attendance band."""
    try:
        lt = round(float(attendance_bins[1]), 1)
        ht = round(float(attendance_bins[2]), 1)
        if pred_class == "Low":
            return f"Below {lt}%"
        elif pred_class == "High":
            return f"Above {ht}%"
        else:
            return f"Between {lt}% and {ht}%"
    except Exception:
        return ""


def get_feature_importances(model, model_name, feature_names):
    """
    Extract feature importances from the model.
    Works for tree-based models and logistic regression.
    Returns a dict of {feature_name: importance} or None.
    """
    imp = None
    try:
        # Tree-based models: feature_importances_
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(feature_names, model.feature_importances_))
        # Linear models: coef_ (use absolute mean across classes)
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if coef.ndim > 1:
                coef = coef.mean(axis=0)
            imp = dict(zip(feature_names, coef))
        # Naive Bayes: use variance of theta_ across classes
        elif hasattr(model, "theta_"):
            variance = np.var(model.theta_, axis=0)
            imp = dict(zip(feature_names, variance))
    except Exception:
        pass
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f3af  Prediction Engine",
    "\U0001f4ca  Experiment Matrix",
    "\U0001f4c8  Historical Insights",
    "\U0001f50d  Model Explainability",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown('<div class="section-header">Input Summary</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f'<div class="metric-card"><div class="metric-label">Subject</div>'
            f'<div class="metric-value" style="font-size:0.95rem;">'
            f'{sel_subject}</div></div>', unsafe_allow_html=True)
        c2.markdown(
            f'<div class="metric-card"><div class="metric-label">Day / Lecture</div>'
            f'<div class="metric-value">{sel_day} #{sel_lecture_num}</div></div>',
            unsafe_allow_html=True)
        c3.markdown(
            f'<div class="metric-card"><div class="metric-label">Weather / Time</div>'
            f'<div class="metric-value" style="font-size:0.95rem;">'
            f'{sel_weather} / {sel_time_cluster}</div></div>',
            unsafe_allow_html=True)
        c4.markdown(
            f'<div class="metric-card"><div class="metric-label">Prev Attendance</div>'
            f'<div class="metric-value">{sel_prev_att:.1f}%</div></div>',
            unsafe_allow_html=True)

        st.markdown("---")

        # Build feature vector
        input_df, auto_filled = build_input(
            feature_cols, global_mean,
            sel_date, sel_subject, sel_day, sel_lecture_num, sel_time_cluster,
            sel_weather, sel_pt, sel_prev_att,
            sel_week_before_exam, sel_holiday_adj, sel_gap)

        # Compute expected attendance from learned affection rates
        base_momentum, expected_att, active_weights = compute_expected(
            input_df, affection_rates, global_mean)

        # Auto-filled features expander
        with st.expander(
            f"\U0001f527 Auto-Computed Engineered Features "
            f"({len(auto_filled)} fields) — click to inspect"
        ):
            st.markdown(
                '<div class="info-box">These features are auto-derived from '
                '<b>training data statistics</b> and your inputs. They '
                'replicate the pipeline used during training.</div>',
                unsafe_allow_html=True)
            af_df = pd.DataFrame(
                list(auto_filled.items()),
                columns=["Engineered Feature", "Auto-Filled Value"])
            st.dataframe(af_df, width="stretch", hide_index=True)

        st.markdown(
            '<div class="section-header">Run Prediction</div>',
            unsafe_allow_html=True)
        pcol, mcol = st.columns([3, 1])
        pcol.markdown(f"**Selected Model:** `{sel_model_name}`")
        run_btn = mcol.button(
            "Generate Prediction", type="primary",
            width="stretch")

        if run_btn:
            model        = models[sel_model_name]
            input_scaled = scaler.transform(input_df)
            pred_encoded = model.predict(input_scaled)
            pred_class   = le.inverse_transform(pred_encoded)[0]

            proba_dict, confidence = {}, None
            try:
                probas     = model.predict_proba(input_scaled)[0]
                proba_dict = {
                    cls: round(float(p) * 100, 1)
                    for cls, p in zip(le.classes_, probas)
                }
                confidence = round(max(probas) * 100, 1)
            except Exception:
                pass

            desc    = band_desc(pred_class, attendance_bins)
            css_map = {
                "High": "result-high",
                "Medium": "result-medium",
                "Low": "result-low",
            }
            icon_map = {
                "High": "&#129001; HIGH",
                "Medium": "&#128997; MEDIUM",
                "Low": "&#128308; LOW",
            }

            st.markdown(
                f'<div class="{css_map.get(pred_class, "result-medium")}">'
                f'<div class="result-label">'
                f'{icon_map.get(pred_class, pred_class.upper())} '
                f'ATTENDANCE EXPECTED</div>'
                f'<div class="result-sub">{desc}</div></div>',
                unsafe_allow_html=True)
            st.markdown("---")

            # Key metrics
            if confidence is not None:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Model Confidence</div>'
                    f'<div class="metric-value">{confidence}%</div>'
                    f'<div class="metric-delta">'
                    f'predict_proba for predicted class'
                    f'</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Predicted Class</div>'
                    f'<div class="metric-value">{pred_class}</div>'
                    f'<div class="metric-delta">'
                    f'Model does not support probability output'
                    f'</div></div>', unsafe_allow_html=True)

            # Probability breakdown chart
            if proba_dict:
                st.markdown(
                    '<div class="section-header">'
                    'Class Probability Breakdown</div>',
                    unsafe_allow_html=True)
                sorted_cls = sorted(proba_dict.keys())
                cmap = {
                    "High": "#10b981",
                    "Medium": "#f59e0b",
                    "Low": "#ef4444",
                }
                fig_p = go.Figure(go.Bar(
                    x=sorted_cls,
                    y=[proba_dict[c] for c in sorted_cls],
                    marker_color=[cmap.get(c, "#6c757d") for c in sorted_cls],
                    text=[f"{proba_dict[c]}%" for c in sorted_cls],
                    textposition="outside", width=0.4,
                ))
                fig_p.update_layout(
                    title=f"Prediction Probabilities — {sel_model_name}",
                    xaxis_title="Attendance Class",
                    yaxis_title="Probability (%)",
                    yaxis=dict(range=[0, 115]),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, height=320,
                    font=dict(family="Inter", size=13),
                )
                st.plotly_chart(fig_p, width="stretch")

            # Feature importance for THIS prediction
            importances = get_feature_importances(
                model, sel_model_name, feature_cols)
            if importances:
                st.markdown(
                    '<div class="section-header">'
                    'What the Model Considers Important</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    "Feature importances extracted directly from "
                    f"**{sel_model_name}** — the model itself learned "
                    "these weights from the data.")
                imp_sorted = sorted(
                    importances.items(), key=lambda x: x[1], reverse=True)
                top_n = imp_sorted[:15]
                imp_keys = [k for k, _ in top_n]
                imp_vals = [v for _, v in top_n]
                fig_imp = go.Figure(go.Bar(
                    y=list(reversed(imp_keys)),
                    x=list(reversed(imp_vals)),
                    orientation="h",
                    marker_color="#2d5da3",
                    text=[f"{v:.4f}" for v in reversed(imp_vals)],
                    textposition="outside",
                ))
                fig_imp.update_layout(
                    title=f"Top 15 Feature Importances — {sel_model_name}",
                    xaxis_title="Importance Score",
                    yaxis_title="",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    height=max(350, 28 * len(top_n)),
                    font=dict(family="Inter", size=11),
                    margin=dict(l=10, r=80),
                )
                st.plotly_chart(fig_imp, width="stretch")

    with col_side:
        st.markdown(
            '<div class="section-header">Feature Flags</div>',
            unsafe_allow_html=True)
        flags = {
            "Week Before Exam":    "Yes" if sel_week_before_exam else "No",
            "Holiday Adjacent":    "Yes" if sel_holiday_adj else "No",
            "Class Type":          sel_pt,
            "Time Cluster":        sel_time_cluster,
            "Gap to Prev Lecture": f"{sel_gap} days",
            "Lecture Position":    (
                "First of day" if sel_lecture_num == 1
                else "After lunch" if sel_lecture_num == 5
                else "Mid-day"),
        }
        for k, v in flags.items():
            st.markdown(f"**{k}:** {v}")

        st.divider()
        # ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="section-header">Experiment Tracking Matrix</div>',
        unsafe_allow_html=True)
    st.markdown(
        "Models were trained and evaluated using **strict temporal "
        "(chronological) train/validation/test splits** — no random "
        "shuffling. This is the project deliverable experiment matrix.")

    if exp_df.empty:
        st.info(
            "Experiment results not found. Run training notebooks to "
            "generate experiment_results.csv.")
    else:
        reg_df   = exp_df[exp_df["Model_Type"] == "Regression"].copy()
        class_df = exp_df[exp_df["Model_Type"] == "Classification"].copy()

        # ── Regression ──────────────────────────────────────────────────
        st.markdown("#### \U0001f522 Regression Models")
        if not reg_df.empty:
            reg_cols = ["Model_ID"] + [
                c for c in ["Val_MAE", "Val_RMSE", "Val_MAPE", "Val_R2"]
                if c in reg_df.columns]
            st.dataframe(
                reg_df[reg_cols],
                width="stretch", hide_index=True)

            if "Val_MAE" in reg_df.columns:
                rd = reg_df.sort_values("Val_MAE")
                fig_r = px.bar(
                    rd, x="Model_ID", y="Val_MAE",
                    title="Validation MAE by Regression Model (lower = better)",
                    color="Val_MAE", color_continuous_scale="Blues_r",
                    text=rd["Val_MAE"].apply(lambda x: f"{x:.3f}"),
                    labels={"Model_ID": "", "Val_MAE": "MAE"})
                fig_r.update_traces(textposition="outside")
                fig_r.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, height=360,
                    font=dict(family="Inter", size=12),
                    xaxis=dict(tickangle=-25))
                st.plotly_chart(fig_r, width="stretch")

        # ── Classification ──────────────────────────────────────────────
        st.markdown("#### \U0001f3f7 Classification Models")
        if not class_df.empty:
            cls_cols = ["Model_ID"] + [
                c for c in ["Val_Accuracy", "Val_F1", "Val_ROCAUC"]
                if c in class_df.columns]
            st.dataframe(
                class_df[cls_cols],
                width="stretch", hide_index=True)

            fig_c = go.Figure()
            for metric, color in [
                ("Val_Accuracy", "#2d5da3"),
                ("Val_F1", "#10b981"),
                ("Val_ROCAUC", "#f59e0b"),
            ]:
                if metric in class_df.columns:
                    fig_c.add_trace(go.Bar(
                        name=metric.replace("Val_", ""),
                        x=class_df["Model_ID"],
                        y=class_df[metric],
                        marker_color=color,
                        text=class_df[metric].apply(
                            lambda x: f"{x:.3f}" if pd.notnull(x) else ""),
                        textposition="outside",
                    ))
            fig_c.update_layout(
                barmode="group",
                title="Classification Performance — Accuracy, F1, ROC-AUC",
                xaxis_title="", yaxis_title="Score",
                yaxis=dict(range=[0, 1.15]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=380, font=dict(family="Inter", size=11),
                xaxis=dict(tickangle=-25),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_c, width="stretch")

            if "Val_F1" in class_df.columns:
                best = class_df.loc[class_df["Val_F1"].idxmax()]
                st.success(
                    f"\U0001f3c6 Champion Model (Best Val F1): "
                    f"`{best['Model_ID']}` — F1: `{best['Val_F1']:.4f}` "
                    f"| Accuracy: `{best.get('Val_Accuracy', 'N/A')}`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if hist_df.empty:
        st.info(
            "Historical data not found. Place attendance_cleaned.csv "
            "in data/processed/.")
    else:
        st.markdown(
            "Charts fulfilling **Section 6.1 deliverables**: identify "
            "low-attendance time slots, highlight poor-attendance subjects, "
            "and estimate impact of exams, weather, and timetable shifts.")

        # ── Summary metrics ─────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        avg_v  = (hist_df["Attendance_Percentage"].mean()
                  if "Attendance_Percentage" in hist_df.columns else 0)
        n_subj = (hist_df["Subject"].nunique()
                  if "Subject" in hist_df.columns else 0)
        enr_v  = (hist_df["Total_Enrolled"].iloc[0]
                  if "Total_Enrolled" in hist_df.columns else "N/A")
        n_days = (hist_df["Date"].nunique()
                  if "Date" in hist_df.columns else 0)
        m1.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Total Lectures</div>'
            f'<div class="metric-value">{len(hist_df)}</div></div>',
            unsafe_allow_html=True)
        m2.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Overall Avg Attendance</div>'
            f'<div class="metric-value">{avg_v:.1f}%</div></div>',
            unsafe_allow_html=True)
        m3.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Subjects Tracked</div>'
            f'<div class="metric-value">{n_subj}</div></div>',
            unsafe_allow_html=True)
        m4.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Days Observed</div>'
            f'<div class="metric-value">{n_days}</div></div>',
            unsafe_allow_html=True)
        st.markdown("---")

        # ── INSIGHT 1: Subject-wise attendance ──────────────────────────
        st.markdown(
            '<div class="section-header">'
            'Insight 1 — Subjects with Poor Attendance</div>',
            unsafe_allow_html=True)
        st.markdown(
            "Highlights subjects that consistently suffer from **low "
            "attendance** (Deliverable 6.1.3).")
        if ("Subject" in hist_df.columns
                and "Attendance_Percentage" in hist_df.columns):
            s_avg = (hist_df.groupby("Subject")["Attendance_Percentage"]
                     .mean().reset_index()
                     .rename(columns={"Attendance_Percentage": "Avg"})
                     .sort_values("Avg"))
            s_avg["Avg"] = s_avg["Avg"].round(2)
            oa = s_avg["Avg"].mean()
            bc = [
                "#ef4444" if v < oa * 0.9
                else "#f59e0b" if v < oa
                else "#10b981"
                for v in s_avg["Avg"]]
            fig_s = go.Figure(go.Bar(
                y=s_avg["Subject"], x=s_avg["Avg"], orientation="h",
                marker_color=bc,
                text=[f"{v:.1f}%" for v in s_avg["Avg"]],
                textposition="outside",
            ))
            fig_s.add_vline(
                x=oa, line_dash="dash", line_color="#2d5da3",
                annotation_text=f"Overall avg: {oa:.1f}%",
                annotation_position="top right")
            fig_s.update_layout(
                title="Average Attendance by Subject "
                      "(Red=Below avg, Green=Above avg)",
                xaxis_title="Avg Attendance %", yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=max(320, 38 * len(s_avg)),
                font=dict(family="Inter", size=12),
                margin=dict(l=10, r=80),
            )
            st.plotly_chart(fig_s, width="stretch")
            worst = s_avg.head(3)
            st.warning(
                "\u26a0\ufe0f Lowest attendance subjects: " +
                ", ".join(
                    f"**{r['Subject']}** ({r['Avg']:.1f}%)"
                    for _, r in worst.iterrows()))

        st.markdown("---")

        # ── INSIGHT 2: Time Slot Trend ──────────────────────────────────
        st.markdown(
            '<div class="section-header">'
            'Insight 2 — Low-Attendance Time Slots</div>',
            unsafe_allow_html=True)
        st.markdown(
            "Identifies **lecture slots and days** with consistently low "
            "attendance (Deliverable 6.1.2).")
        i2c1, i2c2 = st.columns(2)

        if ("Lecture_Number" in hist_df.columns
                and "Attendance_Percentage" in hist_df.columns):
            l_avg = (hist_df.groupby("Lecture_Number")
                     ["Attendance_Percentage"]
                     .mean().reset_index()
                     .rename(columns={"Attendance_Percentage": "Avg"}))
            l_avg["Avg"] = l_avg["Avg"].round(2)
            fig_l = px.line(
                l_avg, x="Lecture_Number", y="Avg", markers=True,
                title="Avg Attendance by Lecture Slot",
                labels={
                    "Lecture_Number": "Lecture Slot",
                    "Avg": "Avg Attendance %"},
                color_discrete_sequence=["#2d5da3"])
            fig_l.add_hline(
                y=l_avg["Avg"].mean(), line_dash="dash",
                line_color="#ef4444",
                annotation_text="Day avg",
                annotation_position="bottom right")
            fig_l.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=300, font=dict(family="Inter", size=12))
            i2c1.plotly_chart(fig_l, width="stretch")

        if ("Day_of_Week" in hist_df.columns
                and "Attendance_Percentage" in hist_df.columns):
            d_order = [
                d for d in [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday"]
                if d in hist_df["Day_of_Week"].unique()]
            d_avg = (hist_df.groupby("Day_of_Week")
                     ["Attendance_Percentage"]
                     .mean().reindex(d_order).reset_index()
                     .rename(columns={"Attendance_Percentage": "Avg"}))
            d_avg["Avg"] = d_avg["Avg"].round(2)
            fig_d = px.bar(
                d_avg, x="Day_of_Week", y="Avg",
                title="Avg Attendance by Day of Week",
                color="Avg", color_continuous_scale="Blues",
                text=[f"{v:.1f}%" for v in d_avg["Avg"]],
                labels={"Day_of_Week": "Day", "Avg": "Avg Attendance %"})
            fig_d.update_traces(textposition="outside")
            fig_d.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=300, font=dict(family="Inter", size=12),
                showlegend=False)
            i2c2.plotly_chart(fig_d, width="stretch")

        st.markdown("---")

        # ── INSIGHT 3: Weather and Exam Impact ──────────────────────────
        st.markdown(
            '<div class="section-header">'
            'Insight 3 — Weather & Examination Impact</div>',
            unsafe_allow_html=True)
        st.markdown(
            "Estimates the **statistical impact** of weather conditions "
            "and exam proximity on cohort attendance (Deliverable 6.1.4).")
        i3c1, i3c2 = st.columns(2)

        if ("Weather" in hist_df.columns
                and "Attendance_Percentage" in hist_df.columns):
            w_avg = (hist_df.groupby("Weather")["Attendance_Percentage"]
                     .agg(["mean", "count"]).reset_index()
                     .rename(columns={"mean": "Avg", "count": "N"}))
            w_avg["Avg"] = w_avg["Avg"].round(2)
            wcol = {
                "Sunny": "#f59e0b", "Rainy": "#2d5da3",
                "Cloudy": "#6c757d"}
            fig_w = go.Figure(go.Bar(
                x=w_avg["Weather"], y=w_avg["Avg"],
                marker_color=[
                    wcol.get(w, "#aaa") for w in w_avg["Weather"]],
                text=[f"{v:.1f}%" for v in w_avg["Avg"]],
                textposition="outside",
            ))
            fig_w.add_hline(
                y=hist_df["Attendance_Percentage"].mean(),
                line_dash="dash", line_color="#ef4444",
                annotation_text="Overall avg")
            fig_w.update_layout(
                title="Attendance by Weather Condition",
                xaxis_title="Weather",
                yaxis_title="Avg Attendance %",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=310, font=dict(family="Inter", size=12),
                showlegend=False)
            i3c1.plotly_chart(fig_w, width="stretch")

        if ("Internal_Test_Week" in hist_df.columns
                and "Attendance_Percentage" in hist_df.columns):
            t_avg = (hist_df.groupby("Internal_Test_Week")
                     ["Attendance_Percentage"]
                     .agg(["mean", "count"]).reset_index()
                     .rename(columns={"mean": "Avg", "count": "N"}))
            t_avg["Avg"] = t_avg["Avg"].round(2)
            fig_t = go.Figure(go.Bar(
                x=t_avg["Internal_Test_Week"].astype(str),
                y=t_avg["Avg"],
                marker_color=[
                    "#2d5da3", "#10b981", "#f59e0b"][:len(t_avg)],
                text=[f"{v:.1f}%" for v in t_avg["Avg"]],
                textposition="outside",
            ))
            fig_t.add_hline(
                y=hist_df["Attendance_Percentage"].mean(),
                line_dash="dash", line_color="#ef4444",
                annotation_text="Overall avg")
            fig_t.update_layout(
                title="Attendance: Internal Test Week vs Normal",
                xaxis_title="Internal Test Week",
                yaxis_title="Avg Attendance %",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=310, font=dict(family="Inter", size=12),
                showlegend=False)
            i3c2.plotly_chart(fig_t, width="stretch")

        st.markdown("---")

        # ── INSIGHT 4: Optimal Lecture Timing Heatmap ───────────────────
        st.markdown(
            '<div class="section-header">'
            'Insight 4 — Optimal Lecture Timing Recommendations</div>',
            unsafe_allow_html=True)
        st.markdown(
            "Suggests **optimal lecture timings** based on historical "
            "peak attendance hours (Extension 6.2.3).")

        if all(c in hist_df.columns for c in [
            "Subject", "Lecture_Number", "Attendance_Percentage"
        ]):
            pivot = hist_df.pivot_table(
                values="Attendance_Percentage",
                index="Subject", columns="Lecture_Number",
                aggfunc="mean").round(1)
            fig_heat = px.imshow(
                pivot, text_auto=".1f",
                color_continuous_scale="RdYlGn",
                labels=dict(
                    x="Lecture Slot", y="Subject",
                    color="Avg Att %"),
                title="Subject × Lecture Slot Heatmap "
                      "(Green = High Attendance)")
            fig_heat.update_layout(
                height=max(350, 40 * len(pivot)),
                font=dict(family="Inter", size=12),
                margin=dict(l=10))
            st.plotly_chart(fig_heat, width="stretch")

            st.markdown("**\U0001f4cb Scheduling Recommendations:**")
            for subj in pivot.index:
                best_slot  = pivot.loc[subj].idxmax()
                worst_slot = pivot.loc[subj].idxmin()
                best_val   = pivot.loc[subj, best_slot]
                worst_val  = pivot.loc[subj, worst_slot]
                st.markdown(
                    f"- **{subj}**: Best at Lecture {best_slot} "
                    f"({best_val:.1f}%), Worst at Lecture "
                    f"{worst_slot} ({worst_val:.1f}%)")

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="section-header">'
        'Model Feature Importance Comparison</div>',
        unsafe_allow_html=True)
    st.markdown(
        "Compare what **each model** considers important. No weights are "
        "imposed — these importances are **learned directly** from the "
        "training data by each algorithm.")

    # Gather importances from all models
    all_importances = {}
    for mname, mobj in models.items():
        imp = get_feature_importances(mobj, mname, feature_cols)
        if imp is not None:
            all_importances[mname] = imp

    if all_importances:
        # Model selector
        sel_explain = st.selectbox(
            "Select Model to Explain",
            options=list(all_importances.keys()),
            key="explain_model")

        imp_data = all_importances[sel_explain]
        imp_sorted = sorted(
            imp_data.items(), key=lambda x: x[1], reverse=True)

        # Full feature importance chart
        imp_keys = [k for k, _ in imp_sorted]
        imp_vals = [v for _, v in imp_sorted]
        fig_full = go.Figure(go.Bar(
            y=list(reversed(imp_keys)),
            x=list(reversed(imp_vals)),
            orientation="h",
            marker=dict(
                color=list(reversed(imp_vals)),
                colorscale="Blues",
            ),
            text=[f"{v:.4f}" for v in reversed(imp_vals)],
            textposition="outside",
        ))
        fig_full.update_layout(
            title=f"All Feature Importances — {sel_explain}",
            xaxis_title="Importance Score", yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=max(500, 25 * len(imp_keys)),
            font=dict(family="Inter", size=11),
            margin=dict(l=10, r=80),
        )
        st.plotly_chart(fig_full, width="stretch")

        # Top 10 comparison across models
        st.markdown("---")
        st.markdown(
            '<div class="section-header">'
            'Top 10 Features Across All Models</div>',
            unsafe_allow_html=True)
        st.markdown(
            "Each model discovers **different patterns** in the data. "
            "This comparison shows which features consistently rank "
            "as important.")

        # Build comparison dataframe
        comp_rows = []
        for mname, imp_dict in all_importances.items():
            top10 = sorted(
                imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for rank, (feat, score) in enumerate(top10, 1):
                comp_rows.append({
                    "Model": mname,
                    "Rank": rank,
                    "Feature": feat,
                    "Importance": round(score, 4),
                })
        comp_df = pd.DataFrame(comp_rows)

        if not comp_df.empty:
            fig_comp = px.bar(
                comp_df, x="Importance", y="Feature",
                color="Model", orientation="h",
                barmode="group",
                title="Feature Importance Comparison (Top 10 per Model)",
                height=500,
            )
            fig_comp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=11),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_comp, width="stretch")

        # Consensus features
        st.markdown("---")
        st.markdown(
            '<div class="section-header">'
            'Consensus: Features Important Across All Models</div>',
            unsafe_allow_html=True)

        # Count how many models have each feature in their top 10
        feat_consensus = {}
        for mname, imp_dict in all_importances.items():
            top10 = sorted(
                imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, _ in top10:
                feat_consensus[feat] = feat_consensus.get(feat, 0) + 1

        consensus_sorted = sorted(
            feat_consensus.items(), key=lambda x: x[1], reverse=True)
        total_models = len(all_importances)

        for feat, count in consensus_sorted:
            bar_pct = count / total_models * 100
            color = (
                "#10b981" if count == total_models
                else "#f59e0b" if count >= total_models / 2
                else "#6c757d")
            st.markdown(
                f"<div style='margin:4px 0;'>"
                f"<span style='font-weight:600;width:350px;"
                f"display:inline-block;'>{feat}</span>"
                f"<span style='background:{color};color:white;"
                f"padding:2px 8px;border-radius:4px;font-size:0.8rem;'>"
                f"{count}/{total_models} models</span></div>",
                unsafe_allow_html=True)
    else:
        st.info(
            "No feature importances could be extracted from the loaded "
            "models. This typically works with tree-based models "
            "(XGBoost, Random Forest, Decision Tree) and Logistic "
            "Regression.")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c757d;font-size:0.8rem;'>"
    "Classroom Attendance Predictor &nbsp;&middot;&nbsp; "
    "MCA Sem 3 Capstone &nbsp;&middot;&nbsp; "
    "Built with Streamlit + Plotly &nbsp;&middot;&nbsp; "
    "All predictions driven by ML models — no hardcoded weights &nbsp;"
    "&middot;&nbsp; Features dynamically loaded from deployment_assets/"
    "</div>", unsafe_allow_html=True)
