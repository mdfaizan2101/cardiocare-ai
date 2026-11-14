import io
import os
import tempfile
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ==========================================
# PAGE CONFIGURATION & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="CardioCare AI | Clinical Decision Support",
    page_icon="🫀",
    layout="wide"
)

HISTORY_FILE = "cardiocare_history.csv"

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Biological Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG Findings",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "Exercise ST Depression",
    "slope": "Peak Exercise ST Slope",
    "ca": "Major Coronary Vessels",
    "thal": "Myocardial Perfusion"
}

PRESETS = {
    "default": {
        "patient_id": "Patient-001", "age": 54, "sex": 1.0, "cp": 4.0,
        "trestbps": 130, "chol": 246, "fbs": 0.0, "restecg": 0.0,
        "thalach": 150, "exang": 0.0, "oldpeak": 1.0, "slope": 2.0,
        "ca": 0.0, "thal": 3.0
    },
    "low_risk": {
        "patient_id": "Sample-LowRisk-01", "age": 42, "sex": 0.0, "cp": 3.0,
        "trestbps": 118, "chol": 195, "fbs": 0.0, "restecg": 0.0,
        "thalach": 172, "exang": 0.0, "oldpeak": 0.0, "slope": 1.0,
        "ca": 0.0, "thal": 3.0
    },
    "high_risk": {
        "patient_id": "Sample-HighRisk-02", "age": 63, "sex": 1.0, "cp": 4.0,
        "trestbps": 150, "chol": 286, "fbs": 0.0, "restecg": 2.0,
        "thalach": 108, "exang": 1.0, "oldpeak": 2.5, "slope": 2.0,
        "ca": 3.0, "thal": 7.0
    }
}

if "latest_input" not in st.session_state:
    st.session_state.latest_input = None
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"
if "form_defaults" not in st.session_state:
    st.session_state.form_defaults = PRESETS["default"].copy()


def navigate_to(page_name: str):
    st.session_state.nav_page = page_name


def load_preset(preset_key: str):
    st.session_state.form_defaults = PRESETS[preset_key].copy()


def format_clinical_value(key: str, val: float) -> str:
    """Translates raw numerical model codes into readable clinical text for reports."""
    mappings = {
        "sex": {1.0: "Male", 0.0: "Female"},
        "cp": {
            1.0: "Typical Angina",
            2.0: "Atypical Angina",
            3.0: "Non-Anginal Chest Pain",
            4.0: "Asymptomatic"
        },
        "fbs": {0.0: "Normal (<= 120 mg/dl)", 1.0: "Elevated (> 120 mg/dl)"},
        "restecg": {
            0.0: "Normal",
            1.0: "ST-T Wave Abnormality",
            2.0: "Left Ventricular Hypertrophy"
        },
        "exang": {0.0: "No", 1.0: "Yes"},
        "slope": {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"},
        "thal": {
            3.0: "Normal Blood Flow",
            6.0: "Fixed Defect",
            7.0: "Reversible Defect"
        }
    }
    if key in mappings:
        return mappings[key].get(float(val), str(val))
    if key == "age":
        return f"{int(val)} years"
    if key == "trestbps":
        return f"{int(val)} mm Hg"
    if key == "chol":
        return f"{int(val)} mg/dl"
    if key == "thalach":
        return f"{int(val)} bpm"
    if key == "oldpeak":
        return f"{val:.1f} mm"
    if key == "ca":
        return f"{int(val)} Vessels"
    return str(val)


# ==========================================
# CACHED MODEL & EXPLAINER RESOURCES
# ==========================================
@st.cache_resource
def load_pipeline():
    return joblib.load("heart_disease_pipeline.pkl")


@st.cache_resource
def get_shap_explainer(_classifier):
    if SHAP_AVAILABLE:
        return shap.TreeExplainer(_classifier)
    return None


pipeline = load_pipeline()
explainer = get_shap_explainer(pipeline.named_steps["classifier"])

# ==========================================
# MINIMALIST SIDEBAR & CARD HOVER CSS
# ==========================================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

.sidebar-brand {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sidebar-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0.6rem 0.9rem !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease-in-out !important;
    box-shadow: none !important;
    margin-bottom: 0.15rem !important;
}

[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: #475569 !important;
    border: 1px solid transparent !important;
}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    transform: translateX(4px);
    border-color: #e2e8f0 !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #fff1f2 !important;
    color: #e11d48 !important;
    border: 1px solid #fecdd3 !important;
    border-left: 4px solid #e11d48 !important;
    font-weight: 600 !important;
}

.nav-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.15s ease;
    height: 100%;
}
.nav-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 6px 16px -4px rgba(15, 23, 42, 0.06);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED COMPUTATION & CHART HELPERS
# ==========================================
@st.cache_data
def generate_recommendations(data: dict, prob: float) -> list:
    recs = []
    if prob >= 0.50:
        recs.append(("Urgent Clinical Evaluation", "High Priority", 
                     "Schedule a comprehensive cardiology consultation, including a diagnostic 12-lead ECG and echocardiogram."))
    if data["trestbps"] >= 130:
        recs.append(("Blood Pressure Management", "Vitals", 
                     f"Resting blood pressure is elevated ({int(data['trestbps'])} mm Hg). Adopt a low-sodium DASH diet (<2,300 mg/day) and monitor blood pressure twice daily."))
    if data["chol"] >= 200:
        recs.append(("Lipid Profile Optimization", "Metabolic", 
                     f"Serum cholesterol is {int(data['chol'])} mg/dl. Increase soluble fiber, reduce saturated fats, and discuss a fasting lipid panel with your physician."))
    if data["fbs"] == 1.0:
        recs.append(("Glycemic Control", "Metabolic", 
                     "Fasting blood sugar > 120 mg/dl indicates potential insulin resistance. Recommend HbA1c screening and reduced refined carbohydrates."))
    if data["exang"] == 1.0 or data["oldpeak"] >= 1.0:
        recs.append(("Exertional Ischemia Precautions", "Cardiac Symptoms", 
                     "Exercise-induced angina or ST segment depression detected. Avoid unmonitored high-intensity exertion until cleared by a cardiologist."))
    if data["thalach"] < 120:
        recs.append(("Aerobic Conditioning", "Fitness", 
                     "Lower peak heart rate observed. Discuss supervised cardiac rehabilitation or graded low-impact aerobic exercise."))
    if not recs:
        recs.append(("Preventive Maintenance", "Routine", 
                     "Clinical biomarkers are within favorable ranges. Maintain 150 minutes/week of moderate aerobic activity and annual wellness screenings."))
    return recs


@st.cache_data
def render_global_importance_png() -> bytes:
    """Renders global Random Forest feature importance across the 13 clinical attributes."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    raw_names = [name.split("__")[1].split("_")[0] for name in feature_names]
    clean_names = [FEATURE_LABELS.get(col, col) for col in raw_names]

    imp_df = pd.DataFrame({"Feature": clean_names, "Importance": classifier.feature_importances_})
    agg_df = imp_df.groupby("Feature", as_index=False)["Importance"].sum()
    agg_df = agg_df.sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
    ax.barh(agg_df["Feature"], agg_df["Importance"], color="#0f172a")
    ax.set_xlabel("Relative Predictive Importance (Gini Gain)")
    ax.set_title("Global Model Feature Importance")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data
def render_shap_png(data_dict: dict) -> bytes:
    """Computes local SHAP values once per patient profile and caches the PNG image."""
    input_df = pd.DataFrame([data_dict])
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_X = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    raw_names = [name.split("__")[1].split("_")[0] for name in feature_names]
    clean_names = [FEATURE_LABELS.get(col, col) for col in raw_names]

    if SHAP_AVAILABLE and explainer is not None:
        shap_vals = explainer.shap_values(transformed_X)
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            vals = shap_vals[0, :, 1]
        else:
            vals = shap_vals[0]
    else:
        vals = classifier.feature_importances_ * transformed_X[0]

    contrib_df = pd.DataFrame({"Feature": clean_names, "Impact": vals})
    agg_df = contrib_df.groupby("Feature", as_index=False)["Impact"].sum()
    agg_df = agg_df.reindex(agg_df["Impact"].abs().sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    colors = ["#e11d48" if v > 0 else "#0284c7" for v in agg_df["Impact"]]
    ax.barh(agg_df["Feature"], agg_df["Impact"], color=colors)
    ax.axvline(0, color="#334155", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Impact on Risk (Red = Increases Risk, Blue = Decreases Risk)")
    ax.set_title("Patient Feature Contribution (Explainable AI)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data
def create_pdf_report(patient_name: str, data: dict, prob: float, pred: int, recs: list, timestamp: str, shap_png_bytes: bytes) -> bytes:
    """Generates a clinical PDF report with decoded clinical values and embedded SHAP plot."""
    pdf = FPDF()
    pdf.add_page()

    # Header Banner
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "  CardioCare AI - Clinical Diagnostic Report", ln=True, fill=True)

    pdf.set_text_color(15, 23, 42)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {timestamp}   |   Patient Reference: {patient_name}", ln=True)
    pdf.ln(2)

    # Verdict Box
    if pred == 1:
        pdf.set_fill_color(255, 241, 242)
        pdf.set_text_color(190, 18, 60)
        status = "HIGH CARDIOVASCULAR RISK DETECTED"
    else:
        pdf.set_fill_color(240, 253, 244)
        pdf.set_text_color(21, 128, 61)
        status = "LOW CARDIOVASCULAR RISK DETECTED"

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"  {status}  (Estimated Risk Probability: {prob:.1%})", ln=True, fill=True)
    pdf.set_text_color(15, 23, 42)
    pdf.ln(4)

    # Decoded Patient Clinical Parameters (2-column layout)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Submitted Clinical Biomarkers:", ln=True)
    pdf.set_font("Helvetica", "", 9.5)

    items = list(data.items())
    for i in range(0, len(items), 2):
        k1, v1 = items[i]
        text1 = f"{FEATURE_LABELS.get(k1, k1)}: {format_clinical_value(k1, v1)}"
        pdf.cell(95, 6, text1, border=0)
        if i + 1 < len(items):
            k2, v2 = items[i + 1]
            text2 = f"{FEATURE_LABELS.get(k2, k2)}: {format_clinical_value(k2, v2)}"
            pdf.cell(95, 6, text2, border=0, ln=True)
        else:
            pdf.ln(6)

    pdf.ln(4)

    # Embed SHAP Plot Image
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Explainable AI - Patient Feature Contribution:", ln=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(shap_png_bytes)
        tmp_path = tmp_file.name
    try:
        pdf.image(tmp_path, x=15, w=175)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    pdf.ln(3)

    # Recommendations
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Personalized Clinical & Lifestyle Guidance:", ln=True)
    pdf.set_font("Helvetica", "", 9.5)
    for title, category, detail in recs:
        pdf.multi_cell(0, 5.5, f"* [{category}] {title}: {detail}")
        pdf.ln(1)

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4.5, "Disclaimer: CardioCare AI is an educational clinical decision-support prototype and does not replace formal medical diagnosis or treatment.")
    return bytes(pdf.output())


# ==========================================
# MINIMALIST SIDEBAR NAVIGATION
# ==========================================
MENU_ITEMS = [
    "Home",
    "Clinical Assessment",
    "Clinical Report",
    "Health Guide",
    "History"
]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🫀 CardioCare AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Cardiovascular Health Platform</div>', unsafe_allow_html=True)

    for item in MENU_ITEMS:
        is_active = (st.session_state.nav_page == item)
        st.button(
            item,
            key=f"nav_{item}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            on_click=navigate_to,
            args=(item,)
        )

    st.divider()
    st.caption("**Medical Disclaimer:** Educational risk stratification only. Not a substitute for clinical diagnosis.")

page = st.session_state.nav_page

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.title("CardioCare AI")
    st.write("Explainable machine learning platform for cardiovascular risk screening, clinical reporting, and preventive health guidance.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", "0.9535", "High Discrimination")
    m2.metric("Sensitivity", "96.0%", "Minimizes False Negatives")
    m3.metric("Test Accuracy", "90.2%", "61 Unseen Patients")
    m4.metric("Biomarkers", "13 Inputs", "Clinical Benchmark")

    st.write("")
    st.subheader("Quick Access")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="nav-card">
            <h4>Clinical Assessment</h4>
            <p style="color:#64748b; font-size:0.9rem;">Enter patient vitals, ECG findings, and exercise stress parameters across a 3-step guided form.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Clinical Assessment →",
            key="home_to_assess",
            use_container_width=True,
            on_click=navigate_to,
            args=("Clinical Assessment",)
        )

    with c2:
        st.markdown("""
        <div class="nav-card">
            <h4>Clinical Report</h4>
            <p style="color:#64748b; font-size:0.9rem;">Review individual risk probabilities, local feature explanations, and downloadable PDF reports.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Clinical Report →",
            key="home_to_report",
            use_container_width=True,
            on_click=navigate_to,
            args=("Clinical Report",)
        )

    with c3:
        st.markdown("""
        <div class="nav-card">
            <h4>Health Guide</h4>
            <p style="color:#64748b; font-size:0.9rem;">Explore target cardiovascular biomarker ranges and track daily preventive habits.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Health Guide →",
            key="home_to_guide",
            use_container_width=True,
            on_click=navigate_to,
            args=("Health Guide",)
        )

    st.divider()
    st.subheader("Model Validation & Global Explainability")
    b_col1, b_col2 = st.columns([1.1, 1.2])
    with b_col1:
        st.markdown("**5-Fold Stratified Cross-Validation & Test Benchmarks**")
        bench_df = pd.DataFrame([
            {"Algorithm": "Tuned Random Forest (Production)", "CV Recall": "78.3%", "CV ROC-AUC": "0.8972", "Test ROC-AUC": "0.9535"},
            {"Algorithm": "Logistic Regression", "CV Recall": "78.3%", "CV ROC-AUC": "0.9025", "Test ROC-AUC": "0.9380"},
            {"Algorithm": "XGBoost Classifier", "CV Recall": "77.5%", "CV ROC-AUC": "0.8597", "Test ROC-AUC": "0.9140"}
        ])
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.caption("Prioritizing sensitivity (96% on unseen test set) ensures minimal false negatives during clinical triage.")
    with b_col2:
        st.image(render_global_importance_png(), use_container_width=True)

# ==========================================
# PAGE 2: CLINICAL ASSESSMENT
# ==========================================
elif page == "Clinical Assessment":
    st.title("Clinical Assessment")
    st.write("Complete the three sections below or load a sample clinical profile to test the prediction engine. Hover over **( ? )** for medical definitions.")

    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    with p_col1:
        st.button("Load Low-Risk Sample", on_click=load_preset, args=("low_risk",), use_container_width=True)
    with p_col2:
        st.button("Load High-Risk Sample", on_click=load_preset, args=("high_risk",), use_container_width=True)

    d = st.session_state.form_defaults

    with st.form("clinical_assessment_form"):
        patient_id = st.text_input(
            "Patient Identifier / Reference",
            value=d["patient_id"],
            help="Enter a reference code or name to label the generated clinical report."
        )

        tab1, tab2, tab3 = st.tabs([
            "1. Vitals & Demographics",
            "2. Cardiac Symptoms & ECG",
            "3. Exercise Stress & Perfusion"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input(
                    "Age (years)",
                    min_value=20, max_value=100, value=int(d["age"]), step=1,
                    help="Patient's chronological age in years. Cardiovascular risk naturally increases with age."
                )
                sex_opts = [(1.0, "Male"), (0.0, "Female")]
                sex_idx = 0 if d["sex"] == 1.0 else 1
                sex = st.selectbox(
                    "Biological Sex",
                    options=sex_opts,
                    index=sex_idx,
                    format_func=lambda x: x[1],
                    help="Biological sex of the patient, used to account for physiological differences in cardiac risk baselines."
                )[0]
                trestbps = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=80, max_value=220, value=int(d["trestbps"]), step=1,
                    help="Systolic blood pressure measured while sitting quietly at rest. Normal adult resting blood pressure is below 120 mm Hg."
                )
            with col2:
                chol = st.number_input(
                    "Serum Cholesterol (mg/dl)",
                    min_value=100, max_value=600, value=int(d["chol"]), step=1,
                    help="Total cholesterol level in the blood (including LDL and HDL). Desirable total cholesterol is below 200 mg/dl."
                )
                fbs_opts = [(0.0, "Normal (<= 120 mg/dl)"), (1.0, "Elevated (> 120 mg/dl)")]
                fbs_idx = 0 if d["fbs"] == 0.0 else 1
                fbs = st.selectbox(
                    "Fasting Blood Sugar",
                    options=fbs_opts,
                    index=fbs_idx,
                    format_func=lambda x: x[1],
                    help="Blood glucose measured after at least 8 hours of fasting. Levels above 120 mg/dl may indicate diabetes or insulin resistance."
                )[0]

        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                cp_opts = [
                    (1.0, "Typical Angina (Exertional Chest Pressure)"),
                    (2.0, "Atypical Angina"),
                    (3.0, "Non-Anginal Chest Pain"),
                    (4.0, "Asymptomatic (Silent / No Chest Pain)")
                ]
                cp_idx = [1.0, 2.0, 3.0, 4.0].index(d["cp"])
                cp = st.selectbox(
                    "Chest Pain Type",
                    options=cp_opts,
                    index=cp_idx,
                    format_func=lambda x: x[1],
                    help="Pattern of chest discomfort: Typical Angina is substernal pressure triggered by exertion; Asymptomatic represents silent ischemia without chest pain."
                )[0]
                restecg_opts = [
                    (0.0, "Normal"),
                    (1.0, "ST-T Wave Abnormality"),
                    (2.0, "Left Ventricular Hypertrophy")
                ]
                restecg_idx = [0.0, 1.0, 2.0].index(d["restecg"])
                restecg = st.selectbox(
                    "Resting ECG Findings",
                    options=restecg_opts,
                    index=restecg_idx,
                    format_func=lambda x: x[1],
                    help="Electrocardiogram recorded at rest. Left Ventricular Hypertrophy indicates thickening of the heart's main pumping chamber."
                )[0]
            with col4:
                thalach = st.number_input(
                    "Maximum Heart Rate Achieved (bpm)",
                    min_value=60, max_value=220, value=int(d["thalach"]), step=1,
                    help="Highest heart rate (in beats per minute) reached during peak exercise stress testing."
                )
                exang_opts = [(0.0, "No"), (1.0, "Yes")]
                exang_idx = 0 if d["exang"] == 0.0 else 1
                exang = st.selectbox(
                    "Exercise-Induced Angina",
                    options=exang_opts,
                    index=exang_idx,
                    format_func=lambda x: x[1],
                    help="Indicates whether physical exercise on a treadmill or stationary bike triggers chest pain or tightness."
                )[0]

        with tab3:
            col5, col6 = st.columns(2)
            with col5:
                oldpeak = st.number_input(
                    "Exercise ST Depression (mm)",
                    min_value=0.0, max_value=7.0, value=float(d["oldpeak"]), step=0.1,
                    help="Measures how far the ST segment on the ECG drops below baseline during exercise compared to rest (in millimeters)."
                )
                slope_opts = [
                    (1.0, "Upsloping"),
                    (2.0, "Flat"),
                    (3.0, "Downsloping")
                ]
                slope_idx = [1.0, 2.0, 3.0].index(d["slope"])
                slope = st.selectbox(
                    "Peak Exercise ST Slope",
                    options=slope_opts,
                    index=slope_idx,
                    format_func=lambda x: x[1],
                    help="Direction of the ECG ST segment during peak exercise. Flat or Downsloping curves are stronger indicators of coronary ischemia."
                )[0]
            with col6:
                ca_opts = [0.0, 1.0, 2.0, 3.0]
                ca_idx = ca_opts.index(d["ca"])
                ca = st.selectbox(
                    "Major Coronary Vessels Narrowed",
                    options=ca_opts,
                    index=ca_idx,
                    format_func=lambda x: f"{int(x)} Vessels",
                    help="Number of major coronary arteries (0 to 3) showing significant narrowing under fluoroscopy contrast dye."
                )
                thal_opts = [
                    (3.0, "Normal Blood Flow"),
                    (6.0, "Fixed Defect (Prior Scar Tissue)"),
                    (7.0, "Reversible Defect (Stress-Induced Ischemia)")
                ]
                thal_idx = [3.0, 6.0, 7.0].index(d["thal"])
                thal = st.selectbox(
                    "Myocardial Perfusion Status",
                    options=thal_opts,
                    index=thal_idx,
                    format_func=lambda x: x[1],
                    help="Nuclear blood-flow scan result: Normal Blood Flow, Fixed Defect (prior scar tissue), or Reversible Defect (stress-induced ischemia)."
                )[0]

            st.divider()
            consent_save = st.toggle("Save anonymized assessment to local History", value=True)

        submitted = st.form_submit_button("Generate Clinical Report", type="primary", use_container_width=True)

    if submitted:
        raw_dict = {
            'age': float(age), 'sex': float(sex), 'cp': float(cp),
            'trestbps': float(trestbps), 'chol': float(chol), 'fbs': float(fbs),
            'restecg': float(restecg), 'thalach': float(thalach), 'exang': float(exang),
            'oldpeak': float(oldpeak), 'slope': float(slope), 'ca': float(ca), 'thal': float(thal)
        }
        input_df = pd.DataFrame([raw_dict])
        pred = int(pipeline.predict(input_df)[0])
        prob = float(pipeline.predict_proba(input_df)[0][1])
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.latest_input = raw_dict
        st.session_state.latest_result = {
            "patient_id": patient_id,
            "prediction": pred,
            "probability": prob,
            "timestamp": timestamp_str
        }

        _ = render_shap_png(raw_dict)

        if consent_save:
            clean_record = {
                "Timestamp": timestamp_str,
                "Patient ID": patient_id,
                "Risk Probability": f"{prob:.1%}",
                "Verdict": "High Risk" if pred == 1 else "Low Risk",
                **{FEATURE_LABELS[k]: format_clinical_value(k, v) for k, v in raw_dict.items()}
            }
            hist_df = pd.DataFrame([clean_record])
            if os.path.exists(HISTORY_FILE):
                hist_df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
            else:
                hist_df.to_csv(HISTORY_FILE, index=False)

        st.session_state.nav_page = "Clinical Report"
        st.rerun()

# ==========================================
# PAGE 3: CLINICAL REPORT
# ==========================================
elif page == "Clinical Report":
    st.title("Clinical Report")

    if st.session_state.latest_input is None:
        st.info("No assessment has been generated yet. Click below to start a **Clinical Assessment**.")
        st.button(
            "Go to Clinical Assessment →",
            type="primary",
            on_click=navigate_to,
            args=("Clinical Assessment",)
        )
    else:
        data = st.session_state.latest_input
        res = st.session_state.latest_result
        prob = res["probability"]
        pred = res["prediction"]

        if data["trestbps"] >= 180:
            st.warning("⚠️ Resting Blood Pressure >= 180 mm Hg requires immediate clinical verification.")
        if data["chol"] >= 300:
            st.warning("⚠️ Serum Cholesterol >= 300 mg/dl noted.")

        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Estimated Risk Probability", f"{prob:.1%}")
            st.progress(prob)
            st.caption(f"Patient: **{res['patient_id']}** | {res['timestamp']}")
        with r2:
            if pred == 1:
                st.error("**High Cardiovascular Risk Detected**\n\nBiomarkers align with patterns of coronary artery disease. Immediate cardiology evaluation is advised.")
            else:
                st.success("**Low Cardiovascular Risk Detected**\n\nClinical profile indicates a low probability of obstructive coronary artery disease.")

        st.divider()

        shap_bytes = render_shap_png(data)
        recs = generate_recommendations(data, prob)

        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.subheader("Feature Contribution (Explainable AI)")
            st.image(shap_bytes, use_container_width=True)

        with col_right:
            st.subheader("Clinical Guidance")
            for title, badge, text in recs:
                st.markdown(f"**[{badge}] {title}** — {text}")

        st.divider()
        if FPDF_AVAILABLE:
            pdf_bytes = create_pdf_report(res["patient_id"], data, prob, pred, recs, res["timestamp"], shap_bytes)
            st.download_button(
                label="📄 Download Official PDF Report (with SHAP Chart)",
                data=pdf_bytes,
                file_name=f"CardioCare_Report_{res['patient_id']}.pdf",
                mime="application/pdf",
                type="primary"
            )
        else:
            report_text = f"CardioCare AI Report - {res['patient_id']}\nRisk Probability: {prob:.1%}\nParameters: {data}"
            st.download_button(
                label="Download Summary Report (.txt)",
                data=report_text,
                file_name=f"CardioCare_Report_{res['patient_id']}.txt"
            )

# ==========================================
# PAGE 4: HEALTH GUIDE
# ==========================================
elif page == "Health Guide":
    st.title("Health Guide")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Clinical Biomarker Reference")
        with st.expander("Resting Blood Pressure", expanded=True):
            st.write("Measures the force of blood against artery walls while at rest. A healthy target is below **120/80 mm Hg**. Sustained readings above 130/80 mm Hg increase arterial stiffness.")
        with st.expander("Serum Cholesterol & Fasting Blood Sugar"):
            st.write("Total serum cholesterol below **200 mg/dl** and fasting blood glucose at or below **120 mg/dl** help prevent plaque buildup inside the coronary arteries.")
        with st.expander("Exercise-Induced Angina & ST Depression"):
            st.write("Chest discomfort during exertion or a downward shift of **1.0 mm or more** in the ECG ST segment during exercise stress testing indicates that the heart muscle may not be receiving enough oxygen.")
        with st.expander("Major Coronary Vessels & Myocardial Perfusion"):
            st.write("Fluoroscopy imaging counts how many major coronary arteries (0–3) show narrowing, while myocardial perfusion scans check whether blood flow drops during stress (Reversible Defect) or reflects prior scar tissue (Fixed Defect).")

    with g2:
        st.subheader("Daily Prevention Tracker")
        c_1 = st.toggle("30+ minutes of moderate aerobic activity")
        c_2 = st.toggle("5+ servings of vegetables, fruits, and whole grains")
        c_3 = st.toggle("Sodium intake kept below 2,300 mg")
        c_4 = st.toggle("Zero tobacco / nicotine exposure")
        c_5 = st.toggle("7–8 hours of restorative sleep")
        c_6 = st.toggle("Prescribed cardiovascular medications taken on schedule")

        completed = sum([c_1, c_2, c_3, c_4, c_5, c_6])
        st.progress(completed / 6)
        st.caption(f"Daily Score: **{completed} / 6 habits completed ({completed/6:.0%})**")

# ==========================================
# PAGE 5: HISTORY
# ==========================================
elif page == "History":
    st.title("History")
    st.write("Saved local assessment records.")

    if os.path.exists(HISTORY_FILE):
        hist_df = pd.read_csv(HISTORY_FILE)
        st.dataframe(hist_df, use_container_width=True)

        col_dl, col_del = st.columns(2)
        with col_dl:
            st.download_button(
                "Export History (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="cardiocare_history.csv",
                mime="text/csv"
            )
        with col_del:
            if st.button("Clear All History", type="secondary"):
                os.remove(HISTORY_FILE)
                st.rerun()
    else:
        st.info("No saved records found.")
