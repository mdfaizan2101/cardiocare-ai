# 🫀 CardioCare AI — Explainable Cardiovascular Risk Assessment Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-crimson)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b)

**CardioCare AI** is an end-to-end clinical decision-support and health education web application that predicts cardiovascular disease risk from 13 clinical biomarkers, explains individual predictions using **SHAP (SHapley Additive exPlanations)**, and generates downloadable **PDF diagnostic reports**.

---

## 🌟 Portfolio Highlights
* **Clinical-Grade Sensitivity (96.0% Recall):** Tuned via 5-fold Stratified Cross-Validation to minimize false negatives on unseen patients, achieving a **0.9535 Test ROC-AUC** and **90.2% overall accuracy**.
* **Zero Data Leakage Architecture:** Bundles median/mode imputation (`SimpleImputer`), continuous feature scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) inside a unified Scikit-Learn `ColumnTransformer` + `RandomForestClassifier` pipeline (`heart_disease_pipeline.pkl`).
* **Patient-Level Explainable AI (SHAP):** Computes local `TreeExplainer` feature contributions in real time so clinicians and patients can see exactly which biomarkers increased or decreased risk.
* **Automated PDF Clinical Report Generation:** Exports a formatted PDF report containing decoded patient vitals, the embedded SHAP explanation chart, and evidence-based clinical recommendations.
* **One-Click Recruiter Demo Presets:** Includes built-in **Low-Risk** and **High-Risk** sample patient profiles for instant testing.

---

## 📊 Model Benchmark Summary

| Model | 5-Fold CV Accuracy | 5-Fold CV Recall | 5-Fold CV ROC-AUC | Unseen Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Tuned Random Forest (Deployed)** | **81.8%** | **78.3%** | **0.8972** | **0.9535** |
| Logistic Regression | 84.7% | 78.3% | 0.9025 | 0.9380 |
| XGBoost Classifier | 79.0% | 77.5% | 0.8597 | 0.9140 |

---

## 🚀 Quickstart (Local Execution)

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/cardiocare-ai.git
cd cardiocare-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
python -m streamlit run app.py
