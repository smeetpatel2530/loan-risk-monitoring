import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Loan Risk Monitor", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.risk-high   { background-color:#ffe0e0; padding:10px; border-radius:8px; border-left:5px solid #e74c3c; }
.risk-medium { background-color:#fff3cd; padding:10px; border-radius:8px; border-left:5px solid #f39c12; }
.risk-low    { background-color:#d4edda; padding:10px; border-radius:8px; border-left:5px solid #2ecc71; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Risk Monitoring System")
st.markdown("**Operational Risk Control Framework** — Predicts default risk and flags high-risk applicants for manual review.")
st.markdown("---")

@st.cache_resource
def load_model():
    try:
        model    = joblib.load("xgb_loan_risk_model.pkl")
        scaler   = joblib.load("scaler.pkl")
        features = joblib.load("feature_names.pkl")
        return model, scaler, features
    except:
        return None, None, None

model, scaler, features = load_model()

tab1, tab2, tab3 = st.tabs(["🔍 Single Applicant", "📊 Batch Processing", "📈 Model Insights"])

# ─── TAB 1: SINGLE APPLICANT ───
with tab1:
    st.subheader("Assess Individual Applicant Risk")
    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amount        = st.number_input("Loan Amount (₹)", 10000, 5000000, 250000, step=10000)
        annual_income      = st.number_input("Annual Income (₹)", 100000, 5000000, 600000, step=50000)
        cibil_score        = st.slider("CIBIL Score", 300, 900, 700)
        age                = st.slider("Age", 21, 65, 32)
    with col2:
        loan_tenure_months = st.selectbox("Loan Tenure (months)", [12, 24, 36, 48, 60, 84, 120])
        employment_type    = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
        loan_purpose       = st.selectbox("Loan Purpose", ["Home", "Personal", "Vehicle", "Education", "Business"])
    with col3:
        existing_emi       = st.number_input("Existing EMI (₹/month)", 0, 100000, 5000, step=1000)
        num_existing_loans = st.slider("Number of Existing Loans", 0, 10, 1)
        missed_payments_6m = st.slider("Missed Payments (Last 6M)", 0, 6, 0)
        collateral_value   = st.number_input("Collateral Value (₹)", 0, 10000000, 300000, step=50000)

    if st.button("🔍 Assess Risk", use_container_width=True, type="primary"):
        emp_map  = {"Salaried": 2, "Self-Employed": 1, "Business": 0}
        purp_map = {"Business": 0, "Education": 1, "Home": 2, "Personal": 3, "Vehicle": 4}

        dti  = loan_amount / annual_income
        eir  = existing_emi / (annual_income / 12)
        ltc  = loan_amount / (collateral_value + 1)
        rc   = int(cibil_score < 600)
        rp   = int(missed_payments_6m > 0)

        input_data = pd.DataFrame([[
            loan_amount, annual_income, cibil_score, loan_tenure_months,
            emp_map[employment_type], existing_emi, num_existing_loans,
            purp_map[loan_purpose], age, missed_payments_6m, collateral_value,
            dti, eir, ltc, rc, rp
        ]], columns=features)

        if model:
            scaled   = scaler.transform(input_data)
            risk_prob = model.predict_proba(scaled)[0][1]

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Default Probability", f"{risk_prob*100:.1f}%")
            c2.metric("CIBIL Score",          f"{cibil_score}")
            c3.metric("Debt-to-Income Ratio",  f"{dti*100:.1f}%")

            st.markdown("### Risk Assessment")
            if risk_prob >= 0.65:
                st.markdown(f'<div class="risk-high">🚨 <b>HIGH RISK — FLAG FOR MANUAL REVIEW</b><br>Default probability: {risk_prob*100:.1f}%. This applicant exceeds the risk threshold and requires manual underwriting review before approval.</div>', unsafe_allow_html=True)
            elif risk_prob >= 0.3:
                st.markdown(f'<div class="risk-medium">⚠️ <b>MEDIUM RISK — CONDITIONAL APPROVAL</b><br>Default probability: {risk_prob*100:.1f}%. Consider additional documentation or collateral before proceeding.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ <b>LOW RISK — APPROVE FOR STANDARD PROCESSING</b><br>Default probability: {risk_prob*100:.1f}%. Applicant meets standard credit risk criteria.</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model not found. Please run the notebook first to generate model files.")

# ─── TAB 2: BATCH PROCESSING ───
with tab2:
    st.subheader("Batch Applicant Risk Screening")
    st.info("Upload a CSV file with applicant data. The system will flag high-risk applicants automatically.")

    st.markdown("**Required CSV columns:**")
    st.code("loan_amount, annual_income, cibil_score, loan_tenure_months, employment_type, existing_emi, num_existing_loans, loan_purpose, age, missed_payments_6m, collateral_value")

    uploaded = st.file_uploader("Upload applicants CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} applicants")

        emp_map  = {"Salaried": 2, "Self-Employed": 1, "Business": 0}
        purp_map = {"Business": 0, "Education": 1, "Home": 2, "Personal": 3, "Vehicle": 4}

        batch_proc = batch_df.copy()
        batch_proc['employment_type'] = batch_proc['employment_type'].map(emp_map)
        batch_proc['loan_purpose']    = batch_proc['loan_purpose'].map(purp_map)
        batch_proc['debt_to_income_ratio'] = batch_proc['loan_amount'] / batch_proc['annual_income']
        batch_proc['emi_to_income_ratio']  = batch_proc['existing_emi'] / (batch_proc['annual_income'] / 12)
        batch_proc['loan_to_collateral']   = batch_proc['loan_amount'] / (batch_proc['collateral_value'] + 1)
        batch_proc['risk_flag_cibil']      = (batch_proc['cibil_score'] < 600).astype(int)
        batch_proc['risk_flag_payments']   = (batch_proc['missed_payments_6m'] > 0).astype(int)

        if model:
            scaled     = scaler.transform(batch_proc[features])
            risk_probs = model.predict_proba(scaled)[:, 1]
            batch_df['Risk Probability (%)'] = (risk_probs * 100).round(1)
            batch_df['Risk Band']   = pd.cut(risk_probs, bins=[0,0.3,0.65,1.0], labels=['Low','Medium','High'])
            batch_df['Flag']        = risk_probs >= 0.65
            batch_df['Action']      = batch_df['Flag'].map({True:'🚨 Manual Review', False:'✅ Approve'})

            flagged_count = batch_df['Flag'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Applicants", len(batch_df))
            col2.metric("Flagged (High Risk)", flagged_count)
            col3.metric("Flag Rate", f"{flagged_count/len(batch_df)*100:.1f}%")

            st.dataframe(batch_df[['cibil_score','loan_amount','Risk Probability (%)','Risk Band','Action']].style.applymap(
                lambda v: 'background-color: #ffe0e0' if v == '🚨 Manual Review' else '', subset=['Action']
            ))

            csv = batch_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "risk_screening_results.csv", "text/csv")

# ─── TAB 3: MODEL INSIGHTS ───
with tab3:
    st.subheader("Model Performance & Key Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", "XGBoost")
    col2.metric("Exception Threshold", "65% risk prob")
    col3.metric("Risk Bands", "Low / Medium / High")

    st.markdown("### 📋 Key Business Insights")
    insights = {
        "Top Risk Driver":       "CIBIL score < 600 is the strongest predictor of loan default",
        "Missed Payments":       "Even 1 missed payment in 6 months doubles default probability",
        "Employment Risk":       "Self-employed applicants show ~18% higher default rate vs salaried",
        "Debt-to-Income":        "Loan-to-income ratio > 50% significantly elevates risk band to High",
        "Exception Logic":       "Applicants with risk_probability ≥ 0.65 are flagged for manual review",
        "Model Performance":     "XGBoost outperforms Decision Tree and Random Forest on ROC-AUC",
    }
    for k, v in insights.items():
        st.markdown(f"- **{k}:** {v}")

    st.markdown("### 🔧 Risk Control Framework")
    st.markdown("""
    | Risk Band | Probability Range | Action |
    |---|---|---|
    | 🟢 Low Risk    | 0% – 30%  | Standard approval processing |
    | 🟡 Medium Risk | 30% – 65% | Conditional approval, additional docs |
    | 🔴 High Risk   | 65%+      | Flagged — mandatory manual review |
    """)
