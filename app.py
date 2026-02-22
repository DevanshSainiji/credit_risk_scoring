import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/Decision_Tree.pkl")

st.title("Credit Risk Scoring App")
st.write("Enter borrower details:")

# --------------------
# User Inputs
# --------------------
rev_util = st.number_input("Revolving Utilization", min_value=0.0, max_value=2.0, value=0.5)
age = st.number_input("Age", min_value=18, max_value=100, value=30)

late_30_59 = st.number_input("Late 30-59 Days", min_value=0, value=0)
late_60_89 = st.number_input("Late 60-89 Days", min_value=0, value=0)
late_90 = st.number_input("Late 90+ Days", min_value=0, value=0)

debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=5.0, value=0.3)
monthly_inc = st.number_input("Monthly Income", min_value=0.0, value=5000.0)

open_credit = st.number_input("Open Credit Lines", min_value=0, value=5)
real_estate = st.number_input("Real Estate Loans", min_value=0, value=1)
dependents = st.number_input("Dependents", min_value=0, value=0)

# --------------------
# Predict Button
# --------------------
if st.button("Predict Credit Risk"):

    input_data = pd.DataFrame([{
        "rev_util": rev_util,
        "age": age,
        "late_30_59": late_30_59,
        "debt_ratio": debt_ratio,
        "monthly_inc": monthly_inc,
        "open_credit": open_credit,
        "late_90": late_90,
        "real_estate": real_estate,
        "late_60_89": late_60_89,
        "dependents": dependents
    }])

    # --------------------
    # Feature Engineering (MUST match training)
    # --------------------
    input_data["total_late"] = (
        input_data["late_30_59"]
        + input_data["late_60_89"]
        + input_data["late_90"]
    )

    input_data["financial_stress"] = (
        input_data["debt_ratio"]
        * input_data["rev_util"]
    )

    # --------------------
    # 🔥 Critical Fix: Column Order
    # --------------------
    input_data = input_data[model.feature_names_in_]

    # --------------------
    # Prediction
    # --------------------
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Default Prediction",
            value="Probably a Defaulter" if prediction == 1 else "Not a Defaulter"
        )

    with col2:
        st.metric(
            label="Probability of Default",
            value=f"{probability:.2%}"
        )

    st.write("---")

    # Risk categorization
    if probability < 0.20:
        risk_label = "Low Risk Borrower"
        risk_msg = "Borrower shows strong repayment profile."
        st.success(risk_label)

    elif probability < 0.60:
        risk_label = "Medium Risk Borrower"
        risk_msg = "Borrower exhibits moderate credit risk."
        st.warning(risk_label)

    else:
        risk_label = "High Risk Borrower"
        risk_msg = "Borrower has elevated probability of default."
        st.error(risk_label)

    st.caption(risk_msg)