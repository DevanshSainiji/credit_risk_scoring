# AI-Driven Credit Risk Analytics System

import streamlit as st
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from src.agent import run_agent
from src.rag import setup_vector_db
from src.pdf_export import generate_pdf_report

# Setup the RAG Vector DB on startup
# This handles reading knowledge_base and preparing local FAISS index
try:
    setup_vector_db()
except Exception as e:
    st.error(f"Error initializing vector database: {e}")

# Load model and scaler
model = joblib.load("models/logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Credit Risk Scoring & Agentic Lending Assistant")
st.write("Intelligent credit risk assessment powered by Machine Learning and Agentic RAG.")

# API Key input in sidebar for public deployment
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key (required for Agent)", type="password")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
else:
    st.sidebar.warning("Please enter your Groq API Key to enable the Agent Assistant.")

# User Inputs (Wrapped in Form for better UI experience)
with st.form("credit_risk_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        rev_util = st.number_input("Revolving Utilization", min_value=0.0, max_value=2.0, value=0.5)
        late_30_59 = st.number_input("Late 30-59 Days", min_value=0, value=0)
        late_90 = st.number_input("Late 90+ Days", min_value=0, value=0)
        monthly_inc = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
        real_estate = st.number_input("Real Estate Loans", min_value=0, value=1)

    with col2:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        late_60_89 = st.number_input("Late 60-89 Days", min_value=0, value=0)
        debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=5.0, value=0.3)
        open_credit = st.number_input("Open Credit Lines", min_value=0, value=5)
        dependents = st.number_input("Dependents", min_value=0, value=0)
    
    submit_button = st.form_submit_button("Predict & Generate Credit Report")

# Predict Logic
if submit_button:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Cannot run Agent without a Groq API Key. Please provide one in the sidebar.")
        st.stop()

    borrower_data = {
        "rev_util": rev_util, "age": age, "late_30_59": late_30_59, "debt_ratio": debt_ratio,
        "monthly_inc": monthly_inc, "open_credit": open_credit, "late_90": late_90,
        "real_estate": real_estate, "late_60_89": late_60_89, "dependents": dependents
    }
    
    input_data = pd.DataFrame([borrower_data])


    # Feature Engineering (MUST match training)

    input_data["total_late"] = (
        input_data["late_30_59"]
        + input_data["late_60_89"]
        + input_data["late_90"]
    )

    input_data["financial_stress"] = (
        input_data["debt_ratio"]
        * input_data["rev_util"]
    )


    # Critical Fix: Column Order (Hardcoded for models without feature_names_in_)

    cols = ['rev_util', 'age', 'late_30_59', 'debt_ratio', 'monthly_inc', 
            'open_credit', 'late_90', 'real_estate', 'late_60_89', 
            'dependents', 'total_late', 'financial_stress']
    input_data = input_data[cols]


    # Scaling (Loaded dynamically from scaler.pkl)

    input_data_scaled = scaler.transform(input_data)


    # Prediction

    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    if probability < 0.3:
        risk_label = "Low Risk"
    elif probability < 0.6:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    # Save to session state to prevent download button reset
    st.session_state["prediction_data"] = {
        "prediction": prediction,
        "probability": probability,
        "risk_label": risk_label
    }


    # Agentic Reasoner

    with st.spinner("Agent is retrieving guidelines and generating report..."):
        try:
            report = run_agent(borrower_data, risk_label, probability)
            st.session_state["agent_report"] = report
            st.session_state["pdf_bytes"] = generate_pdf_report(report)
        except Exception as e:
            st.error(f"Error during agent execution: {e}")

# Render Results from Session State
if "prediction_data" in st.session_state:
    pred_data = st.session_state["prediction_data"]
    prediction = pred_data["prediction"]
    probability = pred_data["probability"]
    risk_label = pred_data["risk_label"]

    st.header("2. ML Model Output")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Default Prediction", value="Probably a Defaulter" if prediction == 1 else "Not a Defaulter")
    with col2:
        st.metric(label="Probability of Default", value=f"{probability:.2%}")

    if risk_label == "Low Risk":
        st.success(f"{risk_label} - Borrower shows strong repayment profile.")
    elif risk_label == "Medium Risk":
        st.warning(f"{risk_label} - Borrower exhibits moderate credit risk.")
    else:
        st.error(f"{risk_label} - Borrower has elevated probability of default.")

    st.divider()

if "agent_report" in st.session_state and "pdf_bytes" in st.session_state:
    st.header("3. Agentic Lending Decision")
    st.markdown(st.session_state["agent_report"])
    
    st.divider()
    st.download_button(
        label="📄 Download Report as PDF",
        data=st.session_state["pdf_bytes"],
        file_name="Lending_Decision_Report.pdf",
        mime="application/pdf"
    )