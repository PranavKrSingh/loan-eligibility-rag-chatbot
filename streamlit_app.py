# streamlit_app.py  –  sleek UI with single + batch prediction
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from models.predict import predict_single

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/streamlit/brand/master/logos/streamlit-logo-secondary-colormark-darktext.png",
        width=140,
    )
    st.markdown(
        """
        ### 🔍 About  
        **Loan Eligibility Predictor**  
        • RAG + ML demo  
        • Built with Streamlit  
        • Dataset: Dream Housing &nbsp;🏠

        **GitHub:**  
        [github.com/your‑repo](https://github.com/)
        """
    )
    st.markdown("---")
    if st.checkbox("Show dataset class balance"):
        df_tmp = pd.read_csv("data/train_fe.csv")
        st.write(df_tmp["Loan_Status"].value_counts(normalize=True)
                 .rename({"Y": "Approved (Y)", "N": "Rejected (N)"}))

# ---------- TABS ----------
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction"])

# ---------- TAB 1: SINGLE ----------
with tab1:
    st.subheader("Applicant details")

    # Two‑column form
    col1, col2 = st.columns(2)
    with st.form("single_form"):
        with col1:
            loan_id = st.text_input("Loan ID", value="LP0XXXX")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_emp = st.selectbox("Self Employed", ["Yes", "No"])
        with col2:
            appl_inc = st.number_input("Applicant Income (₹)", min_value=0, step=500)
            coappl_inc = st.number_input("Co‑applicant Income (₹)", min_value=0, step=500)
            loan_amt = st.number_input("Loan Amount (₹ thousands)", min_value=0, step=10)
            term = st.selectbox("Loan Term (months)", [360, 240, 180, 120, 84, 60, 36, 12])
            credit_hist = st.selectbox("Credit History", [1, 0])
            prop_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        sample = {
            "Loan_ID": loan_id,
            "Gender": gender.title(),
            "Married": married.title(),
            "Dependents": dependents,
            "Education": education.title(),
            "Self_Employed": self_emp.title(),
            "ApplicantIncome": appl_inc,
            "CoapplicantIncome": coappl_inc,
            "LoanAmount": loan_amt,
            "Loan_Amount_Term": term,
            "Credit_History": credit_hist,
            "Property_Area": prop_area.title(),
        }

        result = predict_single(sample)
        # (Optional) probability if you expose it in predict_single
        prob = (
            None
            if not hasattr(predict_single, "prob_last")
            else predict_single.prob_last
        )

        # ----- RESULT CARD -----
        st.markdown("### Result")
        colA, colB = st.columns([1, 3])
        with colA:
            label_color = "#4CAF50" if result == "Approved" else "#E74C3C"
            st.markdown(
                f"""
                <div style="padding:20px;border-radius:10px;background:{label_color};text-align:center;">
                    <span style="font-size:24px;color:#fff;">{result}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with colB:
            st.write(pd.DataFrame(sample, index=["Value"]).T)

        # Confidence bar
        if prob is not None:
            pct = int(prob * 100)
            st.markdown(f"**Confidence:** {pct}%")
            st.progress(pct)

# ---------- TAB 2: BATCH ----------
with tab2:
    st.subheader("Upload CSV for bulk scoring")
    st.write("CSV must have the same columns as the training data (except Loan_Status).")
    batch_file = st.file_uploader("Choose a file", type=["csv"])

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        preds = []
        for _, row in batch_df.iterrows():
            preds.append(predict_single(row.to_dict()))
        batch_df["Loan_Prediction"] = preds
        st.success("✅ Scored file preview")
        st.dataframe(batch_df.head())

        # Download link
        csv_out = batch_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download full predictions",
            data=csv_out,
            file_name="loan_predictions.csv",
            mime="text/csv",
        )

# ---------- FOOTER ----------
st.markdown(
    """
    <style>
        footer {visibility: hidden;}
        .reportview-container .main .block-container{padding-top:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
