import streamlit as st
import pandas as pd
import joblib

# Load model + feature columns
model = joblib.load("model/xgb_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")  # خیلی مهم 👈

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will churn or not")

# -------------------------
# 🧾 USER INPUTS
# -------------------------

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure", 0, 100)
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

# -------------------------
# 🧠 CREATE DATAFRAME
# -------------------------

user_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "PaperlessBilling": paperless,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment
}])

# -------------------------
# 🔄 ONE HOT ENCODING
# -------------------------

user_df = pd.get_dummies(user_df)

# -------------------------
# 🔧 ALIGN WITH TRAIN FEATURES
# -------------------------

user_df = user_df.reindex(columns=feature_names, fill_value=0)

# -------------------------
# 🔮 PREDICTION
# -------------------------

if st.button("Predict Churn"):
    pred = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0][1]

    if pred == 1:
        st.error(f"❌ Customer will CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {prob:.2f})")

    st.write("### Probability Score")
    st.progress(float(prob))