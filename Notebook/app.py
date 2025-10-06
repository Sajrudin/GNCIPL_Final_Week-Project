import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, time

# -----------------------------
# Load Model and Preprocessing
# -----------------------------
MODEL_PATH = os.path.join(r"C:\Users\ACER\Desktop\Projects\Finance\Model", "best_model_augmented.pkl")
PIPELINE_PATH = os.path.join(r"C:\Users\ACER\Desktop\Projects\Finance\Model", "preprocess_pipeline.pkl")

model = joblib.load(MODEL_PATH)
preprocess = joblib.load(PIPELINE_PATH)

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")

st.title("💳 Fraud Detection System")
st.markdown("Check if a transaction is **Fraudulent** or **Legit** using the trained ML model.")

st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app predicts whether a transaction is **Fraudulent (1)** or **Legit (0)**.  
- **Single Prediction Mode**: Enter details manually.  
- **Bulk Prediction Mode**: Upload a CSV/Excel file.  
""")

transaction_devices = [
    "Voice Assistant",
    "POS Mobile Device",
    "ATM",
    "POS Mobile App",
    "Virtual Card",
    "Mobile Device",
    "Payment Gateway Device",
    "Debit/Credit Card",
    "Bank Branch",
    "Desktop/Laptop",
    "Self-service Banking Machine",
    "ATM Booth Kiosk",
    "Biometric Scanner",
    "Web Browser",
    "Tablet",
    "Wearable Device",
    "QR Code Scanner",
    "Smart Card",
    "POS Terminal",
    "Banking Chatbot"
]

# -----------------------------
# Mode Selection
# -----------------------------
mode = st.radio("Choose Mode:", ["Single Transaction", "Bulk Upload"])

# -----------------------------
# SINGLE TRANSACTION MODE
# -----------------------------
if mode == "Single Transaction":
    st.header("📌 Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        Customer_ID = st.text_input("Customer ID", "CUST123")
        Customer_Name = st.text_input("Customer Name", "John Doe")
        Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        State = st.text_input("State", "Uttarakhand")
        City = st.text_input("City", "Dehradun")
        Bank_Branch = st.text_input("Bank Branch", "Main Branch")
        Account_Type = st.selectbox("Account Type", ["Savings", "Current", "Salary"])
        Transaction_ID = st.text_input("Transaction ID", "TXN1001")
        Transaction_Date = st.date_input("Transaction Date", datetime.today())
        Transaction_Time = st.time_input("Transaction Time", time(12, 0))
        Merchant_ID = st.text_input("Merchant ID", "MERCHANT001")

    with col2:
        Transaction_Type = st.selectbox("Transaction Type", ["Online", "POS", "ATM", "UPI", "Other"])
        Merchant_Category = st.text_input("Merchant Category", "Shopping")
        Transaction_Device = st.selectbox("Transaction Device",options=transaction_devices,index=transaction_devices.index("Mobile Device")  # Default choice
        )

        Transaction_Location = st.text_input("Transaction Location", "Mall Road")
        Device_Type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "ATM"])
        Transaction_Currency = st.selectbox("Transaction Currency", ["INR", "USD", "EUR"])
        Customer_Contact = st.text_input("Customer Contact", "+91XXXXXXXXXX")
        Transaction_Description = st.text_area("Transaction Description", "Purchase of goods")
        Customer_Email = st.text_input("Customer Email", "customer@email.com")
        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        Transaction_Amount = st.number_input("Transaction Amount", min_value=1.0, value=5000.0, step=100.0)
        Account_Balance = st.number_input("Account Balance", min_value=0.0, value=20000.0, step=500.0)

    if st.button("🚀 Predict Fraud (Single)"):
        input_data = pd.DataFrame([{
            "Customer_ID": Customer_ID,
            "Customer_Name": Customer_Name,
            "Gender": Gender,
            "State": State,
            "City": City,
            "Bank_Branch": Bank_Branch,
            "Account_Type": Account_Type,
            "Transaction_ID": Transaction_ID,
            "Transaction_Date": pd.to_datetime(Transaction_Date),
            "Transaction_Time": str(Transaction_Time),
            "Merchant_ID": Merchant_ID,
            "Transaction_Type": Transaction_Type,
            "Merchant_Category": Merchant_Category,
            "Transaction_Device": Transaction_Device,
            "Transaction_Location": Transaction_Location,
            "Device_Type": Device_Type,
            "Transaction_Currency": Transaction_Currency,
            "Customer_Contact": Customer_Contact,
            "Transaction_Description": Transaction_Description,
            "Customer_Email": Customer_Email,
            "Age": Age,
            "Transaction_Amount": Transaction_Amount,
            "Account_Balance": Account_Balance
        }])

        processed_input = preprocess.transform(input_data)
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]

        st.subheader("🔎 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Risk: {probability:.2%})")
        else:
            st.success(f"✅ Legit Transaction (Fraud Probability: {probability:.2%})")

# -----------------------------
# BULK UPLOAD MODE
# -----------------------------
elif mode == "Bulk Upload":
    st.header("📂 Upload File for Bulk Predictions")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.write("📊 Uploaded Data Preview:")
            st.dataframe(data.head())

            if st.button("🚀 Predict Fraud (Bulk)"):
                processed_data = preprocess.transform(data)
                predictions = model.predict(processed_data)
                probabilities = model.predict_proba(processed_data)[:, 1]

                data["Prediction"] = predictions
                data["Fraud_Probability"] = probabilities

                st.success("✅ Predictions Completed!")
                st.dataframe(data.head(20))

                # Option to download results
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
