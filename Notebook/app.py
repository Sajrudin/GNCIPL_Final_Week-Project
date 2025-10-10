import streamlit as st
import pandas as pd
import joblib
import os, sys
from datetime import datetime, time
import plotly.graph_objects as go


# Add Notebook folder to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define ROOT directory (one level up from Notebook)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths to model and pipeline
MODEL_PATH = os.path.join(ROOT_DIR, "Model", "best_model_augmented.pkl")
PIPELINE_PATH = os.path.join(ROOT_DIR, "Model", "preprocess_pipeline.pkl")

st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")

# Try loading model and pipeline
try:
    model = joblib.load(MODEL_PATH)
    preprocess = joblib.load(PIPELINE_PATH)
except FileNotFoundError:
    st.error("Model or preprocessing pipeline not found. Please verify the file paths.")
    st.stop()
except Exception as e:
    st.error(f" Error loading model or pipeline: {e}")
    st.stop()


# -----------------------------
# Streamlit App Config
# -----------------------------

st.title("💳 Fraud Detection System")
st.markdown("Check if a transaction is **Fraudulent** or **Legit** using the trained ML model.")

# Sidebar Information
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app predicts whether a transaction is **Fraudulent (1)** or **Legit (0)**.  
- **Single Prediction Mode**: Enter details manually.  
- **Bulk Prediction Mode**: Upload a CSV/Excel file.  
""")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")
prediction_threshold = st.sidebar.slider(
    "Set Fraud Risk Threshold (%)",
    min_value=0,
    max_value=100,
    value=55,
    step=1,
    format="%d%%",
    key="threshold_slider"
) / 100.0  # Convert percentage to decimal

transaction_devices = [
    "Voice Assistant", "POS Mobile Device", "ATM", "POS Mobile App", "Virtual Card",
    "Mobile Device", "Payment Gateway Device", "Debit/Credit Card", "Bank Branch",
    "Desktop/Laptop", "Self-service Banking Machine", "ATM Booth Kiosk",
    "Biometric Scanner", "Web Browser", "Tablet", "Wearable Device",
    "QR Code Scanner", "Smart Card", "POS Terminal", "Banking Chatbot"
]

# -----------------------------
# Enhanced Gauge Chart Function
# -----------------------------
def create_gauge_chart(probability):
    risk_percentage = probability * 100
    
    # Determine risk level text
    if risk_percentage <= 33:
        level = "Low Risk 🟢"
    elif risk_percentage <= 66:
        level = "Medium Risk 🟡"
    else:
        level = "High Risk 🔴"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        number={'suffix': '%', 'font': {'size': 40, 'color': 'black'}},
        title={'text': f"<b>Fraud Risk</b><br><span style='font-size:16px;color:gray'>{level}</span>", 'font': {'size': 20}},
        gauge={
            'shape': "angular",
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "gold"},
                {'range': [66, 100], 'color': "tomato"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'thickness': 0.8,
                'value': risk_percentage
            }
        },
        delta={'reference': prediction_threshold*100, 'increasing': {'color': "red"}}
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'family': "Arial", 'color': "black"}
    )

    return fig


# -----------------------------
# Mode Selection
# -----------------------------
st.markdown("---")
mode = st.radio("Choose Mode:", ["Single Transaction", "Bulk Upload"])
st.markdown("---")

# -----------------------------
# SINGLE TRANSACTION MODE
# -----------------------------
if mode == "Single Transaction":
    st.header("📌 Enter Transaction Details")

    with st.expander("Customer Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            Customer_ID = st.text_input("Customer ID", "CUST123")
            Customer_Name = st.text_input("Customer Name", "John Doe")
            Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        with col2:
            State = st.text_input("State", "Uttarakhand")
            City = st.text_input("City", "Dehradun")
            Customer_Contact = st.text_input("Customer Contact", "+91XXXXXXXXXX")
            Customer_Email = st.text_input("Customer Email", "customer@email.com")

    with st.expander("Account & Bank Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            Bank_Branch = st.text_input("Bank Branch", "Main Branch")
            Account_Type = st.selectbox("Account Type", ["Savings", "Current", "Salary"])
        with col2:
            Account_Balance = st.number_input("Account Balance", min_value=0.0, value=20000.0, step=500.0)

    with st.expander("Transaction Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            Transaction_ID = st.text_input("Transaction ID", "TXN1001")
            Transaction_Date = st.date_input("Transaction Date", datetime.today())
            Transaction_Time = st.time_input("Transaction Time", time(12, 0))
            Transaction_Type = st.selectbox("Transaction Type", ["Online", "POS", "ATM", "UPI", "Other"])
            Transaction_Amount = st.number_input("Transaction Amount", min_value=1.0, value=5000.0, step=100.0)
            Transaction_Currency = st.selectbox("Transaction Currency", ["INR", "USD", "EUR"])
        with col2:
            Merchant_ID = st.text_input("Merchant ID", "MERCHANT001")
            Merchant_Category = st.text_input("Merchant Category", "Shopping")
            Transaction_Device = st.selectbox("Transaction Device", transaction_devices, index=transaction_devices.index("Mobile Device"))
            Device_Type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "ATM"])
            Transaction_Location = st.text_input("Transaction Location", "Mall Road")
            Transaction_Description = st.text_area("Transaction Description", "Purchase of goods")

    st.markdown("---")
    if st.button("🚀 Predict Fraud (Single Transaction)", type="primary"):
        input_data = pd.DataFrame([{
            "Customer_ID": Customer_ID, "Customer_Name": Customer_Name, "Gender": Gender,
            "State": State, "City": City, "Bank_Branch": Bank_Branch,
            "Account_Type": Account_Type, "Transaction_ID": Transaction_ID,
            "Transaction_Date": pd.to_datetime(Transaction_Date),
            "Transaction_Time": str(Transaction_Time), "Merchant_ID": Merchant_ID,
            "Transaction_Type": Transaction_Type, "Merchant_Category": Merchant_Category,
            "Transaction_Device": Transaction_Device, "Transaction_Location": Transaction_Location,
            "Device_Type": Device_Type, "Transaction_Currency": Transaction_Currency,
            "Customer_Contact": Customer_Contact, "Transaction_Description": Transaction_Description,
            "Customer_Email": Customer_Email, "Age": Age,
            "Transaction_Amount": Transaction_Amount, "Account_Balance": Account_Balance
        }])

        try:
            processed_input = preprocess.transform(input_data)
            probability = model.predict_proba(processed_input)[0][1]
            prediction = 1 if probability >= prediction_threshold else 0

            st.subheader("🔎 Prediction Result")
            st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

            if prediction == 1:
                st.error(f"🚨 Fraudulent Transaction Detected! (Risk: **{probability:.2%}**)")
            else:
                st.success(f"✅ Legit Transaction (Fraud Probability: **{probability:.2%}**)")

            st.info(f"Threshold for Fraud Detection: **{prediction_threshold:.0%}** (Adjust in sidebar).")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

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

            st.info(f"Using fraud detection threshold: **{prediction_threshold:.0%}**")

            if st.button("🚀 Predict Fraud (Bulk)", type="primary"):
                with st.spinner("Processing..."):
                    processed_data = preprocess.transform(data)
                    probabilities = model.predict_proba(processed_data)[:, 1]
                    predictions = (probabilities >= prediction_threshold).astype(int)

                    data["Prediction"] = predictions
                    data["Fraud_Probability"] = probabilities

                st.success("✅ Predictions Completed!")
                st.subheader("Results Summary:")
                st.write(f"Total Transactions: **{len(data)}**")
                st.write(f"Fraudulent: **{data['Prediction'].sum()}**")
                st.write(f"Legit: **{len(data) - data['Prediction'].sum()}**")

                st.dataframe(data.head(50))

                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
