import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("hybrid_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="HTH Detection Interface", layout="centered")
st.title("🧪 Hardware Trojan Detection Interface")

st.markdown("Enter waveform features below:")

# User input form
with st.form("input_form"):
    pkpk = st.number_input("Pk-Pk Voltage (V)", value=1.12, format="%.4f")
    maximum = st.number_input("Maximum Voltage (V)", value=0.426, format="%.4f")
    minimum = st.number_input("Minimum Voltage (V)", value=-0.691, format="%.4f")
    freq = st.number_input("Frequency (MHz)", value=1.986, format="%.4f")

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Create dataframe
    input_data = pd.DataFrame([[pkpk, maximum, minimum, freq]],
                              columns=['Pk-Pk', 'Maximum', 'Minimum', 'Frequency'])

    # Scale and predict
    scaled_input = scaler.transform(input_data)
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")
    if pred == 1:
        st.error(f"🚨 Prediction: HTH-Infected\n\nConfidence: {prob * 100:.2f}%")
    else:
        st.success(f"✅ Prediction: HTH-Free\n\nConfidence: {(1 - prob) * 100:.2f}%")
