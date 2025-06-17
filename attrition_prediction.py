import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Attrition Risk Predictor", layout="wide")
st.title("🔍 Prediksi Risiko Karyawan Resign")

# Load model dan scaler
rf_model = joblib.load('random_forest.pkl')
xgb_model = joblib.load('xgboost.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Buat tiga kolom layout: sidebar kiri untuk hasil, kolom tengah untuk input
left_sidebar, center, _ = st.columns([1, 2, 1])

with center:
    st.subheader("📋 Input Data Karyawan")

    distance = st.slider("Jarak dari Rumah ke Kantor (km)", 1, 30, 20)
    companies = st.number_input("Jumlah Perusahaan Sebelumnya", 0, 10, 3)
    rate = st.slider("Monthly Rate", 1000, 20000, 8000, step=500)
    years = st.slider("Total Working Years", 0, 40, 3)
    age = st.slider("Usia", 18, 60, 25)
    level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    overtime = st.radio("Lembur", ["Yes", "No"])
    role = st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative"
    ])
    status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
    travel = st.selectbox("Business Travel", ["Travel_Frequently", "Travel_Rarely", "Non-Travel"])

# Prediksi ketika input valid
input_dict = {
    "DistanceFromHome": distance,
    "NumCompaniesWorked": companies,
    "MonthlyRate": rate,
    "TotalWorkingYears": years,
    "Age": age,
    "JobLevel": level,
    "OverTime_No": int(overtime == "No"),
    "OverTime_Yes": int(overtime == "Yes"),
    **{f"JobRole_{r}": int(role == r) for r in [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative"
    ]},
    **{f"MaritalStatus_{s}": int(status == s) for s in ["Divorced", "Married", "Single"]},
    **{f"BusinessTravel_{t}": int(travel == t) for t in ["Non-Travel", "Travel_Frequently", "Travel_Rarely"]}
}
df_input = pd.DataFrame([input_dict])
df_scaled = loaded_scaler.transform(df_input)

# Jalankan prediksi
rf_proba = rf_model.predict_proba(df_scaled)[:, 1]
xgb_proba = xgb_model.predict_proba(df_scaled)[:, 1]
ensemble_proba = (rf_proba + xgb_proba) / 2
ensemble_pred = int(ensemble_proba[0] >= 0.5)

# Sidebar kiri: hasil prediksi
with left_sidebar:
    st.markdown("### 📊 Hasil Prediksi")
    result_text = "🎯 **Kemungkinan Resign**" if ensemble_pred else "✅ **Kemungkinan Bertahan**"
    st.markdown(result_text)
    st.metric("Probabilitas Leave", f"{ensemble_proba[0]*100:.2f}%")
    st.metric("Probabilitas Stay", f"{(1 - ensemble_proba[0])*100:.2f}%")
    st.progress(float(ensemble_proba[0]))
