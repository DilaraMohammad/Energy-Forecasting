import streamlit as st
import pandas as pd
import joblib

# Load the trained models and scalers
ridge_model = joblib.load('ridge_model.pkl')
logistic_model = joblib.load('logistic_energy_model.pkl')
scaler_r = joblib.load('scaler_r.pkl')
scaler_c = joblib.load('scaler_c.pkl')

st.title("⚡ Energy Consumption Prediction App")
st.write("Enter the energy data to get predictions.")
st.info("💡 Tip: Changing the historical energy values (Lag 1 and Lag 24) will directly affect the model's predictions. Try adjusting them to see how the energy classification changes from Low to High!")
# Input fields for user data
hour = st.slider("Hour", 0, 23, 12)
day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0, 1, 2, 3, 4, 5, 6])
month = st.slider("Month", 1, 12, 6)
year = st.number_input("Year", 2020, 2030, 2024)
is_weekend = st.selectbox("Is Weekend?", [0, 1])
lag1 = st.number_input("PJME_MW Lag 1", value=30000.0)
lag24 = st.number_input("PJME_MW Lag 24", value=30000.0)

# Season encoding matching the exact training columns
season_spring = 1 if month in [3, 4, 5] else 0
season_summer = 1 if month in [6, 7, 8] else 0

# Create input DataFrame with the exact 9 columns and order from training
input_data = pd.DataFrame([[
    hour, day_of_week, month, year, is_weekend, season_spring, season_summer, lag1, lag24
]], columns=[
    'Hour', 'DayOfWeek', 'Month', 'Year', 'Is_Weekend', 'Season_Spring', 'Season_Summer', 'PJME_MW_lag1', 'PJME_MW_lag24'
])

# Predict button
if st.button("Predict"):
    # 1. Regression Prediction (Ridge)
    input_scaled_r = scaler_r.transform(input_data)
    prediction_reg = ridge_model.predict(input_scaled_r)[0]
    
    # 2. Classification Prediction (Logistic)
    input_scaled_c = scaler_c.transform(input_data)
    prediction_clf = logistic_model.predict(input_scaled_c)[0]
    
    # Display results
    st.subheader("Results:")
    st.success(f"Predicted Energy Consumption: *{prediction_reg:.2f} MW*")
    
    result_text = "High Energy Load 🔴" if prediction_clf == 1 else "Low Energy Load 🟢"
    st.info(f"Classification Status: *{result_text}*")