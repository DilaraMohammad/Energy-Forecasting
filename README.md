# ⚡ Energy Consumption Prediction & Classification App

An end-to-end Machine Learning project designed to forecast energy consumption and classify energy demand levels using historical power grid data. This project includes a fully interactive web application built with *Streamlit*.

---

## 🚀 Project Overview
This system utilizes historical time-series data to perform two main tasks:
1. *Regression (Ridge Regression):* Predicts the exact future energy consumption value (in Megawatts).
2. *Classification (Logistic Regression):* Classifies whether the energy demand will be High or Low based on median thresholds.

---

## 🛠️ Tech Stack & Libraries
* *Python*
* *Scikit-Learn* (for Machine Learning models and data scaling)
* *Pandas & NumPy* (for data manipulation and processing)
* *Streamlit* (for the interactive web application)
* *Joblib* (for saving and loading trained models and scalers)

---

## 📁 Project Structure
```text
Energy-Forecasting/
│
├── app.py                      # Streamlit web application
├── energy_model.py             # Data preprocessing, training, and evaluation script
├── pjm_energy.csv              # Dataset file
├── ridge_model.pkl             # Trained Ridge Regression model
├── logistic_energy_model.pkl   # Trained Logistic Regression model
├── scaler_r.pkl                # Scaler for regression model
├── scaler_c.pkl                # Scaler for classification model
└── requirements.txt            # Required Python packages