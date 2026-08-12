# ==========================================
# ULTIMATE ENERGY CONSUMPTION ML & WEB PIPELINE
# ==========================================
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression , RidgeCV, Ridge
from sklearn.metrics import mean_squared_error, classification_report

# ==========================================
# 1. LOAD AND PREPARE DATASET
# ==========================================
FILE_PATH = r"D:\ML-Journey\ML-Specialization courses\Energy-Forecasting\pjm_energy.csv"
    # Defensive check using os library to see if file exists 
 
if os.path.exists(FILE_PATH):
        print(f"Success: File '{FILE_PATH}' found on disk. Loading real data...")
        df = pd.read_csv(FILE_PATH)
else:
        print(f"Warning: File '{FILE_PATH}' NOT found! Generating synthetic energy data dynamically...")
        # Generate synthetic data mimicking real energy columns if file is missing
        np.random.seed(42)
        date_range = pd.date_range(start="2002-01-01", end="2005-01-01", freq="h")
        synthetic_mw = np.random.normal(loc=30000, scale=5000, size=len(date_range))
        df = pd.DataFrame({
            'Datetime': date_range,
            'PJME_MW': synthetic_mw
        })
print(df.columns.to_list())
print(df.shape)
print(df.dtypes)
print(df.head())
df.info()

# Convert timestamp to datetime and set it as index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')

# Extract basic time features
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Month'] = df.index.month
df['Year'] = df.index.year
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Extract seasons based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)

# One-hot encoding for seasons (dropping first to avoid multicollinearity)
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Add Lag Features to dramatically boost Ridge Regression performance
df['PJME_MW_lag1'] = df['PJME_MW'].shift(1)
df['PJME_MW_lag24'] = df['PJME_MW'].shift(24)

# Drop missing values resulting from shifting
df = df.dropna()

# ==========================================
# 2. DEFINE TARGETS FOR REGRESSION & CLASSIFICATION
# ==========================================
# For Regression: Predicting continuous energy consumption (PJME_MW)
y_reg = df['PJME_MW']

# For Classification: Binary target (0 or 1) based on median split
median_energy = df['PJME_MW'].median()
df['y_class'] = (df['PJME_MW'] > median_energy).astype(int)
y_class = df['y_class']

# Define feature columns used for training both models
feature_cols = [
    'Hour', 'DayOfWeek', 'Month', 'Year', 
    'Is_Weekend', 'Season_Spring', 'Season_Summer', 
    'PJME_MW_lag1', 'PJME_MW_lag24'
]

X = df[feature_cols]

# ==========================================
# 3. TRAINING MODELS (REGRESSION & CLASSIFICATION)
# ==========================================
# Define file paths using the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "ridge_model.pkl")
clf_path = os.path.join(current_dir, "logistic_energy_model.pkl")
scaler_r_path = os.path.join(current_dir, "scaler_r.pkl")
scaler_c_path = os.path.join(current_dir, "scaler_c.pkl")

if not os.path.exists(model_path) or not os.path.exists(clf_path) or not os.path.exists(scaler_r_path) or not os.path.exists(scaler_c_path):
    print("--- Training Models from Scratch ---")

    # 3.1 Ridge Regression Training & Evaluation
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    alphas_list = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_model = RidgeCV(alphas=alphas_list, cv=5)
    ridge_model.fit(X_train_r_scaled, y_train_r)
    
    best_alpha = ridge_model.alpha_
    y_pred_ridge = ridge_model.predict(X_test_r_scaled)
    mse = mean_squared_error(y_test_r, y_pred_ridge)
    rmse = np.sqrt(mse)
    r2 = ridge_model.score(X_test_r_scaled, y_test_r)

    print(f"\n[Ridge Regression Results]")
    print(f"Best Alpha: {best_alpha}")
    print(f"Ridge R2 Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # 3.2 Logistic Regression Classification Training & Evaluation
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    logistic_energy_model = LogisticRegression(max_iter=100, random_state=42)
    logistic_energy_model.fit(X_train_c_scaled, y_train_c)

    y_pred_clf = logistic_energy_model.predict(X_test_c_scaled)
    accuracy = logistic_energy_model.score(X_test_c_scaled, y_test_c)

    print(f"\n[Logistic Regression Results]")
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test_c, y_pred_clf))
    print(f"Logisitic Regriton model Iteretion times {logistic_energy_model.n_iter_} ")

#Get the directory where the current script is located


# Save models and scalers using joblib with their full paths
joblib.dump(ridge_model, model_path)
joblib.dump(logistic_energy_model, clf_path)
joblib.dump(scaler_r, scaler_r_path)
joblib.dump(scaler_c, scaler_c_path)

print("Models and Scalers saved successfully in:", current_dir)