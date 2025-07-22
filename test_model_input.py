import joblib

# Load your trained model
model = joblib.load("churn_model.pkl")

# ✅ This is a properly shaped input with all 30 features in correct order
sample_input = [[
    0, 12, 75.5, 900.0,         # Numerical: SeniorCitizen, tenure, MonthlyCharges, TotalCharges
    1, 1, 1, 1,                 # gender_Male, Partner_Yes, Dependents_Yes, PhoneService_Yes
    0, 1,                      # MultipleLines_No phone service, MultipleLines_Yes
    0, 0,                      # InternetService_Fiber optic, InternetService_No
    0, 1,                      # OnlineSecurity_No internet service, OnlineSecurity_Yes
    0, 1,                      # OnlineBackup_No internet service, OnlineBackup_Yes
    0, 1,                      # DeviceProtection_No internet service, DeviceProtection_Yes
    0, 1,                      # TechSupport_No internet service, TechSupport_Yes
    0, 1,                      # StreamingTV_No internet service, StreamingTV_Yes
    0, 1,                      # StreamingMovies_No internet service, StreamingMovies_Yes
    1, 0,                      # Contract_One year, Contract_Two year
    1,                         # PaperlessBilling_Yes
    1, 0, 0                    # PaymentMethod_Credit card (automatic), Electronic check, Mailed check
]]

# Make prediction
prediction = model.predict(sample_input)
print("Prediction:", prediction)
