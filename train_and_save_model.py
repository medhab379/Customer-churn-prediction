# train_and_save_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Drop customerID
data.drop("customerID", axis=1, inplace=True)

# Convert target to binary
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Label encode binary categorical features
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
le = LabelEncoder()
for col in binary_cols:
    data[col] = le.fit_transform(data[col])

# One-hot encode remaining categorical variables
data = pd.get_dummies(data)

# Save feature columns
with open("churn_features.pkl", "wb") as f:
    pickle.dump(data.drop("Churn", axis=1).columns.tolist(), f)

# Split and train
X = data.drop("Churn", axis=1)
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as churn_model.pkl")
