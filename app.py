from flask import Flask, render_template, request
import pandas as pd
import pickle
import os  # Needed for reading environment variables

app = Flask(__name__)

# Load trained model and feature list
model = pickle.load(open("churn_model.pkl", "rb"))
features = pickle.load(open("churn_features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form input data
    input_data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': int(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges']),
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaymentMethod': request.form['PaymentMethod'],
    }

    # Create DataFrame and preprocess it
    df = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=features, fill_value=0)

    # Predict churn
    prediction = model.predict(df_encoded)[0]
    result_text = "Churn" if prediction == 1 else "Not Churn"

    return render_template("result.html", prediction=result_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render gives port in environment
    app.run(host="0.0.0.0", port=port)        # Make app visible externally
