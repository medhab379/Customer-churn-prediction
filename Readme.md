
---

# 📉 Customer Churn Prediction Web App

A complete **end-to-end machine learning web application** that predicts whether a telecom customer is likely to churn (leave the service) based on their usage behavior and account information.

---

## 🔍 Project Objective

The primary objective is to help telecom companies **identify customers at risk of churning** so that they can implement retention strategies in advance. This project simulates a **real-world deployment** scenario with interactive frontend, scalable backend, and a robust machine learning model.

---

## 🛠️ Tech Stack

* **Python 3.9** – core programming language
* **Pandas, NumPy** – data manipulation
* **Scikit-learn** – model building & evaluation
* **Flask** – lightweight Python web framework
* **HTML/CSS** – frontend form interface
* **Joblib** – model serialization
* **VS Code / CMD** – development environment

---

## 📊 Dataset

* **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Size**: 7,043 customer records
* **Target variable**: `Churn` (Yes/No)
* **Preprocessing**:

  * Missing value imputation
  * Label encoding and one-hot encoding
  * Feature scaling (optional for tree models)
  * Removed duplicate or irrelevant features

---

## 🧠 Machine Learning Pipeline

* **Algorithm used**: `RandomForestClassifier`

* **Why Random Forest?**

  * Handles categorical and numerical features well
  * Performs feature importance analysis
  * Robust to outliers and overfitting

* **Evaluation Metrics**:

  * Accuracy: \~80%
  * Precision & Recall: Evaluated for both churn and non-churn classes
  * F1-Score: Balanced performance metric

* **Model Serialization**:

  * Trained model saved as `churn_model.pkl`
  * Feature list saved as `churn_features.pkl`

---

## 🌐 Web Application Overview

This Flask-based app accepts customer inputs through a web form and returns a **churn prediction**.

### 📁 Project Structure

```
Customer_Churn_Project/
│
├── churn_analysis.ipynb         # Full EDA, preprocessing, training
├── train_and_save_model.py      # Training script for model creation
├── churn_model.pkl              # Saved ML model
├── churn_features.pkl           # Saved feature columns used during training
├── app.py                       # Flask backend server
├── templates/
│   └── index.html               # Frontend form interface
├── test_model_input.py          # Manual testing script
└── README.md                    # Project documentation
```

---

## 💻 How to Run This Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/medhab379/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser to interact with the web app.

---

## 📌 Sample Inputs & Outputs

* **Input**: Contract = Month-to-Month, InternetService = Fiber optic, MonthlyCharges = High, Tenure = Low

* **Output**: `Prediction: Churn`

* **Input**: Contract = Two-year, InternetService = DSL, MonthlyCharges = Low, Tenure = High

* **Output**: `Prediction: Not Churn`

---

## 📬 Contact

Feel free to reach out if you'd like to discuss the project or potential opportunities!
medha2419@gmail.com
https://www.linkedin.com/in/medha-bhat-443670260



---