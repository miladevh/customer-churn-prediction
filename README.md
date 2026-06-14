# 📊 Customer Churn Prediction (End-to-End ML Project)

This project focuses on predicting customer churn for a telecom company using multiple machine learning models and interpreting predictions using SHAP (Explainable AI).

---

# 🧠 Problem Statement

Customer churn (when a customer leaves a service) is a critical issue for telecom companies.  
The goal of this project is to:

- Predict whether a customer will churn or not
- Identify key factors influencing churn
- Improve business decision-making using data-driven insights

---

# 📂 Dataset Overview

The dataset contains 7043 customer records with 20 features including:

### Customer Information
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure

### Services Subscribed
- PhoneService
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

### Account Information
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

### Target Variable
- Churn (Yes/No)

---

# 🧹 Data Preprocessing

The following steps were applied:

- Handling missing values
- Converting `TotalCharges` to numeric
- Encoding categorical variables using One-Hot Encoding
- Converting boolean columns to numeric (0/1)
- Feature scaling (where needed)
- Train-test split

---

# ⚖️ Handling Class Imbalance

The dataset was imbalanced (fewer churn cases).  
To solve this:

- SMOTE (Synthetic Minority Over-sampling Technique) was applied

---

# 🤖 Machine Learning Models

Three models were trained and evaluated:

## 1. Logistic Regression
- Simple baseline model
- Good recall but lower precision

## 2. Random Forest Classifier
- Strong overall performance
- Best precision among models

## 3. XGBoost Classifier
- Best recall for churn class
- Best overall balance for business use-case

---

# 📊 Model Performance

| Model               | Accuracy | Precision (Churn) | Recall (Churn) | F1-score |
|--------------------|----------|------------------|---------------|----------|
| Logistic Regression | 0.76     | 0.53             | 0.71          | 0.61     |
| Random Forest       | 0.78     | 0.61             | 0.48          | 0.54     |
| XGBoost             | 0.76     | 0.53             | 0.76          | 0.63     |

---

# 🧠 Best Model Selection

The final model choice depends on business objective:

- If we prioritize **catching churned customers (Recall)** → XGBoost is best
- If we prioritize **reducing false alarms (Precision)** → Random Forest
- Overall balanced performance → XGBoost

---

# 🔍 Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

## Key insights from SHAP analysis:

### Most Important Features:
- tenure (customer lifetime)
- Contract type (Month-to-month is high risk)
- MonthlyCharges
- InternetService
- OnlineSecurity
- TechSupport
- PaymentMethod

---

# 📈 Business Insights

From the model and SHAP analysis:

- New customers are more likely to churn
- Month-to-month contracts are high risk
- Lack of technical support increases churn probability
- Higher monthly charges increase churn risk
- Electronic payment methods are associated with higher churn

---

# 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- SHAP

---

# 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna)
- Feature engineering improvements
- Deployment using Streamlit
- Model monitoring for real-world usage
- SHAP-based interactive dashboard

---

# 📌 Project Highlights

✔ End-to-end ML pipeline  
✔ Handling imbalanced dataset (SMOTE)  
✔ Multiple ML models comparison  
✔ Explainable AI using SHAP  
✔ Business insights extracted from model

---

# 👨‍💻 Author

This project was built as a machine learning portfolio project focusing on classification, imbalance handling, and model interpretability.

---

# 📍 Conclusion

This project demonstrates how machine learning can be used not only for prediction but also for extracting actionable business insights through explainable AI techniques like SHAP.
