# 💳 Credit Card Fraud Detection

## 📌 Overview
This project is an end-to-end machine learning system designed to detect fraudulent credit card transactions. It involves data preprocessing, exploratory data analysis (EDA), feature engineering, and model building using a **Random Forest Classifier**.I used stratified splitting to maintain the fraud-to-legitimate ratio across training and testing datasets. The model is deployed using **Streamlit** to allow real-time fraud prediction.

---

## 🎯 Features
- Predicts whether a transaction is **fraudulent or legitimate**
- Displays **fraud probability score**
- Uses a **custom threshold (0.2)** to reduce false negatives
- Interactive UI for entering transaction details
- Real-time predictions via web app

---

## 🧠 Model Details
- **Algorithm:** Random Forest Classifier
- **Library:** Scikit-learn
- **Evaluation Focus:** Reducing false negatives (fraud cases missed)
- **Threshold:** 0.2 (instead of default 0.5)

---

## 🔄 Data Processing
### 🧹 Data Cleaning
- Removed inconsistent and invalid entries
- Handled missing values
- Ensured correct data types

### 🔄 Data Transformation
- Encoded categorical variables (gender, transaction category)
- Scaled numerical features using StandardScaler

### 🧠 Feature Engineering
- Extracted time-based features:
  - Transaction hour
  - Day, month, weekday
- Created behavioral indicators from transaction patterns
- Encoded features like merchant, job, and state

---

## 📊 Exploratory Data Analysis (EDA)
- Analyzed fraud vs non-fraud distribution
- Identified class imbalance
- Explored relationships between:
  - Transaction amount
  - Time patterns
  - Categories

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
