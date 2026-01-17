# 🛡️ Cybersecurity Anomaly Detection System

A high-performance, unsupervised machine learning application designed to identify suspicious patterns in access logs without requiring pre-labeled attack data.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📖 Project Overview
Traditional cybersecurity detection often relies on known "signatures." This project takes a **Behavioral Analysis** approach using the **Isolation Forest** algorithm. By analyzing features like login frequency and session duration, the model identifies "outliers" that deviate significantly from the norm—potential indicators of credential stuffing, account takeover, or insider threats.

## 🚀 Key Features
- **Unsupervised Learning:** No target labels required. The model learns what "normal" looks like on its own.
- **Dynamic Feature Selection:** Automatically identifies high-variance features to focus on the most informative data points.
- **Interactive Visualizations:**
    - **Decision Boundaries:** See exactly where the model draws the line between safe and suspicious.
    - **SHAP Integration:** Explainable AI (XAI) that tells you *why* a specific user session was flagged.
- **Real-time Analytics:** Instant metrics on anomaly percentages and severity scores.

## 🛠️ Tech Stack
- **Core:** Python 3.10+
- **ML Framework:** Scikit-Learn (Isolation Forest)
- **Explainability:** SHAP (Shapley Additive Explanations)
- **UI/Frontend:** Streamlit
- **Visualization:** Matplotlib, NumPy, Pandas

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ikennaphill/Access-Log-Anomaly-Detection-Isolation-Forest.git

A high-performance, unsupervised machine learning application designed to identify suspicious patterns in access logs without requiring pre-labeled attack data.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📖 Project Overview
Traditional cybersecurity detection often relies on known "signatures." This project takes a **Behavioral Analysis** approach using the **Isolation Forest** algorithm. By analyzing features like login frequency and session duration, the model identifies "outliers" that deviate significantly from the norm—potential indicators of credential stuffing, account takeover, or insider threats.

## 🚀 Key Features
- **Unsupervised Learning:** No target labels required. The model learns what "normal" looks like on its own.
- **Dynamic Feature Selection:** Automatically identifies high-variance features to focus on the most informative data points.
- **Interactive Visualizations:**
    - **Decision Boundaries:** See exactly where the model draws the line between safe and suspicious.
    - **SHAP Integration:** Explainable AI (XAI) that tells you *why* a specific user session was flagged.
- **Real-time Analytics:** Instant metrics on anomaly percentages and severity scores.

## 🛠️ Tech Stack
- **Core:** Python 3.10+
- **ML Framework:** Scikit-Learn (Isolation Forest)
- **Explainability:** SHAP (Shapley Additive Explanations)
- **UI/Frontend:** Streamlit
- **Visualization:** Matplotlib, NumPy, Pandas

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ikennaphill/Access-Log-Anomaly-Detection-Isolation-Forest.git
   cd Access-Log-Anomaly-Detection-Isolation-Forest/Access-Log-Anomaly-Detection-Isolation-Forest.git
   cd Access-Log-Anomaly-Detection-Isolation-Forest
   
2. **Install Dependencies**
   ```bash
   pip install streamlit pandas scikit-learn shap matplotlib numpy
   
3. Run the app
   ```bash
   streamlit run app.py

## 📊 Expected Data Format
The application expects a `.csv` file with (at minimum) the following columns:
- `login_attempt_count`: Number of attempts in a timeframe.
- `session_duration`: Time spent on the platform.
- `user_activity_frequency`: Actions performed per minute.

## 🧠 Logic Flow
- Data Ingestion: User uploads an access log CSV.
- Feature Engineering: The app calculates variance across features and selects the top 2 for 2D visualization.
- Isolation: The Isolation Forest isolates observations by randomly selecting a feature and then randomly selecting a split value.
- Scoring: Points that isolate quickly (short path lengths in the tree) are assigned a lower anomaly score (flagged as anomalies).
- Interpretation: SHAP values break down the contribution of each feature to the final outlier score.
