🛡️ Cybersecurity Anomaly Detection (Unsupervised)
An interactive web application built with Streamlit that uses Unsupervised Machine Learning to detect suspicious patterns in cybersecurity access logs.
This tool specifically utilizes the Isolation Forest algorithm to identify anomalies based on feature behavior, intentionally ignoring pre-existing labels to simulate a real-world "blind" threat-hunting scenario.
🚀 Overview
In modern cybersecurity, labeled datasets (knowing exactly who is an attacker) are rare. This project demonstrates how to find "needles in the haystack" using purely unsupervised methods. It analyzes user behavior features—like login frequency and session duration—to flag outliers that could represent unauthorized access or system abuse.
✨ Key Features
CSV Data Upload: Process custom access-log datasets on the fly.
Automated Feature Selection: Automatically identifies the top 2 features based on Variance to visualize the most significant data fluctuations.
Isolation Forest Modeling: Uses an ensemble-based anomaly detection algorithm to score and flag outliers.
Interactive Visualizations:
Scatter Plots: View predicted "Normal" vs. "Anomaly" points.
Decision Boundary: Visualize the "safety zones" the model has drawn around your data.
SHAP Explainability: Understand why the model flagged a specific log entry as an anomaly.
Severity Ranking: A filtered table showing the most severe anomalies (lowest decision scores) first.
🛠️ Tech Stack
Frontend: Streamlit
Machine Learning: Scikit-Learn (Isolation Forest)
Explainability: SHAP (SHapley Additive exPlanations)
Data Handling: Pandas & NumPy
Visualization: Matplotlib
📋 Dataset Requirements
To run the analysis, your uploaded CSV should contain the following columns:
anomaly_score
login_attempt_count
session_duration
user_activity_frequency
target (Note: This is ignored by the model and used only for data validation).
💻 Installation & Setup
Clone the repository
code
Bash
git clone https://github.com/your-username/cyber-anomaly-detection.git
cd cyber-anomaly-detection
Create a virtual environment (Optional but recommended)
code
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies
code
Bash
pip install streamlit pandas sklearn shap matplotlib numpy
Run the Application
code
Bash
streamlit run app.py
🧠 How it Works: The Logic
Variance Filtering: The app calculates the variance of all features. It selects the two features with the highest spread, as these usually contain the most information for distinguishing between "typical" and "atypical" behavior.
Isolation Forest: Unlike standard clustering (which finds similarities), Isolation Forest explicitly isolates anomalies. It builds random trees; because anomalies have rare values, they are isolated closer to the root of the tree than normal points.
Contamination: The model is set to a contamination=0.05 (5%), assuming that roughly 5% of your logs are likely suspicious.
Explainability: By using SHAP values, we can see which specific feature pushed a data point into the "Anomaly" zone (e.g., "This was flagged primarily because the login attempt count was 10x the average").
📄 License
This project is open-source and available under the MIT License.
