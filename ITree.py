import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import IsolationForest
from typing import Tuple, List

# --- Configuration & Styling ---
st.set_page_config(page_title="Cyber-Anomaly Detector", layout="wide")

def load_data(uploaded_file) -> pd.DataFrame:
    """Loads CSV data and caches it."""
    return pd.read_csv(uploaded_file)

def get_top_variance_features(df: pd.DataFrame, feature_cols: List[str], k: int = 2) -> List[str]:
    """Selects top K features based on variance (Unsupervised Selection)."""
    variances = df[feature_cols].var().sort_values(ascending=False)
    return variances.index[:k].tolist()

@st.cache_resource
def train_model(data: pd.DataFrame, contamination: float = 0.05):
    """Trains Isolation Forest and returns the model."""
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(data)
    return model

def plot_decision_boundary(model, X: pd.DataFrame, features: List[str]):
    """Generates the Isolation Forest decision boundary contour plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create mesh grid
    x_min, x_max = X[features[0]].min() - 0.1, X[features[0]].max() + 0.1
    y_min, y_max = X[features[1]].min() - 0.1, X[features[1]].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict over mesh
    mesh_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=features)
    Z = model.decision_function(mesh_df)
    Z = Z.reshape(xx.shape)

    # Plotting
    contour = ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn, alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black') # Boundary line
    plt.colorbar(contour, ax=ax, label="Anomaly Score")
    
    ax.scatter(X[features[0]], X[features[1]], c='blue', s=20, edgecolor='k', alpha=0.5)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("Isolation Forest Decision Boundary")
    return fig

# --- UI Layout ---
st.title("🛡️ Cybersecurity Anomaly Detection")
st.markdown("### Unsupervised Threat Hunting using Isolation Forest")

uploaded_file = st.file_uploader("Upload Access Log Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    required_cols = ["login_attempt_count", "session_duration", "user_activity_frequency"]
    
    # Verification
    if all(col in df.columns for col in required_cols):
        # 1. Feature Selection
        top2_features = get_top_variance_features(df, required_cols)
        X = df[top2_features]
        
        st.success(f"Selected Key Features: **{', '.join(top2_features)}**")

        # 2. Model Training
        with st.spinner("Analyzing patterns..."):
            model = train_model(X)
            df["anomaly_score"] = model.decision_function(X)
            df["is_anomaly"] = model.predict(X)
            df["is_anomaly"] = df["is_anomaly"].map({1: "Normal", -1: "Anomaly"})

        # 3. Dashboard Metrics
        col1, col2, col3 = st.columns(3)
        total_logs = len(df)
        anomaly_count = len(df[df["is_anomaly"] == "Anomaly"])
        col1.metric("Total Logs", total_logs)
        col2.metric("Anomalies Detected", anomaly_count, delta=f"{anomaly_count/total_logs:.2%}", delta_color="inverse")
        col3.metric("Model Status", "Active")

        # 4. Visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Scatter Analysis", "🗺️ Decision Boundary", "🧠 Explainability (SHAP)"])
        
        with tab1:
            st.subheader("Cluster Distribution")
            fig1, ax1 = plt.subplots()
            colors = {"Normal": "seagreen", "Anomaly": "crimson"}
            for label, color in colors.items():
                subset = df[df["is_anomaly"] == label]
                ax1.scatter(subset[top2_features[0]], subset[top2_features[1]], 
                           c=color, label=label, alpha=0.6, edgecolors='w')
            ax1.set_xlabel(top2_features[0])
            ax1.set_ylabel(top2_features[1])
            ax1.legend()
            st.pyplot(fig1)

        with tab2:
            st.subheader("Model's Latent 'Safe Zone'")
            st.pyplot(plot_decision_boundary(model, X, top2_features))

        with tab3:
            st.subheader("Why were these flagged?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            fig3, ax3 = plt.subplots()
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig3)

        # 5. Data Export
        st.subheader("Detailed Anomaly Report")
        anomalies_only = df[df["is_anomaly"] == "Anomaly"].sort_values(by="anomaly_score")
        st.dataframe(anomalies_only[top2_features + ["anomaly_score"]], use_container_width=True)
        
    else:
        st.error(f"Error: Dataset must contain these columns: {required_cols}")
else:
    st.info("Please upload a CSV file to begin analysis.")
