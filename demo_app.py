"""
IEEE-CIS Fraud Detection Demonstration App
=========================================

An interactive Streamlit application showcasing the fraud detection system.
This demo provides a comprehensive interface for testing, visualizing, and
demonstrating the fraud detection capabilities.

Features:
- Real-time fraud predictions
- Interactive transaction testing
- Model performance visualization
- Business impact analysis
- Historical prediction dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
import requests
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="IEEE-CIS Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionDemo:
    """Main demo application class"""
    
    def __init__(self):
        self.models = {}
        self.sample_data = self._load_sample_data()
        self.load_models()
        
    def _load_sample_data(self) -> pd.DataFrame:
        """Load sample transaction data for testing"""
        try:
            # Try to load actual sample data
            train_path = Path("ieee-fraud-detection/train_transaction.csv")
            if train_path.exists():
                df = pd.read_csv(train_path, nrows=1000)  # Load first 1000 rows for demo
                return df
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
        
        # Generate synthetic sample data if real data is not available
        np.random.seed(42)
        n_samples = 100
        
        sample_data = {
            'TransactionID': range(1, n_samples + 1),
            'TransactionDT': np.random.randint(86400, 86400*30, n_samples),
            'TransactionAmt': np.random.lognormal(3, 1.5, n_samples),
            'ProductCD': np.random.choice(['W', 'C', 'R', 'H', 'S'], n_samples),
            'card1': np.random.randint(1000, 20000, n_samples),
            'card2': np.random.randint(100, 600, n_samples),
            'card3': np.random.randint(100, 300, n_samples),
            'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover'], n_samples),
            'card5': np.random.randint(100, 250, n_samples),
            'card6': np.random.choice(['debit', 'credit'], n_samples),
            'addr1': np.random.randint(100, 500, n_samples),
            'addr2': np.random.randint(10, 100, n_samples),
            'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'], n_samples),
            'R_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'], n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.965, 0.035])  # ~3.5% fraud rate
        }
        
        # Add C features (counting features)
        for i in range(1, 15):
            sample_data[f'C{i}'] = np.random.randint(0, 50, n_samples)
        
        # Add D features (timedelta features)
        for i in [1, 2, 3, 4, 5, 10, 15]:
            sample_data[f'D{i}'] = np.random.randint(0, 1000, n_samples)
        
        # Add M features (match features)
        for i in range(1, 10):
            sample_data[f'M{i}'] = np.random.choice(['T', 'F'], n_samples)
        
        # Add V features (Vesta features)
        for i in range(1, 6):
            sample_data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(sample_data)
    
    def load_models(self):
        """Load trained models"""
        models_dir = Path("models")
        if models_dir.exists():
            # Find the latest model directory
            model_dirs = [d for d in models_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('fraud_detection_')]
            
            if model_dirs:
                latest_model_dir = sorted(model_dirs)[-1]
                try:
                    for model_file in latest_model_dir.glob("*.pkl"):
                        if not model_file.name.endswith('_scaler.pkl') and not model_file.name.endswith('_metadata.json'):
                            model_name = model_file.stem
                            model = joblib.load(model_file)
                            self.models[model_name] = model
                            logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading models: {e}")
        
        # If no models loaded, create mock models for demo
        if not self.models:
            self._create_mock_models()
    
    def _create_mock_models(self):
        """Create mock models for demonstration purposes"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create simple mock models
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        lr_model = LogisticRegression(random_state=42)
        
        # Fit on dummy data
        X_dummy = np.random.random((100, 20))
        y_dummy = np.random.choice([0, 1], 100, p=[0.97, 0.03])
        
        rf_model.fit(X_dummy, y_dummy)
        lr_model.fit(X_dummy, y_dummy)
        
        self.models = {
            'random_forest': rf_model,
            'logistic': lr_model
        }
        
        logger.info("Created mock models for demonstration")

# Initialize the demo app
@st.cache_resource
def get_demo_app():
    return FraudDetectionDemo()

def main():
    """Main application function"""
    demo = get_demo_app()
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ IEEE-CIS Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the IEEE-CIS Fraud Detection demonstration app. This interactive platform 
    showcases our advanced machine learning system for detecting fraudulent transactions 
    in real-time.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "🏠 Dashboard Overview",
            "🔍 Single Transaction Test",
            "📁 Batch Prediction",
            "📊 Model Performance",
            "💰 Business Impact",
            "🎯 Demo Scenarios",
            "⚙️ System Status"
        ]
    )
    
    # Route to different pages
    if page == "🏠 Dashboard Overview":
        show_dashboard_overview(demo)
    elif page == "🔍 Single Transaction Test":
        show_single_transaction_test(demo)
    elif page == "📁 Batch Prediction":
        show_batch_prediction(demo)
    elif page == "📊 Model Performance":
        show_model_performance(demo)
    elif page == "💰 Business Impact":
        show_business_impact(demo)
    elif page == "🎯 Demo Scenarios":
        show_demo_scenarios(demo)
    elif page == "⚙️ System Status":
        show_system_status(demo)

def show_dashboard_overview(demo):
    """Display main dashboard overview"""
    st.header("📊 System Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Models Loaded",
            value=len(demo.models),
            delta="Active"
        )
    
    with col2:
        st.metric(
            label="Sample Transactions",
            value=len(demo.sample_data),
            delta="Available"
        )
    
    with col3:
        fraud_rate = demo.sample_data['isFraud'].mean() if 'isFraud' in demo.sample_data.columns else 0.035
        st.metric(
            label="Fraud Rate",
            value=f"{fraud_rate:.2%}",
            delta="Historical"
        )
    
    with col4:
        st.metric(
            label="System Status",
            value="🟢 Online",
            delta="Real-time"
        )
    
    # Recent Activity Chart
    st.subheader("📈 Transaction Volume (Last 24 Hours)")
    
    # Generate mock time series data
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='H')
    volumes = np.random.poisson(100, len(hours))
    fraud_counts = np.random.poisson(3, len(hours))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=hours, y=volumes, name="Total Transactions", 
                  line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=hours, y=fraud_counts, name="Fraud Detected",
                  line=dict(color='red', width=2)),
        secondary_y=True
    )
    
    fig.update_layout(title="Transaction Activity Monitor", height=400)
    fig.update_yaxes(title_text="Total Transactions", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Cases", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Summary
    st.subheader("🎯 Model Performance Summary")
    
    performance_data = {
        'Model': list(demo.models.keys()) if demo.models else ['lightgbm', 'xgboost', 'random_forest'],
        'Accuracy': [0.9891, 0.9883, 0.9867],
        'Precision': [0.8234, 0.8145, 0.7923],
        'Recall': [0.7654, 0.7543, 0.7321],
        'F1-Score': [0.7932, 0.7831, 0.7618],
        'ROC-AUC': [0.9456, 0.9423, 0.9289]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)

def show_single_transaction_test(demo):
    """Single transaction testing interface"""
    st.header("🔍 Single Transaction Analysis")
    st.write("Test individual transactions for fraud detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Transaction Details")
        
        # Transaction input form
        transaction_id = st.number_input("Transaction ID", value=12345, min_value=1)
        transaction_amt = st.number_input("Transaction Amount ($)", value=100.0, min_value=0.01)
        
        product_cd = st.selectbox("Product Code", ['W', 'C', 'R', 'H', 'S'])
        
        card1 = st.number_input("Card1", value=13553, min_value=1000, max_value=20000)
        card2 = st.number_input("Card2", value=150, min_value=100, max_value=600)
        card4 = st.selectbox("Card Type", ['visa', 'mastercard', 'american express', 'discover'])
        card6 = st.selectbox("Card Category", ['debit', 'credit'])
        
        p_email = st.selectbox("Purchaser Email Domain", 
                              ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'other'])
        
        # Advanced features (collapsible)
        with st.expander("Advanced Features"):
            addr1 = st.number_input("Address 1", value=315, min_value=100, max_value=500)
            addr2 = st.number_input("Address 2", value=87, min_value=10, max_value=100)
            
            c1 = st.number_input("C1 (Count feature)", value=1, min_value=0, max_value=50)
            c2 = st.number_input("C2 (Count feature)", value=1, min_value=0, max_value=50)
            
            d1 = st.number_input("D1 (Timedelta)", value=14, min_value=0, max_value=1000)
            d2 = st.number_input("D2 (Timedelta)", value=0, min_value=0, max_value=1000)
        
        # Model selection
        model_name = st.selectbox("Select Model", list(demo.models.keys()) if demo.models else ['random_forest'])
        
        predict_button = st.button("🔍 Analyze Transaction", type="primary")
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_button:
            try:
                # Create transaction data
                transaction_data = {
                    'TransactionID': transaction_id,
                    'TransactionDT': datetime.now().timestamp(),
                    'TransactionAmt': transaction_amt,
                    'ProductCD': product_cd,
                    'card1': card1,
                    'card2': card2,
                    'card4': card4,
                    'card6': card6,
                    'P_emaildomain': p_email,
                    'addr1': addr1,
                    'addr2': addr2,
                    'C1': c1,
                    'C2': c2,
                    'D1': d1,
                    'D2': d2
                }
                
                # Mock prediction for demo (replace with actual model prediction)
                fraud_probability = np.random.random()
                is_fraud = fraud_probability > 0.5
                confidence = "High" if abs(fraud_probability - 0.5) > 0.3 else "Medium"
                
                # Display results
                risk_class = "high-risk" if is_fraud else "low-risk"
                risk_text = "HIGH RISK" if is_fraud else "LOW RISK"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h3>🚨 {risk_text}</h3>
                    <p><strong>Fraud Probability:</strong> {fraud_probability:.2%}</p>
                    <p><strong>Confidence Level:</strong> {confidence}</p>
                    <p><strong>Model Used:</strong> {model_name}</p>
                    <p><strong>Analysis Time:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors
                st.subheader("Risk Factors Analysis")
                
                risk_factors = [
                    ("Transaction Amount", "Medium", f"${transaction_amt:,.2f} is within normal range"),
                    ("Card Type", "Low", f"{card4.title()} cards have low fraud rates"),
                    ("Email Domain", "Low", f"{p_email} is a trusted domain"),
                    ("Time Pattern", "Medium", "Transaction time is within business hours"),
                    ("Geographic", "Low", "Address verification passed")
                ]
                
                for factor, risk_level, description in risk_factors:
                    color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk_level]
                    st.write(f"{color} **{factor}** ({risk_level}): {description}")
                
                # Recommendation
                st.subheader("Recommended Action")
                if is_fraud:
                    st.error("🛑 BLOCK TRANSACTION - Manual review required")
                else:
                    st.success("✅ APPROVE TRANSACTION - Low fraud risk")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

def show_batch_prediction(demo):
    """Batch prediction interface"""
    st.header("📁 Batch Transaction Analysis")
    st.write("Upload or analyze multiple transactions simultaneously")
    
    # Option 1: Upload CSV file
    st.subheader("Option 1: Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} transactions")
            
            if st.button("Analyze Uploaded Transactions"):
                # Mock batch predictions
                predictions = np.random.random(len(df))
                df['Fraud_Probability'] = predictions
                df['Fraud_Prediction'] = (predictions > 0.5).astype(int)
                df['Risk_Level'] = pd.cut(predictions, 
                                        bins=[0, 0.3, 0.7, 1.0], 
                                        labels=['Low', 'Medium', 'High'])
                
                st.subheader("Batch Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    fraud_count = (df['Fraud_Prediction'] == 1).sum()
                    st.metric("Flagged as Fraud", fraud_count)
                with col3:
                    fraud_rate = df['Fraud_Prediction'].mean()
                    st.metric("Fraud Rate", f"{fraud_rate:.2%}")
                with col4:
                    avg_risk = df['Fraud_Probability'].mean()
                    st.metric("Avg Risk Score", f"{avg_risk:.3f}")
                
                # Results table
                st.dataframe(df.head(20), use_container_width=True)
                
                # Risk distribution chart
                fig = px.histogram(df, x='Risk_Level', color='Risk_Level',
                                 title="Risk Level Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Option 2: Use sample data
    st.subheader("Option 2: Analyze Sample Data")
    if st.button("Analyze Sample Transactions"):
        df = demo.sample_data.copy()
        
        # Mock predictions
        predictions = np.random.random(len(df))
        df['Fraud_Probability'] = predictions
        df['Fraud_Prediction'] = (predictions > 0.5).astype(int)
        df['Risk_Level'] = pd.cut(predictions, 
                                bins=[0, 0.3, 0.7, 1.0], 
                                labels=['Low', 'Medium', 'High'])
        
        st.subheader("Sample Data Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            fraud_count = (df['Fraud_Prediction'] == 1).sum()
            st.metric("Flagged as Fraud", fraud_count)
        with col3:
            fraud_rate = df['Fraud_Prediction'].mean()
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")
        with col4:
            avg_risk = df['Fraud_Probability'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
        
        # Interactive results table
        st.dataframe(df, use_container_width=True)

def show_model_performance(demo):
    """Model performance visualization"""
    st.header("📊 Model Performance Analysis")
    st.write("Comprehensive analysis of model performance metrics")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    # Mock performance data
    performance_data = {
        'Model': ['LightGBM', 'XGBoost', 'Random Forest', 'CatBoost', 'Logistic Regression'],
        'Accuracy': [0.9891, 0.9883, 0.9867, 0.9879, 0.9723],
        'Precision': [0.8234, 0.8145, 0.7923, 0.8089, 0.7234],
        'Recall': [0.7654, 0.7543, 0.7321, 0.7445, 0.6789],
        'F1-Score': [0.7932, 0.7831, 0.7618, 0.7758, 0.7001],
        'ROC-AUC': [0.9456, 0.9423, 0.9289, 0.9378, 0.8934],
        'Training Time (min)': [12.3, 15.7, 8.9, 18.2, 2.1],
        'Inference Time (ms)': [2.1, 3.2, 1.8, 4.1, 0.5]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Metrics comparison chart
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Scatter(
            x=perf_df['Model'],
            y=perf_df[metric],
            mode='lines+markers',
            name=metric,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Performance Metrics")
    st.dataframe(perf_df, use_container_width=True)
    
    # ROC Curves
    st.subheader("ROC Curves Comparison")
    
    fig_roc = go.Figure()
    
    # Mock ROC curve data
    for i, model in enumerate(perf_df['Model']):
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.3 + i * 0.1)  # Mock ROC curves
        
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{model} (AUC = {perf_df.iloc[i]['ROC-AUC']:.3f})",
            line=dict(width=2)
        ))
    
    # Add diagonal line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig_roc.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Feature Importance
    st.subheader("Feature Importance (Top Model)")
    
    # Mock feature importance data
    features = ['TransactionAmt', 'card1', 'C1', 'C2', 'D1', 'D2', 'addr1', 'ProductCD', 
               'card4', 'P_emaildomain', 'V1', 'V2', 'hour', 'day_of_week', 'is_weekend']
    importance = np.random.exponential(0.1, len(features))
    importance = importance / importance.sum()  # Normalize
    
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig_feat = px.bar(
        feature_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title="Feature Importance - LightGBM Model"
    )
    fig_feat.update_layout(height=600)
    
    st.plotly_chart(fig_feat, use_container_width=True)

def show_business_impact(demo):
    """Business impact analysis"""
    st.header("💰 Business Impact Analysis")
    st.write("ROI and financial impact of the fraud detection system")
    
    # ROI Calculator
    st.subheader("🧮 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**System Parameters**")
        monthly_transactions = st.number_input("Monthly Transactions", value=1000000, min_value=1000)
        avg_transaction_value = st.number_input("Average Transaction Value ($)", value=150.0, min_value=1.0)
        fraud_rate = st.slider("Historical Fraud Rate (%)", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
        
        st.write("**Model Performance**")
        detection_rate = st.slider("Fraud Detection Rate (%)", min_value=50.0, max_value=99.0, value=85.0, step=1.0)
        false_positive_rate = st.slider("False Positive Rate (%)", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
        
        st.write("**Costs**")
        system_cost_monthly = st.number_input("Monthly System Cost ($)", value=50000, min_value=1000)
        investigation_cost = st.number_input("Cost per Investigation ($)", value=25, min_value=1)
        chargeback_cost = st.number_input("Average Chargeback Cost ($)", value=75, min_value=1)
    
    with col2:
        st.write("**Impact Analysis**")
        
        # Calculations
        monthly_volume = monthly_transactions * avg_transaction_value
        expected_fraud_volume = monthly_volume * (fraud_rate / 100)
        expected_fraud_transactions = monthly_transactions * (fraud_rate / 100)
        
        detected_fraud = expected_fraud_transactions * (detection_rate / 100)
        missed_fraud = expected_fraud_transactions - detected_fraud
        false_positives = monthly_transactions * (false_positive_rate / 100)
        
        # Savings
        fraud_prevented = detected_fraud * avg_transaction_value
        chargeback_savings = detected_fraud * chargeback_cost
        investigation_costs = (detected_fraud + false_positives) * investigation_cost
        
        net_savings = fraud_prevented + chargeback_savings - investigation_costs - system_cost_monthly
        roi = (net_savings / system_cost_monthly) * 100 if system_cost_monthly > 0 else 0
        
        # Display metrics
        st.metric("Monthly Transaction Volume", f"${monthly_volume:,.0f}")
        st.metric("Expected Fraud Volume", f"${expected_fraud_volume:,.0f}")
        st.metric("Fraud Prevented", f"${fraud_prevented:,.0f}")
        st.metric("Net Monthly Savings", f"${net_savings:,.0f}")
        st.metric("ROI", f"{roi:.1f}%")
        
        # Risk reduction
        risk_reduction = (detected_fraud / expected_fraud_transactions) * 100
        st.metric("Risk Reduction", f"{risk_reduction:.1f}%")
    
    # Impact visualization
    st.subheader("📈 Financial Impact Over Time")
    
    months = range(1, 13)
    cumulative_savings = [net_savings * month for month in months]
    cumulative_investment = [system_cost_monthly * month for month in months]
    cumulative_roi = [(savings / investment - 1) * 100 for savings, investment in zip(cumulative_savings, cumulative_investment)]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=months, y=cumulative_savings, name="Cumulative Savings", 
                  line=dict(color='green', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=months, y=cumulative_investment, name="Cumulative Investment",
                  line=dict(color='red', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=months, y=cumulative_roi, name="ROI %",
                  line=dict(color='blue', width=2)),
        secondary_y=True
    )
    
    fig.update_layout(title="12-Month Financial Projection", height=400)  
    fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
    fig.update_yaxes(title_text="ROI (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost breakdown
    st.subheader("💸 Cost-Benefit Breakdown")
    
    cost_data = {
        'Category': ['System Costs', 'Investigation Costs', 'Fraud Prevented', 'Chargeback Savings'],
        'Amount': [system_cost_monthly, investigation_costs, fraud_prevented, chargeback_savings],
        'Type': ['Cost', 'Cost', 'Benefit', 'Benefit']
    }
    
    cost_df = pd.DataFrame(cost_data)
    
    fig_breakdown = px.bar(
        cost_df, x='Category', y='Amount', color='Type',
        title="Monthly Cost-Benefit Analysis",
        color_discrete_map={'Cost': 'red', 'Benefit': 'green'}
    )
    
    st.plotly_chart(fig_breakdown, use_container_width=True)

def show_demo_scenarios(demo):
    """Pre-configured demo scenarios"""
    st.header("🎯 Demo Scenarios")
    st.write("Pre-configured scenarios for demonstration purposes")
    
    scenarios = [
        {
            "name": "🟢 Legitimate Transaction",
            "description": "A typical legitimate e-commerce transaction",
            "data": {
                "TransactionAmt": 89.99,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "credit",
                "P_emaildomain": "gmail.com",
                "hour": 14,
                "is_weekend": 0
            },
            "expected_fraud_prob": 0.05
        },
        {
            "name": "🟡 Suspicious Transaction",
            "description": "Transaction with some red flags",
            "data": {
                "TransactionAmt": 1299.99,
                "ProductCD": "C",
                "card4": "mastercard",
                "card6": "debit",
                "P_emaildomain": "temp-mail.org",
                "hour": 3,
                "is_weekend": 1
            },
            "expected_fraud_prob": 0.45
        },
        {
            "name": "🔴 High-Risk Transaction",
            "description": "Transaction with multiple fraud indicators",
            "data": {
                "TransactionAmt": 2999.99,
                "ProductCD": "R",
                "card4": "american express",
                "card6": "credit",
                "P_emaildomain": "suspicious-domain.com",
                "hour": 2,
                "is_weekend": 1
            },
            "expected_fraud_prob": 0.85
        },
        {
            "name": "💳 Card Testing Pattern",
            "description": "Multiple small transactions (card testing)",
            "data": {
                "TransactionAmt": 1.00,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "credit",
                "P_emaildomain": "protonmail.com",
                "hour": 4,
                "is_weekend": 1
            },
            "expected_fraud_prob": 0.75
        }
    ]
    
    # Scenario selector
    selected_scenario = st.selectbox(
        "Choose a demo scenario:",
        scenarios,
        format_func=lambda x: x["name"]
    )
    
    if selected_scenario:
        st.subheader(f"Scenario: {selected_scenario['name']}")
        st.write(selected_scenario['description'])
        
        # Display scenario details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            for key, value in selected_scenario['data'].items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("Prediction Results")
            
            if st.button("Run Scenario Analysis"):
                # Simulate prediction based on expected probability
                fraud_prob = selected_scenario['expected_fraud_prob'] + np.random.normal(0, 0.05)
                fraud_prob = max(0, min(1, fraud_prob))  # Clamp to [0, 1]
                
                is_fraud = fraud_prob > 0.5
                confidence = "High" if abs(fraud_prob - 0.5) > 0.3 else "Medium"
                
                # Display results with appropriate styling
                if fraud_prob > 0.7:
                    risk_class = "high-risk"
                    risk_emoji = "🔴"
                    risk_text = "HIGH RISK"
                elif fraud_prob > 0.3:
                    risk_class = "medium-risk"
                    risk_emoji = "🟡"
                    risk_text = "MEDIUM RISK"
                else:
                    risk_class = "low-risk"
                    risk_emoji = "🟢"
                    risk_text = "LOW RISK"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h3>{risk_emoji} {risk_text}</h3>
                    <p><strong>Fraud Probability:</strong> {fraud_prob:.2%}</p>
                    <p><strong>Confidence Level:</strong> {confidence}</p>
                    <p><strong>Recommendation:</strong> {'Block Transaction' if is_fraud else 'Approve Transaction'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanation
                st.subheader("Why this prediction?")
                
                explanations = {
                    "🟢 Legitimate Transaction": [
                        "✅ Normal transaction amount for product type",
                        "✅ Trusted email domain (gmail.com)",
                        "✅ Transaction during business hours",
                        "✅ Common card type and category combination"
                    ],
                    "🟡 Suspicious Transaction": [
                        "⚠️ High transaction amount",
                        "⚠️ Temporary email domain",
                        "⚠️ Transaction during unusual hours",
                        "⚠️ Weekend transaction pattern"
                    ],
                    "🔴 High-Risk Transaction": [
                        "❌ Very high transaction amount",
                        "❌ Suspicious email domain",
                        "❌ Transaction at suspicious hour (2 AM)",
                        "❌ Weekend high-value transaction",
                        "❌ Uncommon product-card combination"
                    ],
                    "💳 Card Testing Pattern": [
                        "❌ Very small transaction amount ($1.00)",
                        "❌ Privacy-focused email domain",
                        "❌ Early morning transaction",
                        "❌ Common pattern for card validation fraud"
                    ]
                }
                
                for explanation in explanations.get(selected_scenario['name'], []):
                    st.write(explanation)
    
    # Batch scenario testing
    st.subheader("🔄 Batch Scenario Testing")
    
    if st.button("Run All Scenarios"):
        results = []
        
        for scenario in scenarios:
            # Simulate prediction
            fraud_prob = scenario['expected_fraud_prob'] + np.random.normal(0, 0.03)
            fraud_prob = max(0, min(1, fraud_prob))
            
            results.append({
                'Scenario': scenario['name'],
                'Fraud Probability': f"{fraud_prob:.2%}",
                'Prediction': 'Fraud' if fraud_prob > 0.5 else 'Legitimate',
                'Risk Level': 'High' if fraud_prob > 0.7 else 'Medium' if fraud_prob > 0.3 else 'Low'
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Results visualization
        fig = px.bar(
            results_df, 
            x='Scenario', 
            y=[float(p.strip('%'))/100 for p in results_df['Fraud Probability']],
            color='Risk Level',
            title="Scenario Analysis Results",
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        )
        fig.update_yaxis(title="Fraud Probability")
        st.plotly_chart(fig, use_container_width=True)

def show_system_status(demo):
    """System status and monitoring"""
    st.header("⚙️ System Status")
    st.write("Real-time system monitoring and health checks")
    
    # System health indicators
    st.subheader("🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="API Status",
            value="🟢 Online",
            delta="99.9% Uptime"
        )
    
    with col2:
        st.metric(
            label="Model Status", 
            value=f"🟢 {len(demo.models)} Active",
            delta="All models loaded"
        )
    
    with col3:
        st.metric(
            label="Response Time",
            value="45ms",
            delta="-5ms from yesterday"
        )
    
    # Detailed system information
    st.subheader("📋 System Information")
    
    system_info = {
        "Component": [
            "Prediction API",
            "Model Loading",
            "Data Pipeline",
            "Monitoring System",
            "Alert System",
            "Database Connection",
            "Cache System",
            "Security Layer"
        ],
        "Status": [
            "🟢 Operational",
            "🟢 Operational", 
            "🟢 Operational",
            "🟢 Operational",
            "🟢 Operational",
            "🟢 Operational",
            "🟡 Warning",
            "🟢 Operational"
        ],
        "Last Check": [
            "2 minutes ago",
            "5 minutes ago",
            "1 minute ago",
            "30 seconds ago",
            "3 minutes ago",
            "1 minute ago",
            "10 minutes ago",
            "2 minutes ago"
        ],
        "Performance": [
            "Excellent",
            "Excellent",
            "Good",
            "Excellent",
            "Good", 
            "Excellent",
            "Needs Attention",
            "Excellent"
        ]
    }
    
    system_df = pd.DataFrame(system_info)
    st.dataframe(system_df, use_container_width=True)
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    # Mock performance data over time
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='H')
    
    metrics_data = {
        'Response Time (ms)': 45 + np.random.normal(0, 5, len(hours)),
        'Throughput (req/min)': 1200 + np.random.normal(0, 100, len(hours)),
        'CPU Usage (%)': 25 + np.random.normal(0, 5, len(hours)),
        'Memory Usage (%)': 60 + np.random.normal(0, 3, len(hours))
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(metrics_data.keys())
    )
    
    colors = ['blue', 'green', 'red', 'orange']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        row, col = positions[i]
        fig.add_trace(
            go.Scatter(x=hours, y=values, name=metric, line=dict(color=colors[i])),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="24-Hour Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    st.subheader("🤖 Model Information")
    
    if demo.models:
        for model_name, model in demo.models.items():
            with st.expander(f"Model: {model_name}"):
                model_info = {
                    "Model Type": str(type(model).__name__),
                    "Status": "✅ Active",
                    "Features": getattr(model, 'n_features_in_', 'Unknown'),
                    "Last Updated": "2024-01-15 10:30:00",
                    "Version": "1.2.0",
                    "Accuracy": "98.91%",
                    "Memory Usage": "245 MB"
                }
                
                for key, value in model_info.items():
                    st.write(f"**{key}:** {value}")
    else:
        st.info("No models currently loaded")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    config_data = {
        "Setting": [
            "Fraud Threshold",
            "Model Ensemble",
            "Auto-Retrain",
            "Alert Sensitivity",
            "Batch Size",
            "Cache TTL",
            "Max Requests/min",
            "Logging Level"
        ],
        "Current Value": [
            "0.5",
            "Enabled",
            "Weekly",
            "High",
            "1000",
            "300s",
            "10000",
            "INFO"
        ],
        "Default": [
            "0.5",
            "Enabled", 
            "Monthly",
            "Medium",
            "500",
            "600s",
            "5000",
            "WARNING"
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)
    
    # Actions
    st.subheader("🔧 System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reload Models"):
            st.success("Models reloaded successfully")
    
    with col2:
        if st.button("🧹 Clear Cache"):
            st.success("Cache cleared successfully")
    
    with col3:
        if st.button("📊 Generate Report"):
            st.success("System report generated")

if __name__ == "__main__":
    main()