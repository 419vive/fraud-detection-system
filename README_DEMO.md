# IEEE-CIS Fraud Detection Demo Application

🛡️ **Interactive demonstration of the IEEE-CIS Fraud Detection System**

This comprehensive demo application showcases the capabilities of our advanced machine learning fraud detection system through an intuitive web interface built with Streamlit.

## 🌟 Features

### Core Functionality
- **Real-time Fraud Prediction**: Analyze individual transactions instantly
- **Batch Processing**: Upload and analyze multiple transactions simultaneously
- **Model Comparison**: Compare performance across different ML models
- **Interactive Scenarios**: Pre-configured test cases for demonstrations
- **Business Impact Calculator**: ROI analysis and cost-benefit calculations

### Technical Capabilities
- **Multi-Model Support**: LightGBM, XGBoost, Random Forest, CatBoost, Logistic Regression
- **Feature Engineering**: Advanced feature extraction and transformation
- **Explainable AI**: Feature importance and prediction explanations
- **Performance Monitoring**: Real-time system health and metrics
- **Data Visualization**: Interactive charts and business intelligence dashboards

### Demo Sections
1. **🏠 Dashboard Overview**: System status and key metrics
2. **🔍 Single Transaction Test**: Individual transaction analysis
3. **📁 Batch Prediction**: Multiple transaction processing
4. **📊 Model Performance**: Comprehensive model evaluation
5. **💰 Business Impact**: ROI calculator and financial analysis
6. **🎯 Demo Scenarios**: Pre-configured test cases
7. **⚙️ System Status**: Health monitoring and configuration

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository and navigate to the project directory
cd "DS PROJECT/project 2"

# Run the automated setup and launch script
python run_demo.py
```

This will automatically:
- Check Python version compatibility
- Install required dependencies
- Generate sample data
- Launch the demo application

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r demo_requirements.txt

# Generate sample data
python -c "from demo_utils import create_sample_datasets; create_sample_datasets()"

# Launch the app
streamlit run demo_app.py
```

### Option 3: With Prediction API

```bash
# Launch both the demo app and the FastAPI prediction service
python run_demo.py --launch --with-api
```

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.10.0
scikit-learn>=1.1.0
joblib>=1.2.0
requests>=2.28.0
```

## 🎮 How to Use

### 1. Dashboard Overview
- View system status and key performance indicators
- Monitor transaction volume and fraud detection rates
- Access quick navigation to all demo sections

### 2. Single Transaction Testing
```python
# Example transaction for testing
{
    "TransactionID": 12345,
    "TransactionAmt": 1299.99,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "user@gmail.com",
    "hour": 14,
    "is_weekend": 0
}
```

### 3. Batch Processing
- Upload CSV files with transaction data
- Process up to 10,000 transactions simultaneously
- Download results with fraud predictions and risk scores

### 4. Demo Scenarios

#### 🟢 Legitimate Transaction
- Normal business hours transaction
- Trusted email domain
- Reasonable amount for product type

#### 🟡 Suspicious Transaction  
- High transaction amount
- Unusual timing (late night/early morning)
- Temporary email domain

#### 🔴 High-Risk Transaction
- Very high amount
- Multiple red flags
- Suspicious patterns

#### 💳 Card Testing Pattern
- Small amounts ($1-10)
- Multiple rapid transactions
- Privacy-focused email domains

### 5. Business Impact Analysis
- Calculate ROI based on system parameters
- Analyze cost savings from fraud prevention
- Compare investigation costs vs. fraud losses
- Generate 12-month financial projections

## 📊 Sample Data

The demo includes three pre-generated datasets:

- **Small**: 100 transactions (5% fraud rate)
- **Medium**: 1,000 transactions (3.5% fraud rate)  
- **Large**: 10,000 transactions (3% fraud rate)

### Data Features
- **Transaction Details**: Amount, product code, timestamp
- **Card Information**: Type, category, issuer details
- **User Information**: Email domains, addresses
- **Behavioral Features**: Time patterns, spending habits
- **Risk Indicators**: Derived fraud indicators

## 🔧 Configuration

### Demo Configuration (`demo_config.json`)
```json
{
  "demo_settings": {
    "default_model": "lightgbm",
    "fraud_threshold": 0.5,
    "max_batch_size": 10000
  },
  "business_settings": {
    "average_transaction_value": 150.0,
    "investigation_cost": 25.0,
    "chargeback_cost": 75.0,
    "system_cost_monthly": 50000.0
  }
}
```

### Customization Options
- Adjust fraud detection thresholds
- Modify business cost parameters
- Configure UI display options
- Set performance monitoring alerts

## 🎯 Use Cases

### For Stakeholder Demonstrations
- **Executive Presentations**: Business impact and ROI analysis
- **Technical Reviews**: Model performance and architecture
- **Product Demos**: Real-time fraud detection capabilities
- **Sales Presentations**: Competitive advantages and features

### For Development and Testing
- **Model Validation**: Compare different algorithms
- **Feature Testing**: Evaluate new feature importance
- **Performance Benchmarking**: System load and response times
- **Integration Testing**: API connectivity and data flow

### For Training and Education
- **Team Training**: Fraud detection concepts and methods
- **Customer Onboarding**: System capabilities and usage
- **Academic Presentations**: Machine learning in finance
- **Compliance Demonstrations**: Regulatory requirement compliance

## 🔍 Advanced Features

### Model Explanations
- Feature importance rankings
- SHAP value analysis (simulated)
- Decision boundary visualization
- Prediction confidence levels

### Performance Monitoring
- Real-time prediction latency
- System resource utilization
- Model accuracy tracking
- Alert notifications

### Business Intelligence
- Fraud pattern analysis
- Revenue impact calculations
- Cost-benefit optimization
- Risk assessment reports

## 🚦 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Use a different port
python run_demo.py --launch --port 8502
```

#### Missing Dependencies
```bash
# Force reinstall dependencies
python run_demo.py --setup --force-install
```

#### Memory Issues
```bash
# Reduce sample data size in demo_utils.py
# Or increase system memory allocation
```

#### Streamlit Not Found
```bash
pip install streamlit
# Or use the full setup
python run_demo.py --setup
```

### Performance Optimization
- Use smaller datasets for faster loading
- Enable browser caching
- Optimize visualization refresh rates
- Monitor system resource usage

## 📈 Monitoring and Analytics

### Key Metrics Tracked
- **Prediction Accuracy**: Real-time model performance
- **Response Time**: API and UI response latency
- **Throughput**: Transactions processed per minute
- **System Health**: Resource utilization and availability

### Business Metrics
- **Fraud Detection Rate**: Percentage of fraud caught
- **False Positive Rate**: Legitimate transactions flagged
- **Cost Savings**: Financial impact of fraud prevention
- **ROI**: Return on investment calculations

## 🔒 Security Considerations

### Data Privacy
- Sample data is synthetically generated
- No real customer information is used
- Data is processed locally by default
- Optional API integration for production scenarios

### Production Deployment
- Use HTTPS for web interface
- Implement authentication and authorization
- Encrypt sensitive data in transit and at rest
- Regular security audits and updates

## 🤝 Support and Contribution

### Getting Help
- Check the troubleshooting section above
- Review log files for error messages
- Test with different browsers
- Verify system requirements are met

### Customization
- Modify `demo_utils.py` for custom data generation
- Update `demo_app.py` for UI customizations
- Adjust configuration in `demo_config.json`
- Extend model simulation in `ModelSimulator` class

## 📝 License and Usage

This demo application is part of the IEEE-CIS Fraud Detection project and is intended for:
- Educational and demonstration purposes
- Internal stakeholder presentations
- Development and testing scenarios
- Academic research and learning

For production use, please ensure compliance with relevant data protection regulations and implement appropriate security measures.

---

**🛡️ Built with advanced machine learning for fraud detection excellence**

For questions, support, or customization requests, please contact the development team.