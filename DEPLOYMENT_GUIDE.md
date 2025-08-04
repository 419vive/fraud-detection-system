# IEEE-CIS Fraud Detection Demo - Deployment Guide

This guide provides comprehensive instructions for deploying the fraud detection demo application in various environments.

## 🚀 Quick Deployment

### Local Development
```bash
# Navigate to project directory
cd "/Users/jerrylaivivemachi/DS PROJECT/project 2"

# Automated setup and launch
python run_demo.py

# Or step by step:
python run_demo.py --setup    # Setup only
python run_demo.py --launch   # Launch only
```

The application will be available at: `http://localhost:8501`

### Custom Configuration
```bash
# Launch on different port
python run_demo.py --launch --port 8502

# Launch with API service
python run_demo.py --launch --with-api

# Force dependency reinstall
python run_demo.py --setup --force-install
```

## 🌐 Production Deployment

### 1. Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY demo_requirements.txt .
RUN pip install -r demo_requirements.txt

# Copy application files
COPY demo_app.py demo_utils.py run_demo.py ./
COPY demo_data/ ./demo_data/
COPY demo_config.json .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "demo_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
# Build image
docker build -t fraud-detection-demo .

# Run container
docker run -p 8501:8501 fraud-detection-demo
```

### 2. Cloud Deployment

#### Streamlit Community Cloud
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and select `demo_app.py`
4. Deploy automatically

#### Heroku Deployment
```bash
# Create Procfile
echo "web: streamlit run demo_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create fraud-detection-demo
git push heroku main
```

#### AWS EC2 Deployment
```bash
# On EC2 instance
sudo yum update -y
sudo yum install python3 python3-pip -y

# Clone repository
git clone <repository-url>
cd fraud-detection-demo

# Install dependencies
pip3 install -r demo_requirements.txt

# Run with nohup for background execution
nohup streamlit run demo_app.py --server.port=8501 --server.address=0.0.0.0 &
```

### 3. Kubernetes Deployment

Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fraud-detection-demo
  template:
    metadata:
      labels:
        app: fraud-detection-demo
    spec:
      containers:
      - name: demo-app
        image: fraud-detection-demo:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-demo-service
spec:
  selector:
    app: fraud-detection-demo
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

## 🔧 Environment Configuration

### Environment Variables
```bash
# Optional environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Demo-specific variables
export DEMO_DEFAULT_MODEL=lightgbm
export DEMO_MAX_BATCH_SIZE=10000
export DEMO_ENABLE_API=false
```

### Configuration File (`demo_config.json`)
```json
{
  "demo_settings": {
    "default_model": "lightgbm",
    "fraud_threshold": 0.5,
    "max_batch_size": 10000,
    "enable_explanations": true
  },
  "model_settings": {
    "available_models": ["lightgbm", "xgboost", "random_forest"],
    "model_refresh_interval": 3600,
    "fallback_model": "random_forest"
  },
  "ui_settings": {
    "default_page": "Dashboard Overview",
    "enable_advanced_features": true,
    "max_display_rows": 1000
  },
  "business_settings": {
    "average_transaction_value": 150.0,
    "investigation_cost": 25.0,
    "chargeback_cost": 75.0,
    "system_cost_monthly": 50000.0
  }
}
```

## 📊 Performance Optimization

### Resource Requirements
- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB RAM, 2 CPU cores
- **High Load**: 8GB RAM, 4 CPU cores

### Optimization Tips
1. **Data Loading**: Use `@st.cache_data` for expensive computations
2. **Memory Management**: Limit sample data size for better performance
3. **UI Responsiveness**: Use `st.spinner()` for long-running operations
4. **Caching**: Enable browser caching for static assets

### Streamlit Configuration (`config.toml`)
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
runOnSave = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🔒 Security Configuration

### Production Security Checklist
- [ ] Enable HTTPS/TLS encryption
- [ ] Implement authentication (if required)
- [ ] Configure CORS policies
- [ ] Set up rate limiting
- [ ] Enable security headers
- [ ] Regular dependency updates
- [ ] Monitor access logs

### Authentication Integration
```python
# Example: Simple password protection
import streamlit as st

def authenticate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        password = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if password == "demo_password":  # Use secure method in production
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")
        return False
    return True

# Add to main() function
if not authenticate():
    return
```

## 📈 Monitoring and Logging

### Application Monitoring
```python
# Add to demo_app.py
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_app.log'),
        logging.StreamHandler()
    ]
)

# Usage tracking
def track_usage(action, details=None):
    logger.info(f"User action: {action}, Details: {details}")
```

### Health Check Endpoint
```python
# Add to run_demo.py for health checks
def health_check():
    try:
        # Check demo app health
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        return response.status_code == 200
    except:
        return False
```

## 🚦 Troubleshooting

### Common Issues and Solutions

#### Port Already in Use
```bash
# Find process using port
lsof -ti:8501

# Kill process
kill -9 $(lsof -ti:8501)

# Or use different port
python run_demo.py --launch --port 8502
```

#### Memory Issues
```bash
# Monitor memory usage
ps aux | grep streamlit

# Reduce sample data size
# Edit demo_utils.py to generate smaller datasets
```

#### Slow Performance
```bash
# Check system resources
top -p $(pgrep -f streamlit)

# Optimize configuration
# Enable caching in demo_app.py
# Reduce visualization complexity
```

#### Dependency Conflicts
```bash
# Create clean virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r demo_requirements.txt
```

### Log Analysis
```bash
# View application logs
tail -f demo_app.log

# Monitor Streamlit logs
streamlit run demo_app.py --logger.level=debug
```

## 🔄 Continuous Deployment

### GitHub Actions Workflow (`.github/workflows/deploy.yml`)
```yaml
name: Deploy Demo App

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r demo_requirements.txt
    - name: Run tests
      run: |
        python test_demo.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to production
      run: |
        # Add deployment script here
        echo "Deploying to production..."
```

## 📋 Maintenance

### Regular Tasks
1. **Weekly**: Check application logs for errors
2. **Monthly**: Update dependencies and security patches
3. **Quarterly**: Review and update sample data
4. **Annually**: Comprehensive security audit

### Backup Strategy
```bash
# Backup configuration and data
tar -czf backup_$(date +%Y%m%d).tar.gz demo_config.json demo_data/

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "$BACKUP_DIR/fraud_demo_backup_$DATE.tar.gz" \
    demo_config.json demo_data/ demo_app.py demo_utils.py
```

### Update Procedure
```bash
# 1. Backup current version
cp -r fraud-detection-demo fraud-detection-demo.backup

# 2. Pull updates
git pull origin main

# 3. Update dependencies
pip install -r demo_requirements.txt --upgrade

# 4. Run tests
python test_demo.py

# 5. Restart application
# (Implementation depends on deployment method)
```

---

## 📞 Support

For deployment issues or questions:
1. Check the troubleshooting section above
2. Review application logs
3. Verify system requirements
4. Test with minimal configuration
5. Contact the development team

**🛡️ Ready for production deployment with enterprise-grade reliability**