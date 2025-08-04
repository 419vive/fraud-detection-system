# Performance Benchmarking System

A comprehensive performance benchmarking framework for the IEEE-CIS Fraud Detection Machine Learning models, featuring automated testing, regression detection, and detailed visualization dashboards.

## 🎯 Overview

This benchmarking system provides:

- **Comprehensive Performance Analysis**: Training speed, inference latency, memory usage, and accuracy benchmarks
- **Model Comparison Framework**: Side-by-side comparison of LightGBM, XGBoost, CatBoost, and Random Forest
- **Automated Regression Detection**: Identifies performance degradations with configurable thresholds
- **Advanced Visualizations**: Interactive dashboards and detailed performance reports
- **CI/CD Integration**: Automated benchmarks on code changes with GitHub Actions
- **Real-time Performance Monitoring**: Production-ready performance tracking

## 📊 Key Features

### 1. Model Performance Benchmarks
- **Training Speed**: Time to train, memory usage, CPU utilization
- **Inference Latency**: P50, P95, P99 latencies across different batch sizes
- **Accuracy Metrics**: ROC-AUC, Precision, Recall, F1-Score with cross-validation
- **Scalability Testing**: Performance across different dataset sizes
- **Resource Utilization**: CPU, memory, and GPU usage monitoring

### 2. System Performance Analysis
- **Feature Engineering Pipeline**: Performance of data preprocessing steps
- **End-to-End Throughput**: Complete pipeline performance measurement
- **Batch vs Real-time**: Performance comparison between processing modes
- **Memory Efficiency**: Memory usage optimization and leak detection
- **Optimization Impact**: Before/after performance comparisons

### 3. Automated Quality Assurance
- **Performance Regression Detection**: Automatic detection of performance degradations
- **Threshold Monitoring**: Alerts when performance falls below acceptable levels
- **Historical Tracking**: Long-term performance trend analysis
- **Baseline Management**: Automatic baseline establishment and updates

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd project-2

# Install dependencies
pip install -r requirements.txt
pip install schedule pyyaml requests  # Additional for automation
```

### 2. Create Sample Data

```bash
# Create a sample fraud detection dataset
python run_benchmark.py create-data --data sample_data.csv --size 10000
```

### 3. Run Quick Benchmark

```bash
# Run a quick benchmark with LightGBM and XGBoost
python run_benchmark.py quick --data sample_data.csv
```

### 4. View Results

Results will be saved to `quick_benchmark_results/` with:
- JSON results file with all metrics
- CSV summaries for easy analysis
- Performance visualizations (if generated)

## 📈 Benchmark Modes

### Quick Benchmark
Fast benchmark with essential models (LightGBM, XGBoost):
```bash
python run_benchmark.py quick --data your_data.csv
```

### Full Benchmark
Comprehensive benchmark with all models and optimizations:
```bash
python run_benchmark.py full --data your_data.csv
```

### Automated Benchmark
Configuration-driven benchmark with regression detection:
```bash
python run_benchmark.py auto --config benchmark_config.yaml --data your_data.csv
```

## 🔧 Configuration

Edit `benchmark_config.yaml` to customize:

```yaml
# Dataset Configuration
dataset_path: "data/sample_fraud_data.csv"
target_column: "isFraud"
models_to_test: ["lightgbm", "xgboost", "catboost", "random_forest"]

# Performance Thresholds
max_training_time_minutes: 30.0
min_roc_auc: 0.85
max_inference_latency_ms: 100.0
min_throughput_per_second: 100

# Regression Detection
enable_regression_detection: true
regression_threshold_percent: 5.0

# Alerts
enable_alerts: true
alert_webhook_url: "your_slack_webhook"
```

## 📊 Visualization Dashboard

Generate interactive performance dashboards:

```bash
# Generate visualizations from results
python run_benchmark.py visualize --results benchmark_results/benchmark_results_*.json
```

The system creates multiple dashboards:
- **Training Performance**: Training time, memory usage, convergence analysis
- **Inference Performance**: Latency distributions, throughput analysis, batch optimization
- **Accuracy Comparison**: ROC curves, precision-recall analysis, cross-validation results
- **Comprehensive Matrix**: Side-by-side model comparison with normalized scores
- **Executive Summary**: High-level performance overview with recommendations

## 🤖 CI/CD Integration

### GitHub Actions

The included workflow (`.github/workflows/performance_benchmark.yml`) provides:

- **Automatic Benchmarks**: Run on every push and pull request
- **Scheduled Full Benchmarks**: Daily comprehensive analysis
- **Regression Detection**: Fail builds on performance degradations
- **PR Comments**: Automatic performance reports on pull requests
- **Artifact Upload**: Store results for historical analysis

### Slack Integration

Add your Slack webhook URL to GitHub secrets as `SLACK_WEBHOOK_URL` for automatic notifications.

## 📋 Performance Metrics & Targets

### Training Performance
- **Target Training Time**: < 30 minutes per model
- **Memory Usage**: < 8GB peak usage
- **CPU Efficiency**: > 70% utilization during training

### Inference Performance
- **Latency Targets**:
  - P50: < 10ms
  - P95: < 50ms  
  - P99: < 100ms
- **Throughput**: > 100 predictions/second
- **Batch Optimization**: Find optimal batch size for throughput

### Accuracy Targets
- **ROC-AUC**: > 0.85 (target: > 0.90)
- **Precision**: > 0.80 
- **Recall**: > 0.75
- **F1-Score**: > 0.77

## 🔍 Regression Detection

The system automatically detects:

- **Training Regressions**: Increased training time or memory usage
- **Accuracy Regressions**: Decreased model performance
- **Inference Regressions**: Increased latency or decreased throughput
- **Resource Regressions**: Higher CPU/memory usage

### Configurable Thresholds
- Default: 5% performance degradation triggers alert
- Severity levels: Low, Medium, High, Critical
- Historical baseline automatically maintained

## 📁 Output Structure

```
benchmark_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json  # Complete results
├── training_benchmarks_YYYYMMDD_HHMMSS.csv  # Training metrics
├── accuracy_benchmarks_YYYYMMDD_HHMMSS.csv  # Accuracy metrics  
├── inference_benchmarks_YYYYMMDD_HHMMSS.csv # Inference metrics
├── benchmark_summary_YYYYMMDD_HHMMSS.txt    # Human-readable summary
├── ci_cd_summary.txt                        # CI/CD specific summary
└── historical_results.json                  # Historical data for regression detection

benchmark_visualizations/
├── index.html                              # Dashboard index
├── training_performance_dashboard.html     # Training analysis
├── inference_performance_dashboard.html    # Inference analysis
├── accuracy_comparison_dashboard.html      # Model accuracy comparison
├── comprehensive_comparison_matrix.html    # Model comparison matrix
├── performance_trend_analysis.html         # Trend analysis
├── optimization_impact_analysis.html       # Optimization effectiveness
└── executive_summary_dashboard.html        # Executive overview
```

## 🎛️ Advanced Usage

### Custom Model Integration

Add custom models to the benchmark:

```python
from src.performance_benchmarking import PerformanceBenchmarkSuite

# Define custom models
custom_models = {
    'my_model': MyCustomModel(),
    'neural_net': MyNeuralNetwork()
}

# Run benchmark
suite = PerformanceBenchmarkSuite()
results = suite.run_comprehensive_benchmark(
    custom_models, X_train, y_train, X_test, y_test
)
```

### Programmatic Usage

```python
from src.performance_benchmarking import run_comprehensive_fraud_detection_benchmark
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Run benchmark
results = run_comprehensive_fraud_detection_benchmark(
    df=df,
    target_col='isFraud',
    models_to_test=['lightgbm', 'xgboost'],
    output_dir='my_results'
)

# Access results
best_model = results['comparative_analysis']['overall_ranking'][0]
print(f"Best model: {best_model['model_name']} (Score: {best_model['overall_score']:.3f})")
```

### Performance Optimization Testing

Test different optimization levels:

```python
from src.performance_optimizer import InferenceOptimizer

optimizer = InferenceOptimizer()

# Test optimization levels
for level in ['light', 'medium', 'aggressive']:
    optimized_model = optimizer.optimize_model_for_inference(model, 'lightgbm', level)
    # Benchmark optimized model
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce dataset size for testing
   - Enable memory optimization in config
   - Use batch processing for large datasets

2. **Training Timeouts**
   - Increase `max_training_time_minutes` in config
   - Use quick benchmark mode for development
   - Enable early stopping in model configs

3. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install schedule pyyaml requests memory-profiler
   ```

4. **Visualization Issues**
   - Ensure plotly and pandas are installed
   - Check that results JSON file exists
   - Verify sufficient disk space for outputs

### Performance Tips

1. **For Development**: Use quick benchmark mode with small datasets
2. **For CI/CD**: Enable regression detection with appropriate thresholds
3. **For Production**: Use scheduled full benchmarks with larger datasets
4. **For Analysis**: Generate all visualizations for comprehensive insights

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run benchmarks to ensure no regressions (`python run_benchmark.py quick --data sample_data.csv`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the configuration documentation

---

**Built with ❤️ for high-performance fraud detection systems**