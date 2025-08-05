# Stock Market Prediction using Machine Learning

A comprehensive Python project for predicting stock price movements using machine learning techniques, technical indicators, and historical market data.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for stock price prediction, including:

- **Data Collection**: Automated fetching of historical stock data using Yahoo Finance API
- **Feature Engineering**: Creation of 60+ technical indicators and features
- **Machine Learning**: Multiple ML models for classification and regression tasks
- **Evaluation**: Comprehensive model evaluation with time series validation
- **Visualization**: Interactive analysis through Jupyter notebooks

## 📁 Project Structure

```
Stock_Market/
├── data/                          # Raw and processed data
├── src/                           # Source code
│   ├── data_collector.py         # Data collection from Yahoo Finance
│   ├── feature_engineering.py    # Technical indicators and feature creation
│   └── ml_models.py              # Machine learning models and evaluation
├── models/                        # Saved trained models
├── notebooks/                     # Jupyter notebooks for analysis
│   └── stock_prediction_demo.ipynb
├── results/                       # Output files and results
├── main.py                       # Main script for end-to-end execution
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic Prediction

```bash
# Predict Apple stock direction for next day
python main.py AAPL

# Compare multiple models
python main.py AAPL --compare

# Predict price change instead of direction
python main.py TSLA --target-type price_change --period 1y

# Save the trained model
python main.py GOOGL --save-model
```

### 3. Use Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open and run: notebooks/stock_prediction_demo.ipynb
```

## 🛠️ Features

### Data Collection
- **Real-time data** from Yahoo Finance API
- **Configurable periods**: 1d to max historical data
- **Multiple timeframes**: 1m, 5m, 1h, 1d, 1wk, 1mo
- **Batch processing** for multiple stocks

### Technical Indicators (60+ features)
- **Moving Averages**: SMA, EMA (5, 10, 20, 50 days)
- **Momentum**: RSI, Stochastic Oscillator, MACD
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA
- **Price Features**: High-Low ratios, gaps, rolling statistics

### Machine Learning Models
- **Classification**: Random Forest, Logistic Regression, SVM
- **Regression**: Random Forest, Linear Regression, SVR
- **Time Series Validation**: Proper temporal splitting
- **Feature Importance**: Analysis of most predictive features

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: MSE, RMSE, MAE, R²
- **Cross-validation**: Time series cross-validation
- **Visualization**: Performance plots and analysis

## 📊 Usage Examples

### Command Line Interface

```bash
# Basic usage
python main.py AAPL

# Advanced options
python main.py MSFT --period 5y --target-days 5 --model random_forest --compare --save-model

# Different prediction targets
python main.py TSLA --target-type direction    # Predict up/down
python main.py TSLA --target-type price_change # Predict % change
python main.py TSLA --target-type price       # Predict actual price
```

### Python API

```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineering
from src.ml_models import StockPredictor

# Collect data
collector = StockDataCollector()
data = collector.fetch_stock_data('AAPL', period='2y')

# Create features
fe = FeatureEngineering()
processed_data = fe.prepare_features(data, target_type='direction')

# Train model
predictor = StockPredictor()
X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
predictor.train_model(X_train, y_train, 'random_forest', 'classification')

# Evaluate
metrics = predictor.evaluate_model(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

## 📈 Model Performance

Typical results for major stocks (1-day direction prediction):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 55-58% | 0.56 | 0.58 | 0.57 |
| Logistic Regression | 52-55% | 0.53 | 0.55 | 0.54 |

**Note**: Stock prediction is inherently difficult. Results above 55% accuracy for daily direction prediction are considered good.

## 🔧 Configuration

### Prediction Targets
- `direction`: Binary up/down prediction (classification)
- `price_change`: Percentage change prediction (regression)
- `price`: Absolute price prediction (regression)

### Available Models
- **Classification**: `random_forest`, `logistic_regression`, `svm`
- **Regression**: `random_forest`, `linear_regression`, `svm`

### Time Periods
- Short: `1d`, `5d`, `1mo`, `3mo`, `6mo`
- Medium: `1y`, `2y`, `5y`
- Long: `10y`, `ytd`, `max`

## 📚 Key Dependencies

- **yfinance**: Stock data collection
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models
- **ta**: Technical analysis indicators
- **matplotlib/seaborn**: Visualization
- **jupyter**: Interactive analysis

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This is for educational purposes only
2. **Past Performance**: Historical data doesn't guarantee future results
3. **Market Risk**: Stock markets are inherently unpredictable
4. **Model Limitations**: No model can consistently predict market movements
5. **Transaction Costs**: Real trading involves fees and slippage

## 🚧 Future Enhancements

- [ ] **Deep Learning**: LSTM/GRU networks for sequence modeling
- [ ] **Alternative Data**: News sentiment, social media, economic indicators
- [ ] **Real-time Predictions**: Live trading signals
- [ ] **Portfolio Optimization**: Multi-stock strategies
- [ ] **Risk Management**: Stop-loss, position sizing
- [ ] **Backtesting**: Historical strategy performance
- [ ] **Web Interface**: Interactive dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is for educational purposes. Use at your own risk for any trading decisions.

## 📞 Support

For questions or issues:
1. Check existing issues in the project
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Remember**: Always do your own research and never invest more than you can afford to lose! 📈📉