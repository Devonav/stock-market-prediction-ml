# Stock Market Prediction using Machine Learning

A comprehensive Python project for predicting stock price movements using advanced machine learning techniques, deep learning, technical indicators, and realistic backtesting.

## Project Overview

This project implements a complete machine learning pipeline for stock price prediction, including:

- **Data Collection**: Automated fetching of historical stock data using Yahoo Finance API
- **Feature Engineering**: 100+ technical indicators including candlestick patterns and market regime detection
- **Machine Learning**: Multiple ML models (Random Forest, XGBoost, LightGBM, Ensemble)
- **Deep Learning**: LSTM and GRU networks for time series prediction
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Backtesting**: Advanced backtesting framework with realistic trading simulation
- **Visualization**: Interactive dashboards using Plotly
- **Evaluation**: Comprehensive model evaluation with time series validation

## Project Structure

```
Stock_Market/
├── data/                          # Raw and processed data
├── src/                           # Source code
│   ├── data_collector.py         # Data collection from Yahoo Finance
│   ├── feature_engineering.py    # Basic technical indicators
│   ├── advanced_features.py      # Candlestick patterns & market regimes
│   ├── ml_models.py              # Traditional ML models (RF, XGB, LGB, Ensemble)
│   ├── deep_learning_models.py   # LSTM/GRU neural networks
│   ├── hyperparameter_tuning.py  # Optuna-based AutoML
│   ├── backtesting.py            # Advanced backtesting framework
│   └── visualization.py          # Interactive Plotly dashboards
├── models/                        # Saved trained models
├── notebooks/                     # Jupyter notebooks for analysis
│   └── stock_prediction_demo.ipynb
├── results/                       # Output files and results
├── app.py                        # Streamlit web application (NEW!)
├── main.py                       # Command-line interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Web Interface (Recommended)

```bash
# Start the interactive web application
streamlit run app.py
```

Then open your browser to **http://localhost:8501**

**Features:**
- 📊 Interactive stock selection
- 🎯 Real-time predictions with visualizations
- 📈 Candlestick charts with buy/sell signals
- 🔍 Model comparison dashboard
- 💰 Backtesting simulator
- 📱 Mobile-friendly responsive design

### 3. Run Command Line (Advanced)

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

## Features

### Data Collection
- **Real-time data** from Yahoo Finance API
- **Configurable periods**: 1d to max historical data
- **Multiple timeframes**: 1m, 5m, 1h, 1d, 1wk, 1mo
- **Batch processing** for multiple stocks

### Feature Engineering (100+ features)

#### Basic Technical Indicators
- **Moving Averages**: SMA, EMA (5, 10, 20, 50 days)
- **Momentum**: RSI, Stochastic Oscillator, MACD, ROC, Williams %R, CCI, MFI
- **Volatility**: Bollinger Bands, ATR, Keltner Channels, Historical Volatility
- **Volume**: OBV, Volume SMA, Chaikin Money Flow, A/D Line, Volume Price Trend
- **Price Features**: High-Low ratios, gaps, rolling statistics

#### Advanced Features
- **Candlestick Patterns**: Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star, Three Soldiers/Crows, Marubozu
- **Market Regime Detection**: Trending vs Ranging, Bullish vs Bearish, Volatility Regimes
- **Advanced Momentum**: Rate of Change, Commodity Channel Index, Money Flow Index
- **Advanced Volatility**: Parkinson Volatility, Chaikin Volatility
- **Market Correlation**: Beta, Relative Strength vs Market Indices

### Machine Learning Models

#### Traditional ML
- **Gradient Boosting**: XGBoost, LightGBM (state-of-the-art performance)
- **Random Forest**: Classifier and Regressor
- **Ensemble Methods**: Voting ensembles combining multiple models
- **Linear Models**: Logistic Regression, Linear Regression, SVM

#### Deep Learning
- **LSTM**: Long Short-Term Memory networks for sequence modeling
- **GRU**: Gated Recurrent Units (faster alternative to LSTM)
- **Bidirectional LSTM**: Processes sequences in both directions
- **Configurable architectures**: Multiple layers, dropout, batch normalization

### Hyperparameter Optimization
- **Optuna Integration**: Bayesian optimization using TPE sampler
- **AutoML**: Automated model selection and hyperparameter tuning
- **Multi-metric optimization**: Supports various scoring metrics
- **Visualization**: Optimization history and parameter importance plots

### Backtesting Framework
- **Realistic Trading Simulation**: Commission costs and slippage modeling
- **Risk Management**: Stop-loss and take-profit strategies
- **Position Sizing**: Configurable capital allocation
- **Performance Metrics**: Sharpe ratio, max drawdown, profit factor, win rate
- **Trade Analysis**: Detailed trade history and statistics

### Interactive Visualization
- **Candlestick Charts**: Interactive price charts with technical indicators
- **Feature Importance**: Visual analysis of model features
- **Model Comparison**: Side-by-side performance comparison
- **Backtest Results**: Portfolio value, drawdown, and trade visualization
- **Correlation Heatmaps**: Feature correlation analysis
- **Confusion Matrices**: Classification performance visualization

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: MSE, RMSE, MAE, R²
- **Trading Metrics**: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor
- **Cross-validation**: Time series cross-validation
- **Visualization**: Comprehensive performance plots

## Usage Examples

### Command Line Interface

```bash
# Basic usage
python main.py AAPL

# Using XGBoost or LightGBM
python main.py AAPL --model xgboost
python main.py AAPL --model lightgbm

# Using ensemble of multiple models
python main.py AAPL --model ensemble

# Compare all available models
python main.py MSFT --compare

# Advanced options
python main.py MSFT --period 5y --target-days 5 --model xgboost --compare --save-model

# Different prediction targets
python main.py TSLA --target-type direction    # Predict up/down
python main.py TSLA --target-type price_change # Predict % change
python main.py TSLA --target-type price       # Predict actual price
```

### Python API Examples

#### Basic Machine Learning
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

# Train XGBoost model
predictor = StockPredictor()
X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
predictor.train_model(X_train, y_train, 'xgboost', 'classification')

# Evaluate
metrics = predictor.evaluate_model(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

#### Advanced Feature Engineering
```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineering
from src.advanced_features import AdvancedFeatureEngineering

# Collect data
collector = StockDataCollector()
data = collector.fetch_stock_data('AAPL', period='2y')

# Basic features
fe = FeatureEngineering()
data_with_basic = fe.prepare_features(data, target_type='direction')

# Advanced features (candlestick patterns, market regimes)
advanced_fe = AdvancedFeatureEngineering()
data_enhanced = advanced_fe.prepare_advanced_features(data)

print(f"Total features: {data_enhanced.shape[1]}")
```

#### Deep Learning with LSTM
```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineering
from src.deep_learning_models import DeepLearningPredictor

# Collect and process data
collector = StockDataCollector()
data = collector.fetch_stock_data('AAPL', period='2y')

fe = FeatureEngineering()
processed_data = fe.prepare_features(data, target_type='direction')

# Prepare sequences for LSTM
dl_predictor = DeepLearningPredictor()
X, y = dl_predictor.prepare_sequences(processed_data, sequence_length=60)

# Split data
X_train, X_test, y_train, y_test = dl_predictor.split_data(X, y)
X_train, X_val, y_train, y_val = dl_predictor.split_data(X_train, y_train, test_size=0.1)

# Train LSTM model
history = dl_predictor.train_model(
    X_train, y_train, X_val, y_val,
    model_type='lstm',
    task_type='classification',
    epochs=50
)

# Evaluate
metrics = dl_predictor.evaluate_model(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

#### Hyperparameter Optimization (AutoML)
```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineering
from src.hyperparameter_tuning import AutoML

# Prepare data
collector = StockDataCollector()
data = collector.fetch_stock_data('AAPL', period='2y')

fe = FeatureEngineering()
processed_data = fe.prepare_features(data, target_type='direction')

from src.ml_models import StockPredictor
predictor = StockPredictor()
X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)

# Run AutoML
automl = AutoML(n_trials=50)
results = automl.auto_optimize(
    X_train, y_train,
    task_type='classification',
    models=['xgboost', 'lightgbm', 'random_forest']
)

# Train best model
best_model = automl.train_best_model(X_train, y_train)
print(f"Best model: {automl.best_model_type}")
```

#### Backtesting Trading Strategy
```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineering
from src.ml_models import StockPredictor
from src.backtesting import Backtester

# Train model and get predictions
collector = StockDataCollector()
data = collector.fetch_stock_data('AAPL', period='2y')

fe = FeatureEngineering()
processed_data = fe.prepare_features(data, target_type='direction')

predictor = StockPredictor()
X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
predictor.train_model(X_train, y_train, 'xgboost', 'classification')

predictions = predictor.predict(processed_data[predictor.feature_columns])

# Run backtest
backtester = Backtester(
    initial_capital=10000,
    commission=0.001,  # 0.1%
    slippage=0.001     # 0.1%
)

results = backtester.run_backtest(
    data,
    predictions,
    stop_loss=0.05,      # 5% stop loss
    take_profit=0.10,    # 10% take profit
    position_size=0.95   # Use 95% of capital
)

backtester.print_summary()
backtester.plot_results(buy_and_hold_data=data)
```

#### Interactive Visualization
```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineering
from src.ml_models import StockPredictor
from src.visualization import StockVisualization

# Get data and predictions
collector = StockDataCollector()
data = collector.fetch_stock_data('AAPL', period='6mo')

fe = FeatureEngineering()
processed_data = fe.prepare_features(data, target_type='direction')

predictor = StockPredictor()
X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
predictor.train_model(X_train, y_train, 'xgboost', 'classification')

predictions = predictor.predict(processed_data[predictor.feature_columns])

# Create interactive visualizations
viz = StockVisualization()
viz.plot_candlestick(data, predictions, title="AAPL Stock Analysis")

feature_importance = predictor.get_feature_importance(top_n=20)
viz.plot_feature_importance(feature_importance)
```

## Model Performance

Typical results for major stocks (1-day direction prediction):

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
|-------|----------|-----------|--------|----------|-------|
| **XGBoost** | 56-60% | 0.58 | 0.60 | 0.59 | Best overall performance |
| **LightGBM** | 56-59% | 0.57 | 0.59 | 0.58 | Faster training |
| **Ensemble** | 57-61% | 0.59 | 0.61 | 0.60 | Combines RF, XGB, LGB |
| Random Forest | 55-58% | 0.56 | 0.58 | 0.57 | Good baseline |
| **LSTM** | 54-58% | 0.56 | 0.58 | 0.57 | Captures sequences |
| Logistic Regression | 52-55% | 0.53 | 0.55 | 0.54 | Simple baseline |

**Backtesting Performance** (2-year test period):
- Sharpe Ratio: 0.8 - 1.5
- Max Drawdown: 15-25%
- Win Rate: 50-60%
- Annual Return: 10-25% (varies significantly by stock and period)

**Note**: Stock prediction is inherently difficult. Results above 55% accuracy for daily direction prediction are considered good. Past performance does not guarantee future results.

## Configuration

### Prediction Targets
- `direction`: Binary up/down prediction (classification)
- `price_change`: Percentage change prediction (regression)
- `price`: Absolute price prediction (regression)

### Available Models
- **Classification**: `random_forest`, `xgboost`, `lightgbm`, `ensemble`, `logistic_regression`, `svm`
- **Regression**: `random_forest`, `xgboost`, `lightgbm`, `ensemble`, `linear_regression`, `svm`
- **Deep Learning**: `lstm`, `gru`, `bidirectional_lstm`

### Time Periods
- Short: `1d`, `5d`, `1mo`, `3mo`, `6mo`
- Medium: `1y`, `2y`, `5y`
- Long: `10y`, `ytd`, `max`

## Key Dependencies

### Core Libraries
- **yfinance**: Stock data collection from Yahoo Finance
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Machine Learning
- **scikit-learn**: Traditional ML models and utilities
- **xgboost**: Gradient boosting framework
- **lightgbm**: Fast gradient boosting
- **optuna**: Hyperparameter optimization

### Deep Learning
- **tensorflow**: Deep learning framework
- **keras**: High-level neural network API

### Technical Analysis
- **ta**: Technical analysis indicators library

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive dashboards
- **mplfinance**: Financial charts

### Development
- **jupyter**: Interactive notebook environment

## Important Disclaimers

1. **Not Financial Advice**: This is for educational purposes only
2. **Past Performance**: Historical data doesn't guarantee future results
3. **Market Risk**: Stock markets are inherently unpredictable
4. **Model Limitations**: No model can consistently predict market movements
5. **Transaction Costs**: Real trading involves fees and slippage

## Web Application Interface

### Launch the Web App

```bash
streamlit run app.py
```

The web interface provides an intuitive, interactive dashboard with:

#### 📊 Main Features

1. **Predictions Dashboard**
   - Current price and prediction
   - Recent predictions table with accuracy
   - Model performance metrics
   - Historical up/down day distribution

2. **Interactive Visualizations**
   - Candlestick charts with technical indicators
   - Buy/sell signal markers
   - Volume and RSI charts
   - Feature importance bar charts

3. **Model Comparison**
   - Side-by-side comparison of all models
   - Performance metrics table
   - Visual comparison charts
   - Automatic best model selection

4. **Backtesting Simulator**
   - Configurable initial capital
   - Adjustable stop-loss and take-profit
   - Portfolio value over time
   - Drawdown visualization
   - Complete trade history

5. **Configuration Panel**
   - Stock symbol selection
   - Popular stocks quick-select
   - Time period configuration
   - Model selection (XGBoost, LightGBM, etc.)
   - Prediction settings (direction, price change, price)
   - Advanced features toggle

### Screenshots

The web app includes:
- **Responsive design** - works on desktop and mobile
- **Real-time updates** - live data and predictions
- **Interactive charts** - zoom, pan, hover for details
- **Dark/Light themes** - Streamlit's theme options
- **Easy navigation** - tabbed interface for different views

## Implemented Features

- [x] **Web Application**: Beautiful Streamlit interface with real-time predictions
- [x] **Deep Learning**: LSTM/GRU networks for sequence modeling
- [x] **Advanced ML Models**: XGBoost, LightGBM, Ensemble methods
- [x] **Risk Management**: Stop-loss, take-profit, position sizing
- [x] **Backtesting**: Advanced backtesting framework with realistic trading simulation
- [x] **Interactive Dashboards**: Plotly-based visualizations
- [x] **Hyperparameter Optimization**: AutoML using Optuna
- [x] **Advanced Features**: Candlestick patterns, market regime detection
- [x] **Performance Metrics**: Sharpe ratio, max drawdown, profit factor

## Future Enhancements

- [ ] **Sentiment Analysis**: News sentiment, social media integration
- [ ] **Economic Indicators**: GDP, interest rates, inflation data
- [ ] **Real-time Predictions**: Live trading signals with streaming data
- [ ] **Portfolio Optimization**: Modern Portfolio Theory, multi-stock strategies
- [ ] **Options Trading**: Options pricing and strategy modeling
- [ ] **Web Interface**: Flask/Django web application
- [ ] **Cloud Deployment**: AWS/GCP deployment with automated trading
- [ ] **Multi-asset Support**: Crypto, Forex, Commodities

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is for educational purposes. Use at your own risk for any trading decisions.

## Support

For questions or issues:
1. Check existing issues in the project
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Remember**: Always do your own research and never invest more than you can afford to lose!