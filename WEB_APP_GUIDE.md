# 📈 Stock Market Prediction Web App - User Guide

## 🚀 Getting Started

### Launch the Web App

```bash
# Make sure you're in the project directory and virtual environment is activated
cd C:\Users\devon\Stock_Market
venv\Scripts\activate

# Launch the app
streamlit run app.py
```

The app will automatically open in your default browser at **http://localhost:8501**

---

## 📱 User Interface Overview

### Sidebar (Left Panel)

The sidebar contains all configuration options:

#### 1. Stock Selection
- **Stock Symbol**: Enter any valid ticker (e.g., AAPL, TSLA, MSFT, GOOGL)
- **Popular Stocks Dropdown**: Quick-select from pre-configured stocks

#### 2. Time Period
Choose how much historical data to use:
- **6mo**: 6 months (faster, less data)
- **1y**: 1 year
- **2y**: 2 years (recommended)
- **5y**: 5 years (slower, more patterns)
- **max**: All available data

#### 3. Model Settings
Choose your prediction model:
- **xgboost**: ⭐ Best accuracy (55-60%)
- **lightgbm**: Fast and accurate (54-58%)
- **random_forest**: Good baseline (50-55%)
- **ensemble**: Combines multiple models (56-59%)
- **logistic_regression**: Simple baseline (48-52%)

#### 4. Prediction Settings
- **Prediction Type**:
  - `direction`: UP/DOWN (classification)
  - `price_change`: Percentage change (regression)
  - `price`: Actual future price (regression)

- **Days Ahead**: Predict 1-10 days in the future

#### 5. Advanced Options
- ☑️ **Use Advanced Features**: Adds candlestick patterns & market regimes
- ☑️ **Compare All Models**: Tests all 5 models automatically
- ☑️ **Run Backtesting**: Simulates trading strategy

### Main Content (Tabs)

The app has **5 main tabs**:

---

## 📊 Tab 1: Predictions

### What You'll See

1. **Current Price Card**
   - Latest closing price
   - Percentage change from previous day

2. **Prediction Card**
   - Tomorrow's predicted direction (📈 UP or 📉 DOWN)
   - Or predicted price/percentage change

3. **Historical Data Card**
   - Percentage of historical up days
   - Data quality indicator

### Recent Predictions Table

Shows last 20 trading days with:
- **Date**: Trading date
- **Close**: Closing price
- **Actual**: What actually happened
- **Predicted**: What the model predicted
- **Correct**: ✅ or ❌ indicator
- **Recent Accuracy**: Overall accuracy on recent data

### Model Performance Metrics

#### For Classification (Direction):
- **🎯 Accuracy**: Overall correctness (target: 55-60%)
- **🎪 Precision**: When model says UP, how often is it right?
- **📡 Recall**: Of all actual UP days, how many did we catch?
- **⚖️ F1-Score**: Balance of precision and recall

#### For Regression (Price/Change):
- **📉 MSE**: Mean Squared Error (lower is better)
- **📊 RMSE**: Root MSE (typical error magnitude)
- **📏 MAE**: Mean Absolute Error (average error)
- **📈 R²**: Variance explained (higher is better, max 1.0)

---

## 📈 Tab 2: Visualizations

### Interactive Candlestick Chart

**Features:**
- **Green candles**: Price went up that day
- **Red candles**: Price went down that day
- **Moving averages**: SMA 20 (orange), SMA 50 (blue)
- **Buy signals**: Green triangles pointing up
- **Sell signals**: Red triangles pointing down

**Controls:**
- 🖱️ **Zoom**: Click and drag to zoom in
- 📏 **Pan**: Hold shift and drag to pan
- 🔍 **Hover**: See exact values
- 🏠 **Reset**: Double-click to reset view

### Volume Chart

Shows trading volume for each day:
- **Green bars**: Price closed higher than open
- **Red bars**: Price closed lower than open

### RSI Indicator

Relative Strength Index (0-100):
- **Above 70**: Overbought (consider selling)
- **Below 30**: Oversold (consider buying)
- **Between 30-70**: Neutral zone

### Feature Importance Chart

Horizontal bar chart showing top 20 features:
- **Longer bars**: More important for predictions
- **Color gradient**: Importance level
- Common important features:
  - SMA (moving averages)
  - ATR (volatility)
  - RSI (momentum)
  - Volume indicators

---

## 🔍 Tab 3: Model Comparison

Only visible when "Compare All Models" is enabled.

### Performance Table

Compares all 5 models side-by-side:
- Rows: Different models
- Columns: Performance metrics
- **Green highlighted**: Best value for each metric

### Visual Comparison Chart

Bar chart showing model performance:
- **Taller bars**: Better performance
- **Color scale**: Darker = better
- Quick visual identification of best model

### Best Model Indicator

Shows which model performed best and its score:
```
🏆 Best Model: xgboost (Accuracy: 56.0%)
```

---

## 💰 Tab 4: Backtesting

Simulates actual trading with your predictions.

### Configuration

**Initial Capital**: Starting money (default: $10,000)
**Stop Loss**: Exit position if it drops by X% (default: 5%)
**Take Profit**: Exit position if it gains X% (default: 10%)

### Performance Summary

Shows 4 key metrics:

1. **💵 Final Value**
   - How much money you'd have at the end
   - Green/red percentage showing profit/loss

2. **🔄 Total Trades**
   - Number of buy/sell cycles executed

3. **🎯 Win Rate**
   - Percentage of profitable trades
   - Target: 50-60%

4. **📊 Sharpe Ratio**
   - Risk-adjusted returns
   - > 1.0 is good, > 2.0 is excellent

### Portfolio Value Chart

**Top Panel**:
- Blue line: Your portfolio value over time
- Gray dashed line: Initial capital
- Rising line: Making money ✅
- Falling line: Losing money ❌

**Bottom Panel**:
- Red filled area: Drawdown (how far below peak)
- Deeper red: Bigger losses
- At zero: At new peak value

### Trade History Table

Detailed log of every trade:
- **Entry Date**: When you bought
- **Exit Date**: When you sold
- **Entry/Exit Price**: Buy and sell prices
- **Shares**: Number of shares traded
- **Profit**: Dollar amount gained/lost
- **Return**: Percentage return
- **Exit Reason**: Why you sold (signal, stop_loss, take_profit)

---

## 📚 Tab 5: About

Information about the app, features, and disclaimers.

---

## 🎯 Step-by-Step Usage Example

### Example 1: Basic Stock Prediction

1. **Launch the app**: `streamlit run app.py`
2. **Enter stock symbol**: Type "AAPL" in sidebar
3. **Select model**: Choose "xgboost"
4. **Click "🚀 Run Analysis"**
5. **Wait 30-60 seconds** for processing
6. **View results** in Predictions tab

**Expected Output:**
```
✅ Collected 502 trading days
✅ Created 69 features from 453 samples
✅ Model trained successfully!
🎉 Analysis Complete!

Current Price: $262.82
Prediction (Next Day): 📈 UP
Model Accuracy: 56.0%
```

### Example 2: Compare All Models

1. **Launch the app**
2. **Enter symbol**: "TSLA"
3. **Check "Compare All Models"** in sidebar
4. **Click "🚀 Run Analysis"**
5. **Wait 2-3 minutes** (trains 5 models)
6. **Go to "Model Comparison" tab**
7. **See which model performs best** for TSLA

**Expected Output:**
```
Model Comparison Results:
xgboost         : 58.2% ⭐
lightgbm        : 57.1%
ensemble        : 56.5%
random_forest   : 54.3%
logistic_regression: 51.2%

Best Model: xgboost
```

### Example 3: Backtest a Strategy

1. **Launch the app**
2. **Enter symbol**: "MSFT"
3. **Select model**: "xgboost"
4. **Check "Run Backtesting"**
5. **Set parameters**:
   - Initial Capital: $10,000
   - Stop Loss: 5%
   - Take Profit: 10%
6. **Click "🚀 Run Analysis"**
7. **Go to "Backtesting" tab**
8. **Review performance**

**Expected Output:**
```
Final Value: $12,450
Total Return: 24.5%
Total Trades: 45
Win Rate: 60.0%
Sharpe Ratio: 1.25
Max Drawdown: -18.5%
```

---

## 💡 Tips & Best Practices

### For Best Accuracy
- ✅ Use **2-5 years** of data
- ✅ Choose **XGBoost** or **LightGBM**
- ✅ Enable **Compare All Models** to find best for each stock
- ✅ Check **Recent Accuracy** - should match test accuracy
- ✅ Enable **Advanced Features** for more patterns

### For Faster Results
- ⚡ Use **6mo** or **1y** period
- ⚡ Choose **LightGBM** (faster than XGBoost)
- ⚡ Don't enable "Compare All Models"
- ⚡ Don't enable "Advanced Features"

### For Trading Strategy
- 📈 Always run **backtesting** first
- 📈 Set appropriate **stop-losses** (3-7%)
- 📈 Take profits at **reasonable targets** (8-15%)
- 📈 Look for **Sharpe ratio > 1.0**
- 📈 Check **max drawdown** < 25%

### Understanding Results

#### Good Performance Indicators
- ✅ Accuracy > 55% (classification)
- ✅ R² > 0.3 (regression)
- ✅ Win rate > 50% (backtesting)
- ✅ Sharpe ratio > 1.0
- ✅ Recent accuracy close to test accuracy

#### Warning Signs
- ⚠️ Recent accuracy << Test accuracy (model degrading)
- ⚠️ Win rate < 45% (losing strategy)
- ⚠️ Max drawdown > 30% (too risky)
- ⚠️ Sharpe ratio < 0.5 (poor risk/reward)

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# Make sure Streamlit is installed
pip install streamlit

# Try running with full path
venv\Scripts\streamlit.exe run app.py
```

### Stock Symbol Not Found
- Check spelling (AAPL not Apple)
- Use uppercase (AAPL not aapl)
- Try popular stocks first (AAPL, MSFT, GOOGL)

### Analysis Taking Too Long
- Reduce time period (use 6mo instead of 5y)
- Don't enable "Compare All Models"
- Use faster model (LightGBM instead of ensemble)

### Low Accuracy
- Some stocks are harder to predict
- Try different time periods
- Enable "Compare All Models" to find best
- Use more data (2y or 5y)

---

## ⚠️ Important Reminders

### This Tool is for EDUCATION ONLY
- ❌ Not financial advice
- ❌ No guarantee of profits
- ❌ Past performance ≠ future results
- ❌ Markets are unpredictable

### Before Real Trading
1. **Paper trade** for several months
2. **Verify** results match backtesting
3. **Start small** with money you can afford to lose
4. **Use stop-losses** on every trade
5. **Diversify** - don't bet everything on one stock

### Realistic Expectations
- **55-60% accuracy** is very good for stock prediction
- **50% accuracy** = random guessing (coin flip)
- **Most professional traders** only win 55-60% of trades
- **Backtesting** returns may not match live trading

---

## 📞 Support

### Common Questions

**Q: Why is accuracy only 56%?**
A: Stock markets are extremely difficult to predict. 56% is actually very good! It means you're right more often than wrong, which compounds over time.

**Q: Can I use this for crypto or forex?**
A: The app is designed for stocks, but you can try crypto tickers if they're available on Yahoo Finance (e.g., BTC-USD).

**Q: How often should I retrain?**
A: Retrain monthly or when you notice accuracy dropping. Markets change!

**Q: What's the best model?**
A: Usually XGBoost or LightGBM, but use "Compare All Models" to find the best for each specific stock.

---

## 🎓 Learning Resources

### Understand the Metrics
- **Accuracy**: (Correct predictions) / (Total predictions)
- **Sharpe Ratio**: (Return - Risk Free Rate) / (Volatility)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: (Winning trades) / (Total trades)

### Understand the Features
- **SMA**: Simple Moving Average (trend)
- **RSI**: Relative Strength Index (momentum)
- **MACD**: Moving Average Convergence Divergence
- **ATR**: Average True Range (volatility)
- **Bollinger Bands**: Volatility bands around price

---

**Happy Trading! Remember to always do your own research and trade responsibly! 📈💰**
