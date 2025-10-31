# Stock Market Prediction AI

A full-stack stock market prediction application using Machine Learning and Deep Learning, with a modern React frontend and Flask backend.

## 📁 Project Structure

```
Stock_Market/
├── backend/                 # Backend API and ML models
│   ├── src/                # Source code for ML models
│   │   ├── data_collector.py
│   │   ├── feature_engineering.py
│   │   ├── advanced_features.py
│   │   ├── ml_models.py
│   │   ├── deep_learning_models.py
│   │   ├── backtesting.py
│   │   └── visualization.py
│   ├── scripts/            # Utility scripts
│   │   └── analyze_results.py
│   ├── api.py              # Flask REST API
│   ├── app.py              # Streamlit web app
│   ├── main.py             # CLI interface
│   ├── requirements.txt    # Python dependencies
│   └── requirements-api.txt
│
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Threads.jsx        # WebGL background
│   │   │   ├── Particles.jsx      # Particle effects
│   │   │   ├── GridPattern.jsx    # Grid overlay
│   │   │   ├── Marquee.jsx        # Stock ticker
│   │   │   ├── Shimmer.jsx        # Loading states
│   │   │   ├── Dock.jsx           # Navigation dock
│   │   │   └── ...
│   │   ├── services/       # API services
│   │   │   ├── api.js
│   │   │   └── stockTicker.js
│   │   └── App.jsx
│   ├── .env               # Environment variables (Finnhub API key)
│   ├── package.json
│   └── vite.config.js
│
├── scripts/                # Utility scripts
│   ├── run.bat
│   └── test_stocks.bat
│
├── docs/                   # Documentation
│   ├── README.md          # Full project documentation
│   ├── REACT_SETUP.md     # Frontend setup guide
│   └── WEB_APP_GUIDE.md   # Web app guide
│
├── data/                   # Downloaded stock data
├── models/                 # Trained ML models
├── results/                # Prediction results
├── notebooks/              # Jupyter notebooks
└── .gitignore
```

## 🚀 Quick Start

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run the Flask API:**
   ```bash
   python api.py
   ```
   API will run on http://localhost:5000

### Frontend Setup

1. **Install Node dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Add Finnhub API Key:**
   - Get free API key from https://finnhub.io/register
   - Create `frontend/.env` file:
     ```
     VITE_FINNHUB_API_KEY=your_api_key_here
     ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```
   Frontend will run on http://localhost:5173

## 🎨 Features

### Backend
- ✅ Machine Learning models (XGBoost, LightGBM, Random Forest)
- ✅ Deep Learning models (LSTM, GRU, Bidirectional LSTM)
- ✅ Technical indicators and feature engineering
- ✅ Backtesting framework
- ✅ REST API for predictions

### Frontend
- ✅ **Live Stock Ticker** - Real-time market data
- ✅ **Interactive Dashboard** - Stock predictions and analysis
- ✅ **React Bits Components**:
  - Threads (WebGL animated background)
  - Particles (floating particle effects)
  - Grid Pattern (subtle overlay)
  - Marquee (scrolling ticker)
  - Shimmer (loading states)
  - Dock (macOS-style navigation)
- ✅ **Model Comparison** - Compare different ML models
- ✅ **Backtesting** - Test strategy performance

## 📊 Usage

### CLI (Command Line)
```bash
cd backend
python main.py AAPL                    # Basic prediction
python main.py AAPL --period 5y        # 5-year data
python main.py AAPL --compare          # Compare models
```

### API Endpoints
- `GET /api/health` - Health check
- `POST /api/predict` - Get stock predictions
- `POST /api/compare` - Compare models
- `POST /api/backtest` - Run backtesting

### Web Interface
Visit http://localhost:5173 after starting both servers.

## 🛠️ Technologies

**Backend:**
- Python 3.8+
- Flask (REST API)
- scikit-learn (ML models)
- XGBoost, LightGBM
- TensorFlow/Keras (Deep Learning)
- yfinance (Stock data)
- ta (Technical indicators)

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- Framer Motion
- Plotly (Charts)
- Axios (API calls)

## 📝 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This project is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.
