# Stock Market Prediction AI

A full-stack stock market prediction application using Machine Learning and Deep Learning, with a modern React frontend and Flask backend.

## Project Structure

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
│   │   ├── sentiment_analyzer.py # Sentiment analysis logic
│   │   ├── portfolio_manager.py  # Portfolio management logic
│   │   ├── websocket_server.py   # Real-time data server
│   │   └── visualization.py
│   ├── scripts/            # Utility scripts
│   │   └── analyze_results.py
│   ├── api.py              # Flask REST API
│   ├── app.py              # Streamlit web app
│   ├── main.py             # CLI interface
│   ├── requirements.txt    # Python dependencies
│   └── portfolio.json      # Portfolio persistence
│
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── StockPrediction.jsx
│   │   │   ├── PortfolioDashboard.jsx
│   │   │   ├── SentimentDashboard.jsx
│   │   │   ├── Threads.jsx        # WebGL background
│   │   │   ├── Particles.jsx      # Particle effects
│   │   │   ├── Marquee.jsx        # Stock ticker
│   │   │   ├── Dock.jsx           # Navigation dock
│   │   │   └── ...
│   │   ├── services/       # API services
│   │   │   ├── api.js
│   │   │   └── stockTicker.js
│   │   └── App.jsx
│   ├── .env               # Environment variables
│   ├── package.json
│   └── vite.config.js
│
├── docs/                   # Documentation
│   ├── README.md          # Full project documentation
│   ├── REACT_SETUP.md     # Frontend setup guide
│   └── WEB_APP_GUIDE.md   # Web app guide
│
├── data/                   # Downloaded stock data
├── models/                 # Trained ML models
└── results/                # Prediction results
```

## Quick Start

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
   API will run on http://localhost:5000 (WebSocket enabled)

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

## Features

### Backend
- **Advanced ML Models**: XGBoost, LightGBM, Random Forest, **Ensemble**, **LSTM** (Deep Learning).
- **Real-Time Data**: WebSocket server for live price updates.
- **Sentiment Analysis**: Analyzes news and social media sentiment (simulated).
- **Portfolio Management**: Buy/sell stocks, track holdings, and calculate P&L.
- **Technical Indicators**: RSI, MACD, SMA, EMA, Bollinger Bands.
- **Backtesting**: Validate strategies against historical data.

### Frontend
- **Live Stock Ticker**: Real-time market data via WebSockets.
- **Interactive Dashboard**: Glassmorphism UI with animated components.
- **Stock Prediction**: Visualize predicted trends and confidence scores.
- **Portfolio Dashboard**: Manage your virtual portfolio with real-time valuation.
- **Sentiment Dashboard**: View sentiment scores and impact analysis.
- **Data Export**: Download predictions and portfolio history as CSV.
- **Dynamic Charts**: Interactive candlestick charts with multiple timeframes (1w, 1m, 3m, 1y, etc.).

## Usage

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
- `GET /api/portfolio` - Get portfolio state
- `POST /api/portfolio/trade` - Execute trade
- `GET /api/sentiment` - Get sentiment analysis

### Web Interface
Visit http://localhost:5173 after starting both servers.

## Technologies

**Backend:**
- Python 3.8+
- Flask & Flask-SocketIO (Real-time API)
- scikit-learn (ML models)
- XGBoost, LightGBM
- TensorFlow/Keras (LSTM/Deep Learning)
- TextBlob (Sentiment Analysis)
- yfinance (Stock data)

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- Framer Motion (Animations)
- Socket.IO Client (Real-time data)
- Plotly (Charts)
- Lucide React (Icons)

## License

MIT License - See LICENSE file for details

## Disclaimer

This project is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.
