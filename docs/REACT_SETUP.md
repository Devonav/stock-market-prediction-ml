# React Frontend Setup Guide

## Overview

Your Stock Market Prediction app now has a modern React frontend with animated components inspired by React Bits! This replaces the Streamlit interface with a much more polished, interactive experience.

## Architecture

### Backend (Python)
- **Flask REST API** (`api.py`)
- Serves ML predictions, data, and backtesting
- Runs on port 5000

### Frontend (React)
- **React + Vite** for fast development
- **Framer Motion** for animations (React Bits style)
- **Plotly** for interactive charts
- **TailwindCSS** for styling
- Runs on port 3000

## Installation

### 1. Install Backend Dependencies

```bash
# Install Flask API dependencies
pip install -r requirements-api.txt
```

### 2. Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install
```

## Running the Application

You need to run both the backend and frontend servers.

### Terminal 1: Start Backend API

```bash
# From the root directory (Stock_Market/)
python api.py
```

The API will start on http://localhost:5000

### Terminal 2: Start Frontend

```bash
# Navigate to frontend directory
cd frontend

# Start development server
npm run dev
```

The app will start on http://localhost:3000

## Features

### Animated Components (React Bits Inspired)

- **AnimatedCard**: Smooth fade-in and slide-up animations
- **MetricCard**: Animated metric displays with icons
- **PulseCard**: Pulsing shadow effect for emphasis
- **AnimatedButton**: Hover and tap animations
- **AnimatedInput**: Smooth focus transitions

### Main Dashboard Tabs

1. **Predictions**
   - Real-time stock predictions
   - Interactive candlestick charts
   - Feature importance visualization
   - Recent prediction history
   - Model performance metrics

2. **Model Comparison**
   - Compare all ML models side-by-side
   - Visual performance charts
   - Best model highlighting
   - Comprehensive metrics table

3. **Backtesting**
   - Simulated trading performance
   - Portfolio value over time
   - Drawdown visualization
   - Complete trade history
   - Configurable risk parameters

### Design Features

- **Gradient backgrounds** and modern color schemes
- **Smooth animations** on all interactions
- **Responsive design** for mobile and desktop
- **Interactive charts** with zoom and pan
- **Real-time loading states** and error handling
- **Professional UI** with Tailwind CSS

## API Endpoints

### `POST /api/predict`
Make stock predictions
```json
{
  "symbol": "AAPL",
  "period": "2y",
  "model_type": "xgboost",
  "target_type": "direction",
  "target_days": 1,
  "use_advanced_features": false
}
```

### `POST /api/compare-models`
Compare all models
```json
{
  "symbol": "AAPL",
  "period": "2y",
  "target_type": "direction",
  "target_days": 1,
  "use_advanced_features": false
}
```

### `POST /api/backtest`
Run backtesting
```json
{
  "symbol": "AAPL",
  "model_type": "xgboost",
  "target_type": "direction",
  "initial_capital": 10000,
  "stop_loss": 0.05,
  "take_profit": 0.10,
  "commission": 0.001,
  "slippage": 0.001
}
```

### `POST /api/chart-data`
Get chart data
```json
{
  "symbol": "AAPL",
  "period": "6mo"
}
```

## Development

### Build for Production

```bash
cd frontend
npm run build
```

The production build will be in `frontend/dist/`

### Preview Production Build

```bash
npm run preview
```

## Customization

### Changing Colors

Edit `frontend/tailwind.config.js` to customize the color scheme:

```js
theme: {
  extend: {
    colors: {
      primary: {
        // Your custom colors
      }
    }
  }
}
```

### Adding New Components

Create new components in `frontend/src/components/` and use the animated wrappers:

```jsx
import { AnimatedCard } from './AnimatedCard';

const MyComponent = () => (
  <AnimatedCard delay={0.2}>
    {/* Your content */}
  </AnimatedCard>
);
```

## Troubleshooting

### CORS Issues
Make sure Flask-CORS is installed and the API is running on port 5000.

### API Connection Failed
1. Ensure the backend API is running (`python api.py`)
2. Check that the API URL in `frontend/src/services/api.js` matches your setup

### Build Errors
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Next Steps

### Enhancements You Can Add:

1. **Real-time Data** - Add WebSocket support for live price updates
2. **User Authentication** - Add login/signup with JWT
3. **Saved Strategies** - Let users save and load trading strategies
4. **More Charts** - Add technical indicator overlays
5. **Export Reports** - Generate PDF reports of predictions
6. **Dark Mode** - Add theme toggling
7. **Mobile App** - Convert to React Native

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool
- **TailwindCSS** - Styling
- **Framer Motion** - Animations
- **Plotly.js** - Charts
- **Axios** - HTTP client
- **Flask** - Backend API
- **Python ML** - XGBoost, LightGBM, etc.

## Comparison: Streamlit vs React

| Feature | Streamlit | React |
|---------|-----------|-------|
| **Development Speed** | Fast | Moderate |
| **Customization** | Limited | Unlimited |
| **Animations** | Basic | Advanced |
| **Performance** | Good | Excellent |
| **Mobile** | Basic | Full control |
| **UI/UX** | Functional | Professional |

## License

Educational purposes only. Not financial advice.

---

Enjoy your new modern React frontend! 🚀
