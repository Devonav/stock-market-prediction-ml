"""
Flask REST API for Stock Market Prediction
Provides endpoints for ML predictions, data fetching, and backtesting
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import StockDataCollector
from feature_engineering import FeatureEngineering
from advanced_features import AdvancedFeatureEngineering
from ml_models import StockPredictor, compare_models
from backtesting import Backtester
from sentiment_analyzer import SentimentAnalyzer
from portfolio_manager import PortfolioManager
from websocket_server import WebSocketServer

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize WebSocket
ws = WebSocketServer()
ws.init_app(app)

# Global cache for models
model_cache = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})


@app.route('/api/stock/data', methods=['POST'])
def get_stock_data():
    """
    Fetch stock data
    Body: { symbol, period }
    """
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        period = data.get('period', '2y')

        collector = StockDataCollector()
        stock_data = collector.fetch_stock_data(symbol, period=period)

        if stock_data is None or len(stock_data) == 0:
            return jsonify({"error": f"Could not fetch data for {symbol}"}), 404

        # Convert to JSON-serializable format
        result = {
            "symbol": symbol,
            "period": period,
            "data": stock_data.reset_index().to_dict(orient='records'),
            "latest_price": float(stock_data['Close'].iloc[-1]),
            "total_days": len(stock_data)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    """
    Get sentiment analysis for a stock
    Query params: symbol
    """
    try:
        symbol = request.args.get('symbol', 'AAPL')
        
        analyzer = SentimentAnalyzer()
        result = analyzer.get_sentiment(symbol)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make stock predictions
    Body: {
        symbol,
        period,
        model_type,
        target_type,
        target_days,
        use_advanced_features
    }
    """
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        period = data.get('period', '2y')
        model_type = data.get('model_type', 'xgboost')
        target_type = data.get('target_type', 'direction')
        target_days = data.get('target_days', 1)
        use_advanced_features = data.get('use_advanced_features', False)

        # Step 1: Collect data
        collector = StockDataCollector()
        stock_data = collector.fetch_stock_data(symbol, period=period)

        if stock_data is None or len(stock_data) == 0:
            return jsonify({"error": f"Could not fetch data for {symbol}"}), 404

        # Step 2: Feature engineering
        # Get sentiment if requested (implicitly used if available)
        analyzer = SentimentAnalyzer()
        sentiment_data = analyzer.get_sentiment(symbol)
        sentiment_score = sentiment_data['score']
        
        fe = FeatureEngineering()
        processed_data = fe.prepare_features(stock_data, target_days=target_days, target_type=target_type, sentiment_score=sentiment_score)

        if use_advanced_features:
            advanced_fe = AdvancedFeatureEngineering()
            advanced_data = advanced_fe.prepare_advanced_features(stock_data)
            for col in advanced_data.columns:
                if col not in processed_data.columns and col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    processed_data[col] = advanced_data[col]
            processed_data = processed_data.dropna()

        # Step 3: Train model
        predictor = StockPredictor()
        X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data, time_series_split=True)

        task_type = 'classification' if target_type == 'direction' else 'regression'
        predictor.train_model(X_train, y_train, model_type, task_type)

        # Step 4: Make predictions
        predictions = predictor.predict(processed_data[predictor.feature_columns])

        # Step 5: Evaluate
        metrics = predictor.evaluate_model(X_test, y_test)

        # Get feature importance
        feature_importance = predictor.get_feature_importance(top_n=20)

        # Prepare recent predictions
        recent_data = processed_data.tail(20)
        recent_preds = predictions[-20:]

        result = {
            "symbol": symbol,
            "model_type": model_type,
            "target_type": target_type,
            "latest_prediction": int(predictions[-1]) if target_type == 'direction' else float(predictions[-1]),
            "latest_price": float(stock_data['Close'].iloc[-1]),
            "metrics": {k: float(v) if not isinstance(v, str) else v for k, v in metrics.items()},
            "feature_importance": feature_importance.to_dict(orient='records') if feature_importance is not None else None,
            "recent_predictions": [
                {
                    "date": str(date),
                    "close": float(close),
                    "actual": int(actual) if target_type == 'direction' else float(actual),
                    "predicted": int(pred) if target_type == 'direction' else float(pred)
                }
                for date, close, actual, pred in zip(
                    recent_data.index,
                    recent_data['Close'].values,
                    recent_data['Target'].values,
                    recent_preds
                )
            ],
            "total_features": processed_data.shape[1] - 1,
            "total_samples": processed_data.shape[0]
        }

        # Cache the model and data for backtesting
        cache_key = f"{symbol}_{model_type}_{target_type}"
        model_cache[cache_key] = {
            "predictor": predictor,
            "processed_data": processed_data,
            "predictions": predictions,
            "stock_data": stock_data
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/compare-models', methods=['POST'])
def compare_all_models():
    """
    Compare multiple models
    Body: { symbol, period, target_type, target_days, use_advanced_features }
    """
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        period = data.get('period', '2y')
        target_type = data.get('target_type', 'direction')
        target_days = data.get('target_days', 1)
        use_advanced_features = data.get('use_advanced_features', False)

        # Collect and process data
        collector = StockDataCollector()
        stock_data = collector.fetch_stock_data(symbol, period=period)

        if stock_data is None or len(stock_data) == 0:
            return jsonify({"error": f"Could not fetch data for {symbol}"}), 404

        fe = FeatureEngineering()
        processed_data = fe.prepare_features(stock_data, target_days=target_days, target_type=target_type)

        if use_advanced_features:
            advanced_fe = AdvancedFeatureEngineering()
            advanced_data = advanced_fe.prepare_advanced_features(stock_data)
            for col in advanced_data.columns:
                if col not in processed_data.columns and col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    processed_data[col] = advanced_data[col]
            processed_data = processed_data.dropna()

        # Compare models
        predictor = StockPredictor()
        X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data, time_series_split=True)

        task_type = 'classification' if target_type == 'direction' else 'regression'
        results = compare_models(
            X_train, X_test, y_train, y_test,
            task_type=task_type,
            feature_columns=predictor.feature_columns,
            scaler=predictor.scaler
        )

        # Convert results to JSON-serializable format
        comparison = {
            model_name: {
                "metrics": {k: float(v) if not isinstance(v, str) else v for k, v in result['metrics'].items()}
            }
            for model_name, result in results.items()
        }

        return jsonify({
            "symbol": symbol,
            "target_type": target_type,
            "comparison": comparison
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """
    Run backtesting
    Body: {
        symbol,
        model_type,
        target_type,
        initial_capital,
        stop_loss,
        take_profit,
        commission,
        slippage
    }
    """
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        model_type = data.get('model_type', 'xgboost')
        target_type = data.get('target_type', 'direction')
        initial_capital = data.get('initial_capital', 10000)
        stop_loss = data.get('stop_loss', 0.05)
        take_profit = data.get('take_profit', 0.10)
        commission = data.get('commission', 0.001)
        slippage = data.get('slippage', 0.001)

        # Get cached model and data
        cache_key = f"{symbol}_{model_type}_{target_type}"
        if cache_key not in model_cache:
            return jsonify({"error": "No cached model found. Please run prediction first."}), 400

        cached = model_cache[cache_key]
        predictions = cached['predictions']
        stock_data = cached['stock_data']

        # Run backtest
        backtester = Backtester(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )

        test_data = stock_data.iloc[-len(predictions):]

        results = backtester.run_backtest(
            test_data,
            predictions,
            prediction_type=target_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=0.95
        )

        # Convert results to JSON-serializable format
        result = {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_value": float(results['final_value']),
            "total_return": float(results['total_return']),
            "total_return_pct": float(results['total_return_pct']),
            "num_trades": int(results.get('num_trades', 0)),
            "win_rate": float(results.get('win_rate', 0)),
            "sharpe_ratio": float(results.get('sharpe_ratio', 0)),
            "max_drawdown": float(results.get('max_drawdown', 0)),
            "portfolio_values": backtester.portfolio_values.to_dict(orient='records') if backtester.portfolio_values is not None else [],
            "trades": backtester.trades.to_dict(orient='records') if backtester.trades is not None else []
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stock-data', methods=['GET'])
def get_stock_data_simple():
    """
    Get stock data via GET request
    Query params: symbol, period
    """
    try:
        symbol = request.args.get('symbol', 'AAPL')
        period = request.args.get('period', '6mo')

        # Get stock data
        collector = StockDataCollector()
        stock_data = collector.fetch_stock_data(symbol, period=period)

        if stock_data is None or len(stock_data) == 0:
            return jsonify({"error": f"Could not fetch data for {symbol}"}), 404

        # Prepare candlestick data
        candles = []
        volume = []

        for idx in stock_data.index:
            timestamp = int(idx.timestamp())
            candles.append({
                "time": timestamp,
                "open": float(stock_data.loc[idx, 'Open']),
                "high": float(stock_data.loc[idx, 'High']),
                "low": float(stock_data.loc[idx, 'Low']),
                "close": float(stock_data.loc[idx, 'Close'])
            })
            volume.append({
                "time": timestamp,
                "value": float(stock_data.loc[idx, 'Volume']),
                "color": 'rgba(34, 197, 94, 0.5)' if stock_data.loc[idx, 'Close'] > stock_data.loc[idx, 'Open'] else 'rgba(239, 68, 68, 0.5)'
            })

        return jsonify({
            "candles": candles,
            "volume": volume
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chart-data', methods=['POST'])
def get_chart_data():
    """
    Get data for charts (candlestick, indicators)
    Body: { symbol, period }
    """
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        period = data.get('period', '6mo')

        # Get stock data
        collector = StockDataCollector()
        stock_data = collector.fetch_stock_data(symbol, period=period)

        if stock_data is None or len(stock_data) == 0:
            return jsonify({"error": f"Could not fetch data for {symbol}"}), 404

        # Add indicators
        fe = FeatureEngineering()
        processed_data = fe.prepare_features(stock_data, target_type='direction')

        # Prepare chart data
        chart_data = []
        for idx in stock_data.index:
            point = {
                "date": str(idx),
                "open": float(stock_data.loc[idx, 'Open']),
                "high": float(stock_data.loc[idx, 'High']),
                "low": float(stock_data.loc[idx, 'Low']),
                "close": float(stock_data.loc[idx, 'Close']),
                "volume": float(stock_data.loc[idx, 'Volume'])
            }

            # Add indicators if available
            if idx in processed_data.index:
                if 'SMA_20' in processed_data.columns:
                    point['sma_20'] = float(processed_data.loc[idx, 'SMA_20'])
                if 'SMA_50' in processed_data.columns:
                    point['sma_50'] = float(processed_data.loc[idx, 'SMA_50'])
                if 'RSI' in processed_data.columns:
                    point['rsi'] = float(processed_data.loc[idx, 'RSI'])

            chart_data.append(point)

        return jsonify({
            "symbol": symbol,
            "period": period,
            "data": chart_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio state"""
    try:
        # Get current prices for all holdings
        pm = PortfolioManager()
        holdings = pm.get_portfolio_state()['holdings']
        
        current_prices = {}
        
        # Try to get simulated prices first (faster)
        simulated_prices = ws.get_latest_prices()
        
        if holdings:
            collector = StockDataCollector()
            for holding in holdings:
                symbol = holding['symbol']
                
                if symbol in simulated_prices:
                    current_prices[symbol] = simulated_prices[symbol]
                else:
                    # Fallback to fetch latest price if not in simulation
                    try:
                        stock_data = collector.fetch_stock_data(symbol, period='1d')
                        if stock_data is not None and not stock_data.empty:
                            current_prices[symbol] = float(stock_data['Close'].iloc[-1])
                    except Exception as e:
                        print(f"Error fetching price for {symbol}: {e}")
        
        return jsonify(pm.get_portfolio_state(current_prices))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio/trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    try:
        data = request.json
        symbol = data.get('symbol')
        action = data.get('action')
        quantity = int(data.get('quantity', 0))
        
        if not symbol or not action or quantity <= 0:
            return jsonify({"error": "Invalid trade parameters"}), 400
            
        # Get current price
        simulated_prices = ws.get_latest_prices()
        
        if symbol in simulated_prices:
            current_price = simulated_prices[symbol]
        else:
            # Fallback to fetch latest price
            collector = StockDataCollector()
            stock_data = collector.fetch_stock_data(symbol, period='1d')
            
            if stock_data is None or stock_data.empty:
                return jsonify({"error": "Could not fetch current price"}), 400
                
            current_price = float(stock_data['Close'].iloc[-1])
        
        pm = PortfolioManager()
        result = pm.execute_trade(symbol, action, quantity, current_price)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio/reset', methods=['POST'])
def reset_portfolio():
    """Reset portfolio"""
    try:
        pm = PortfolioManager()
        return jsonify(pm.reset_portfolio())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Use ws.run instead of app.run to enable SocketIO
    ws.run(app, debug=True, port=5000)
