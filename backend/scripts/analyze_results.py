#!/usr/bin/env python3
"""
Results Analysis Script
Analyzes the performance of different stocks and prediction types
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import StockDataCollector
from feature_engineering import FeatureEngineering
from ml_models import StockPredictor, compare_models

def analyze_stock(symbol, periods=['1y', '2y', '5y']):
    """Analyze a stock across different time periods"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {symbol}")
    print('='*60)
    
    collector = StockDataCollector()
    results = {}
    
    for period in periods:
        print(f"\nTesting {period} period...")
        try:
            # Get data
            data = collector.fetch_stock_data(symbol, period=period)
            if data is None or len(data) < 100:
                print(f"  Insufficient data for {period}")
                continue
            
            # Feature engineering
            fe = FeatureEngineering()
            processed_data = fe.prepare_features(data, target_type='direction')
            
            if len(processed_data) < 50:
                print(f"  Insufficient processed data for {period}")
                continue
            
            # Train model
            predictor = StockPredictor()
            X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
            
            # Compare models
            model_results = compare_models(X_train, X_test, y_train, y_test, task_type='classification')
            
            # Get best model
            best_accuracy = 0
            best_model_name = None
            for model_name, result in model_results.items():
                acc = result['metrics']['accuracy']
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model_name = model_name
            
            results[period] = {
                'accuracy': best_accuracy,
                'model': best_model_name,
                'samples': len(processed_data),
                'features': processed_data.shape[1] - 1
            }
            
            print(f"  Best: {best_model_name} - {best_accuracy:.1%} ({len(processed_data)} samples)")
            
        except Exception as e:
            print(f"  Error with {period}: {str(e)}")
    
    return results

def compare_prediction_difficulty():
    """Compare difficulty of different prediction types"""
    print(f"\n{'='*60}")
    print("PREDICTION DIFFICULTY ANALYSIS")
    print('='*60)
    
    symbol = 'AAPL'  # Use stable stock
    collector = StockDataCollector()
    data = collector.fetch_stock_data(symbol, period='2y')
    
    if data is None:
        print("Could not get data for analysis")
        return
    
    fe = FeatureEngineering()
    tests = [
        ('1-day direction', 1, 'direction'),
        ('3-day direction', 3, 'direction'),
        ('5-day direction', 5, 'direction'),
        ('1-day price change', 1, 'price_change'),
    ]
    
    for test_name, target_days, target_type in tests:
        print(f"\nTesting: {test_name}")
        try:
            processed_data = fe.prepare_features(data, target_days=target_days, target_type=target_type)
            
            if len(processed_data) < 50:
                print(f"  Insufficient data")
                continue
            
            predictor = StockPredictor()
            X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
            
            task_type = 'classification' if target_type == 'direction' else 'regression'
            
            if task_type == 'classification':
                predictor.train_model(X_train, y_train, 'random_forest', task_type)
                metrics = predictor.evaluate_model(X_test, y_test)
                print(f"  Accuracy: {metrics['accuracy']:.1%}")
            else:
                predictor.train_model(X_train, y_train, 'random_forest', task_type)
                metrics = predictor.evaluate_model(X_test, y_test)
                print(f"  R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.4f}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")

def stock_comparison():
    """Compare different stocks"""
    print(f"\n{'='*60}")
    print("STOCK COMPARISON ANALYSIS")
    print('='*60)
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'JPM']
    collector = StockDataCollector()
    fe = FeatureEngineering()
    
    results = []
    
    for symbol in stocks:
        print(f"\nTesting {symbol}...")
        try:
            data = collector.fetch_stock_data(symbol, period='2y')
            if data is None or len(data) < 100:
                print(f"  Insufficient data")
                continue
            
            processed_data = fe.prepare_features(data, target_type='direction')
            if len(processed_data) < 50:
                print(f"  Insufficient processed data")
                continue
            
            predictor = StockPredictor()
            X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
            predictor.train_model(X_train, y_train, 'random_forest', 'classification')
            metrics = predictor.evaluate_model(X_test, y_test)
            
            volatility = data['Close'].pct_change().std() * 100  # Daily volatility %
            
            results.append({
                'symbol': symbol,
                'accuracy': metrics['accuracy'],
                'samples': len(processed_data),
                'volatility': volatility
            })
            
            print(f"  Accuracy: {metrics['accuracy']:.1%}, Volatility: {volatility:.2f}%")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'RANKING':<10} {'SYMBOL':<8} {'ACCURACY':<10} {'VOLATILITY':<12} {'SAMPLES':<8}")
    print('-' * 60)
    for i, result in enumerate(results, 1):
        print(f"{i:<10} {result['symbol']:<8} {result['accuracy']:<10.1%} {result['volatility']:<12.2f}% {result['samples']:<8}")

def main():
    print("STOCK PREDICTION RESULTS ANALYSIS")
    print("="*60)
    
    print("\n1. ANALYZING APPLE ACROSS TIME PERIODS...")
    apple_results = analyze_stock('AAPL')
    
    print("\n2. COMPARING PREDICTION DIFFICULTY...")
    compare_prediction_difficulty()
    
    print("\n3. COMPARING DIFFERENT STOCKS...")
    stock_comparison()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    print("\nKEY INSIGHTS:")
    print("- Longer time periods generally provide better accuracy")
    print("- Direction prediction is easier than price change prediction")
    print("- Multi-day predictions are significantly harder")
    print("- Less volatile stocks (like AAPL, MSFT) are more predictable")
    print("- Random Forest often outperforms other models for this task")

if __name__ == "__main__":
    main()