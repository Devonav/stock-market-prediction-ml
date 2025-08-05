#!/usr/bin/env python3
"""
Stock Price Prediction using Machine Learning

This script demonstrates a complete workflow for stock price prediction including:
1. Data collection from Yahoo Finance
2. Feature engineering with technical indicators
3. Machine learning model training and evaluation
4. Model comparison and selection

Usage:
    python main.py [symbol] [--period PERIOD] [--target-days DAYS] [--model MODEL]

Example:
    python main.py AAPL --period 2y --target-days 1 --model random_forest
"""

import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import StockDataCollector
from feature_engineering import FeatureEngineering
from ml_models import StockPredictor, compare_models

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction using Machine Learning')
    parser.add_argument('symbol', nargs='?', default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--period', default='2y', help='Data period (default: 2y)')
    parser.add_argument('--target-days', type=int, default=1, help='Days ahead to predict (default: 1)')
    parser.add_argument('--target-type', choices=['direction', 'price_change', 'price'], 
                       default='direction', help='Prediction target type (default: direction)')
    parser.add_argument('--model', choices=['random_forest', 'logistic_regression', 'linear_regression'], 
                       default='random_forest', help='Model type (default: random_forest)')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STOCK PRICE PREDICTION USING MACHINE LEARNING")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Target: Predict {args.target_type} {args.target_days} day(s) ahead")
    print(f"Model: {args.model}")
    print("-"*60)
    
    # Step 1: Data Collection
    print("\n1. COLLECTING DATA...")
    collector = StockDataCollector()
    data = collector.fetch_stock_data(args.symbol, period=args.period)
    
    if data is None or data.empty:
        print(f"ERROR: Failed to collect data for {args.symbol}")
        return
    
    print(f"SUCCESS: Collected {len(data)} trading days")
    print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
    
    # Step 2: Feature Engineering
    print("\n2. FEATURE ENGINEERING...")
    fe = FeatureEngineering()
    processed_data = fe.prepare_features(data, target_days=args.target_days, target_type=args.target_type)
    
    print(f"SUCCESS: Created {processed_data.shape[1]-1} features")
    print(f"   Final dataset: {processed_data.shape[0]} samples")
    
    if args.target_type == 'direction':
        up_pct = processed_data['Target'].mean() * 100
        print(f"   Target distribution: {up_pct:.1f}% up days, {100-up_pct:.1f}% down days")
    
    # Step 3: Model Training
    print("\n3. MODEL TRAINING...")
    predictor = StockPredictor()
    X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data, time_series_split=True)
    
    task_type = 'classification' if args.target_type == 'direction' else 'regression'
    
    if args.compare:
        # Compare multiple models
        print("Comparing multiple models...")
        results = compare_models(X_train, X_test, y_train, y_test, task_type=task_type)
        
        # Print comparison table
        print(f"\n{'='*60}")
        print("MODEL COMPARISON RESULTS")
        print('='*60)
        
        comparison_df = pd.DataFrame({model: result['metrics'] for model, result in results.items()}).T
        print(comparison_df.round(4))
        
        # Select best model based on primary metric
        if task_type == 'classification':
            best_model_name = comparison_df['accuracy'].idxmax()
            best_score = comparison_df.loc[best_model_name, 'accuracy']
            print(f"\nBEST MODEL: {best_model_name} (Accuracy: {best_score:.1%})")
        else:
            best_model_name = comparison_df['r2'].idxmax()
            best_score = comparison_df.loc[best_model_name, 'r2']
            print(f"\nBEST MODEL: {best_model_name} (R²: {best_score:.4f})")
        
        best_model = results[best_model_name]['model']
        
    else:
        # Train single model
        predictor.train_model(X_train, y_train, args.model, task_type)
        metrics = predictor.evaluate_model(X_test, y_test)
        
        print(f"SUCCESS: Model trained successfully")
        print(f"   Model type: {args.model}")
        print(f"   Metrics: {metrics}")
        
        best_model = predictor
        best_model_name = args.model
    
    # Step 4: Feature Importance (if available)
    print(f"\n4. FEATURE ANALYSIS...")
    feature_importance = best_model.get_feature_importance(top_n=10)
    
    if feature_importance is not None:
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
            feature_name = row['feature'] if row['feature'] is not None else 'Unknown'
            importance_val = row['importance'] if row['importance'] is not None else 0.0
            print(f"   {i:2d}. {feature_name:<20} ({importance_val:.4f})")
    else:
        print("Feature importance not available for this model type")
    
    # Step 5: Save Model (if requested)
    if args.save_model:
        print(f"\n5. SAVING MODEL...")
        model_name = f"{args.symbol}_{args.target_type}_{args.target_days}d_{best_model_name}"
        best_model.save_model(model_name)
        print(f"SUCCESS: Model saved as: {model_name}")
    
    # Step 6: Recent Predictions
    print(f"\n6. RECENT PREDICTIONS...")
    try:
        # Get last 10 days of data for predictions
        recent_data = processed_data.tail(10)
        if hasattr(best_model, 'feature_columns') and best_model.feature_columns is not None:
            predictions = best_model.predict(recent_data[best_model.feature_columns])
        else:
            print("Cannot make predictions: model feature columns not available")
            return
        
        print("Last 10 trading days:")
        print("-" * 80)
        print(f"{'Date':<12} {'Close':<8} {'Actual':<8} {'Predicted':<10} {'Correct':<8}")
        print("-" * 80)
        
        for i, (date, row) in enumerate(recent_data.iterrows()):
            actual = row['Target']
            pred = predictions[i]
            close_price = row['Close']
            
            if args.target_type == 'direction':
                actual_str = "UP" if actual == 1 else "DOWN"
                pred_str = "UP" if pred == 1 else "DOWN"
                correct = "YES" if actual == pred else "NO"
            else:
                actual_str = f"{actual:.3f}"
                pred_str = f"{pred:.3f}"
                error = abs(actual - pred)
                correct = "YES" if error < 0.02 else "NO"  # 2% threshold
            
            print(f"{date.date():<12} ${close_price:<7.2f} {actual_str:<8} {pred_str:<10} {correct:<8}")
        
        if args.target_type == 'direction':
            accuracy = (recent_data['Target'] == predictions).mean()
            print(f"\nRecent accuracy: {accuracy:.1%}")
    
    except Exception as e:
        print(f"ERROR: Error making recent predictions: {str(e)}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Dataset: {len(processed_data)} samples, {processed_data.shape[1]-1} features")
    print(f"Task: {task_type} ({args.target_type})")
    print(f"Best model: {best_model_name}")
    print(f"Performance: {list(best_model.evaluate_model(X_test, y_test).values())[0]:.3f}")
    
    if args.target_type == 'direction':
        print("\nInterpretation:")
        print("   - This model predicts whether stock price will go UP or DOWN")
        print("   - Accuracy above 55% is generally considered good for daily predictions")
        print("   - Remember: Past performance doesn't guarantee future results!")
    else:
        print("\nInterpretation:")
        print("   - This model predicts the actual price/price change")
        print("   - Lower MSE and higher R² indicate better performance")
        print("   - Consider transaction costs and market volatility in real trading!")
    
    print(f"\nProject structure:")
    print(f"   - Data: data/{args.symbol}_stock_data.csv")
    print(f"   - Models: models/ (if saved)")
    print(f"   - Scripts: src/")
    print(f"   - Notebooks: notebooks/stock_prediction_demo.ipynb")

if __name__ == "__main__":
    main()