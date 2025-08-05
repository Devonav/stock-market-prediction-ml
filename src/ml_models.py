import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class StockPredictor:
    def __init__(self, model_dir="../models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.is_classifier = None
    
    def prepare_data(self, df, target_col='Target', test_size=0.2, time_series_split=True):
        """
        Prepare data for training
        
        Args:
            df (pandas.DataFrame): Processed stock data
            target_col (str): Name of target column
            test_size (float): Proportion of data for testing
            time_series_split (bool): Whether to use time series split
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Remove non-numeric columns and target column
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        # Remove any columns that might cause data leakage
        leakage_cols = ['Symbol'] + [col for col in feature_cols if 'Target' in col]
        feature_cols = [col for col in feature_cols if col not in leakage_cols]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Using {len(feature_cols)} features: {feature_cols[:5]}..." + (f" and {len(feature_cols)-5} more" if len(feature_cols) > 5 else ""))
        print(f"Dataset shape: {X.shape}")
        
        if time_series_split:
            # For time series, use the last portion as test set
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest', task_type='classification'):
        """
        Train a machine learning model
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type (str): Type of model ('random_forest', 'logistic_regression', 'svm', 'linear_regression')
            task_type (str): 'classification' or 'regression'
        """
        self.is_classifier = (task_type == 'classification')
        
        if task_type == 'classification':
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'logistic_regression':
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == 'svm':
                self.model = SVC(random_state=42, probability=True)
            else:
                raise ValueError(f"Unknown classification model: {model_type}")
        
        else:  # regression
            if model_type == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'linear_regression':
                self.model = LinearRegression()
            elif model_type == 'svm':
                self.model = SVR()
            else:
                raise ValueError(f"Unknown regression model: {model_type}")
        
        print(f"Training {model_type} model for {task_type}...")
        self.model.fit(X_train, y_train)
        print("Model training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.model.predict(X_test)
        
        if self.is_classifier:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics
    
    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Targets
            cv_folds (int): Number of CV folds
        
        Returns:
            dict: Cross-validation scores
        """
        if self.model is None:
            raise ValueError("Model not initialized!")
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        if self.is_classifier:
            scores = cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy')
            return {'cv_accuracy_mean': scores.mean(), 'cv_accuracy_std': scores.std()}
        else:
            scores = cross_val_score(self.model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            return {'cv_mse_mean': -scores.mean(), 'cv_mse_std': scores.std()}
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance for tree-based models
        
        Args:
            top_n (int): Number of top features to return
        
        Returns:
            pandas.DataFrame: Feature importance
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filename):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_path = os.path.join(self.model_dir, f"{filename}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{filename}_scaler.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        feature_path = os.path.join(self.model_dir, f"{filename}_features.joblib")
        joblib.dump(self.feature_columns, feature_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename):
        """Load a saved model and scaler"""
        model_path = os.path.join(self.model_dir, f"{filename}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{filename}_scaler.joblib")
        feature_path = os.path.join(self.model_dir, f"{filename}_features.joblib")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(feature_path)
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities for classification models"""
        if self.model is None or not self.is_classifier:
            raise ValueError("Model not trained or not a classifier!")
        
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict_proba(X_scaled)

def compare_models(X_train, X_test, y_train, y_test, task_type='classification'):
    """
    Compare multiple models and return results
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data
        task_type (str): 'classification' or 'regression'
    
    Returns:
        dict: Results for each model
    """
    if task_type == 'classification':
        models = ['random_forest', 'logistic_regression']
    else:
        models = ['random_forest', 'linear_regression']
    
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type}")
        print('='*50)
        
        predictor = StockPredictor()
        predictor.train_model(X_train, y_train, model_type, task_type)
        
        # Evaluate
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # Get feature importance if available
        feature_importance = predictor.get_feature_importance()
        
        results[model_type] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model': predictor
        }
        
        print(f"Metrics: {metrics}")
    
    return results

if __name__ == "__main__":
    # Example usage
    from data_collector import StockDataCollector
    from feature_engineering import FeatureEngineering
    
    # Load and process data
    collector = StockDataCollector()
    data = collector.fetch_stock_data('AAPL', period='2y')
    
    if data is not None:
        # Feature engineering
        fe = FeatureEngineering()
        processed_data = fe.prepare_features(data, target_days=1, target_type='direction')
        
        # Train and evaluate models
        predictor = StockPredictor()
        X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
        
        # Compare different models
        results = compare_models(X_train, X_test, y_train, y_test, task_type='classification')
        
        # Print comparison
        print(f"\n{'='*50}")
        print("MODEL COMPARISON")
        print('='*50)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in result['metrics'].items():
                print(f"  {metric}: {value:.4f}")