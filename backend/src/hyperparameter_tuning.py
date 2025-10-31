import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Hyperparameter optimization using Optuna
    """

    def __init__(self, n_trials=50, random_state=42):
        """
        Initialize tuner

        Args:
            n_trials: Number of optimization trials
            random_state: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.study = None

    def optimize_random_forest(self, X_train, y_train, task_type='classification'):
        """
        Optimize Random Forest hyperparameters

        Args:
            X_train: Training features
            y_train: Training targets
            task_type: 'classification' or 'regression'

        Returns:
            dict: Best hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            if task_type == 'classification':
                model = RandomForestClassifier(**params)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(**params)
                scoring = 'neg_mean_squared_error'

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1)

            return scores.mean()

        # Create study
        direction = 'maximize' if task_type == 'classification' else 'maximize'
        self.study = optuna.create_study(direction=direction, sampler=TPESampler(seed=self.random_state))

        print(f"Optimizing Random Forest ({task_type})...")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")

        return self.best_params

    def optimize_xgboost(self, X_train, y_train, task_type='classification'):
        """
        Optimize XGBoost hyperparameters

        Args:
            X_train: Training features
            y_train: Training targets
            task_type: 'classification' or 'regression'

        Returns:
            dict: Best hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            if task_type == 'classification':
                params['eval_metric'] = 'logloss'
                model = xgb.XGBClassifier(**params)
                scoring = 'accuracy'
            else:
                model = xgb.XGBRegressor(**params)
                scoring = 'neg_mean_squared_error'

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1)

            return scores.mean()

        # Create study
        direction = 'maximize' if task_type == 'classification' else 'maximize'
        self.study = optuna.create_study(direction=direction, sampler=TPESampler(seed=self.random_state))

        print(f"Optimizing XGBoost ({task_type})...")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")

        return self.best_params

    def optimize_lightgbm(self, X_train, y_train, task_type='classification'):
        """
        Optimize LightGBM hyperparameters

        Args:
            X_train: Training features
            y_train: Training targets
            task_type: 'classification' or 'regression'

        Returns:
            dict: Best hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }

            if task_type == 'classification':
                model = lgb.LGBMClassifier(**params)
                scoring = 'accuracy'
            else:
                model = lgb.LGBMRegressor(**params)
                scoring = 'neg_mean_squared_error'

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1)

            return scores.mean()

        # Create study
        direction = 'maximize' if task_type == 'classification' else 'maximize'
        self.study = optuna.create_study(direction=direction, sampler=TPESampler(seed=self.random_state))

        print(f"Optimizing LightGBM ({task_type})...")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")

        return self.best_params

    def optimize_model(self, X_train, y_train, model_type='xgboost', task_type='classification'):
        """
        Optimize specified model type

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: 'random_forest', 'xgboost', or 'lightgbm'
            task_type: 'classification' or 'regression'

        Returns:
            dict: Best hyperparameters
        """
        if model_type == 'random_forest':
            return self.optimize_random_forest(X_train, y_train, task_type)
        elif model_type == 'xgboost':
            return self.optimize_xgboost(X_train, y_train, task_type)
        elif model_type == 'lightgbm':
            return self.optimize_lightgbm(X_train, y_train, task_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def plot_optimization_history(self):
        """
        Plot optimization history
        """
        if self.study is None:
            print("No study available to plot!")
            return

        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()

    def plot_param_importances(self):
        """
        Plot parameter importances
        """
        if self.study is None:
            print("No study available to plot!")
            return

        # Plot parameter importances
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()

    def plot_parallel_coordinate(self):
        """
        Plot parallel coordinate plot
        """
        if self.study is None:
            print("No study available to plot!")
            return

        # Plot parallel coordinate
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.show()

    def get_study_summary(self):
        """
        Get summary of optimization study

        Returns:
            DataFrame: Summary of all trials
        """
        if self.study is None:
            print("No study available!")
            return None

        return self.study.trials_dataframe()


class AutoML:
    """
    Automated Machine Learning for stock prediction
    Combines model selection and hyperparameter optimization
    """

    def __init__(self, n_trials=30, random_state=42):
        """
        Initialize AutoML

        Args:
            n_trials: Number of optimization trials per model
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self.results = {}
        self.best_model_type = None
        self.best_params = None

    def auto_optimize(self, X_train, y_train, task_type='classification',
                     models=['random_forest', 'xgboost', 'lightgbm']):
        """
        Automatically optimize and select best model

        Args:
            X_train: Training features
            y_train: Training targets
            task_type: 'classification' or 'regression'
            models: List of models to try

        Returns:
            dict: Results for all models
        """
        print("="*60)
        print("AUTOMATED MACHINE LEARNING")
        print("="*60)
        print(f"Task: {task_type}")
        print(f"Models to evaluate: {models}")
        print(f"Trials per model: {self.n_trials}")
        print("="*60)

        for model_type in models:
            print(f"\n{'='*60}")
            print(f"Optimizing {model_type.upper()}")
            print('='*60)

            tuner = HyperparameterTuner(n_trials=self.n_trials, random_state=self.random_state)
            best_params = tuner.optimize_model(X_train, y_train, model_type, task_type)

            self.results[model_type] = {
                'best_params': best_params,
                'best_score': tuner.best_score,
                'study': tuner.study
            }

        # Find best model
        best_model_type = max(self.results.keys(), key=lambda k: self.results[k]['best_score'])
        self.best_model_type = best_model_type
        self.best_params = self.results[best_model_type]['best_params']

        print("\n" + "="*60)
        print("AUTOML RESULTS")
        print("="*60)
        print("\nModel Performance Summary:")
        for model, result in self.results.items():
            print(f"  {model:<20}: {result['best_score']:.4f}")

        print(f"\nBest Model: {self.best_model_type}")
        print(f"Best Score: {self.results[self.best_model_type]['best_score']:.4f}")
        print(f"Best Parameters: {self.best_params}")
        print("="*60)

        return self.results

    def train_best_model(self, X_train, y_train, task_type='classification'):
        """
        Train the best model with optimized parameters

        Args:
            X_train: Training features
            y_train: Training targets
            task_type: 'classification' or 'regression'

        Returns:
            Trained model
        """
        if self.best_model_type is None or self.best_params is None:
            raise ValueError("Run auto_optimize first!")

        print(f"\nTraining best model ({self.best_model_type}) with optimized parameters...")

        if self.best_model_type == 'random_forest':
            if task_type == 'classification':
                model = RandomForestClassifier(**self.best_params)
            else:
                model = RandomForestRegressor(**self.best_params)

        elif self.best_model_type == 'xgboost':
            if task_type == 'classification':
                self.best_params['eval_metric'] = 'logloss'
                model = xgb.XGBClassifier(**self.best_params)
            else:
                model = xgb.XGBRegressor(**self.best_params)

        elif self.best_model_type == 'lightgbm':
            if task_type == 'classification':
                model = lgb.LGBMClassifier(**self.best_params)
            else:
                model = lgb.LGBMRegressor(**self.best_params)

        else:
            raise ValueError(f"Unknown model type: {self.best_model_type}")

        model.fit(X_train, y_train)
        print("Training complete!")

        return model


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../src')
    from data_collector import StockDataCollector
    from feature_engineering import FeatureEngineering
    from ml_models import StockPredictor

    # Load and process data
    print("Loading stock data...")
    collector = StockDataCollector()
    data = collector.fetch_stock_data('AAPL', period='2y')

    if data is not None:
        # Feature engineering
        print("\nProcessing features...")
        fe = FeatureEngineering()
        processed_data = fe.prepare_features(data, target_days=1, target_type='direction')

        # Prepare data
        predictor = StockPredictor()
        X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)

        # AutoML optimization
        automl = AutoML(n_trials=20)  # Reduced for demo
        results = automl.auto_optimize(
            X_train, y_train,
            task_type='classification',
            models=['xgboost', 'lightgbm']
        )

        # Train best model
        best_model = automl.train_best_model(X_train, y_train, task_type='classification')

        # Evaluate on test set
        print("\nEvaluating on test set...")
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
