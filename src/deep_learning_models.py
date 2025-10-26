import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import os
import joblib


class DeepLearningPredictor:
    """
    Deep Learning models for stock prediction using LSTM and GRU
    """

    def __init__(self, model_dir="../models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60  # Number of time steps to look back
        self.is_classifier = None

    def prepare_sequences(self, data, target_col='Target', sequence_length=60):
        """
        Prepare sequential data for LSTM/GRU models

        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            sequence_length: Number of time steps to look back

        Returns:
            X, y: Sequences and targets
        """
        self.sequence_length = sequence_length

        # Separate features and target
        feature_cols = [col for col in data.columns if col != target_col]
        features = data[feature_cols].values
        targets = data[target_col].values

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(targets[i])

        X = np.array(X)
        y = np.array(y)

        print(f"Created sequences: X shape = {X.shape}, y shape = {y.shape}")
        return X, y

    def split_data(self, X, y, test_size=0.2):
        """
        Split sequential data into train and test sets (time-series aware)

        Args:
            X: Input sequences
            y: Target values
            test_size: Proportion of data for testing

        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape, task_type='classification', layers=[128, 64], dropout=0.2):
        """
        Build LSTM model

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            task_type: 'classification' or 'regression'
            layers: List of LSTM layer sizes
            dropout: Dropout rate

        Returns:
            Compiled Keras model
        """
        self.is_classifier = (task_type == 'classification')

        model = Sequential()

        # First LSTM layer
        if len(layers) > 1:
            model.add(LSTM(layers[0], return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout))

            # Additional LSTM layers
            for units in layers[1:-1]:
                model.add(LSTM(units, return_sequences=True))
                model.add(Dropout(dropout))

            # Last LSTM layer
            model.add(LSTM(layers[-1], return_sequences=False))
            model.add(Dropout(dropout))
        else:
            model.add(LSTM(layers[0], return_sequences=False, input_shape=input_shape))
            model.add(Dropout(dropout))

        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))

        # Output layer
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print(f"LSTM Model built with {sum([np.prod(layer.shape) for layer in model.trainable_weights])} parameters")
        return model

    def build_gru_model(self, input_shape, task_type='classification', layers=[128, 64], dropout=0.2):
        """
        Build GRU model

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            task_type: 'classification' or 'regression'
            layers: List of GRU layer sizes
            dropout: Dropout rate

        Returns:
            Compiled Keras model
        """
        self.is_classifier = (task_type == 'classification')

        model = Sequential()

        # First GRU layer
        if len(layers) > 1:
            model.add(GRU(layers[0], return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout))

            # Additional GRU layers
            for units in layers[1:-1]:
                model.add(GRU(units, return_sequences=True))
                model.add(Dropout(dropout))

            # Last GRU layer
            model.add(GRU(layers[-1], return_sequences=False))
            model.add(Dropout(dropout))
        else:
            model.add(GRU(layers[0], return_sequences=False, input_shape=input_shape))
            model.add(Dropout(dropout))

        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))

        # Output layer
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print(f"GRU Model built with {sum([np.prod(layer.shape) for layer in model.trainable_weights])} parameters")
        return model

    def build_bidirectional_lstm(self, input_shape, task_type='classification', layers=[128, 64], dropout=0.2):
        """
        Build Bidirectional LSTM model

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            task_type: 'classification' or 'regression'
            layers: List of LSTM layer sizes
            dropout: Dropout rate

        Returns:
            Compiled Keras model
        """
        self.is_classifier = (task_type == 'classification')

        model = Sequential()

        # First Bidirectional LSTM layer
        if len(layers) > 1:
            model.add(Bidirectional(LSTM(layers[0], return_sequences=True), input_shape=input_shape))
            model.add(Dropout(dropout))

            # Additional layers
            for units in layers[1:-1]:
                model.add(Bidirectional(LSTM(units, return_sequences=True)))
                model.add(Dropout(dropout))

            # Last layer
            model.add(Bidirectional(LSTM(layers[-1], return_sequences=False)))
            model.add(Dropout(dropout))
        else:
            model.add(Bidirectional(LSTM(layers[0], return_sequences=False), input_shape=input_shape))
            model.add(Dropout(dropout))

        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))

        # Output layer
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print(f"Bidirectional LSTM Model built")
        return model

    def train_model(self, X_train, y_train, X_val, y_val, model_type='lstm',
                   task_type='classification', epochs=50, batch_size=32):
        """
        Train deep learning model

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_type: 'lstm', 'gru', or 'bidirectional_lstm'
            task_type: 'classification' or 'regression'
            epochs: Number of training epochs
            batch_size: Batch size
        """
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Build model
        if model_type == 'lstm':
            self.model = self.build_lstm_model(input_shape, task_type)
        elif model_type == 'gru':
            self.model = self.build_gru_model(input_shape, task_type)
        elif model_type == 'bidirectional_lstm':
            self.model = self.build_bidirectional_lstm(input_shape, task_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        # Train
        print(f"\nTraining {model_type} model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        print("Training completed!")
        return history

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
        y_pred = y_pred.flatten()

        if self.is_classifier:
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred_binary),
                'loss': float(self.model.evaluate(X_test, y_test, verbose=0)[0])
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

        return metrics

    def predict(self, X):
        """
        Make predictions on new data

        Args:
            X: Input sequences

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")

        predictions = self.model.predict(X)

        if self.is_classifier:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return predictions.flatten()

    def save_model(self, filename):
        """
        Save the trained model and scaler

        Args:
            filename: Base name for saved files
        """
        if self.model is None:
            raise ValueError("No model to save!")

        model_path = os.path.join(self.model_dir, f"{filename}_dl_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{filename}_dl_scaler.joblib")

        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'is_classifier': self.is_classifier
        }
        metadata_path = os.path.join(self.model_dir, f"{filename}_dl_metadata.joblib")
        joblib.dump(metadata, metadata_path)

        print(f"Model saved to {model_path}")

    def load_model(self, filename):
        """
        Load a saved model and scaler

        Args:
            filename: Base name of saved files
        """
        model_path = os.path.join(self.model_dir, f"{filename}_dl_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{filename}_dl_scaler.joblib")
        metadata_path = os.path.join(self.model_dir, f"{filename}_dl_metadata.joblib")

        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        metadata = joblib.load(metadata_path)

        self.sequence_length = metadata['sequence_length']
        self.is_classifier = metadata['is_classifier']

        print(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../src')
    from data_collector import StockDataCollector
    from feature_engineering import FeatureEngineering

    # Load and process data
    print("Loading stock data...")
    collector = StockDataCollector()
    data = collector.fetch_stock_data('AAPL', period='2y')

    if data is not None:
        # Feature engineering
        print("\nProcessing features...")
        fe = FeatureEngineering()
        processed_data = fe.prepare_features(data, target_days=1, target_type='direction')

        # Prepare sequences
        print("\nPreparing sequences for deep learning...")
        dl_predictor = DeepLearningPredictor()
        X, y = dl_predictor.prepare_sequences(processed_data, target_col='Target', sequence_length=60)

        # Split data
        X_train, X_test, y_train, y_test = dl_predictor.split_data(X, y, test_size=0.2)

        # Further split training data for validation
        X_train, X_val, y_train, y_val = dl_predictor.split_data(X_train, y_train, test_size=0.1)

        # Train LSTM model
        print("\n" + "="*60)
        print("Training LSTM Model")
        print("="*60)
        history = dl_predictor.train_model(
            X_train, y_train, X_val, y_val,
            model_type='lstm',
            task_type='classification',
            epochs=50,
            batch_size=32
        )

        # Evaluate
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        metrics = dl_predictor.evaluate_model(X_test, y_test)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
