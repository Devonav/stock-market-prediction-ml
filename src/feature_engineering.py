import pandas as pd
import numpy as np
import ta
from ta.utils import dropna

class FeatureEngineering:
    def __init__(self):
        pass
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the stock data
        
        Args:
            df (pandas.DataFrame): Stock data with OHLCV columns
        
        Returns:
            pandas.DataFrame: Data with technical indicators
        """
        data = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Clean data - but keep more data for technical indicators
        data = data.dropna()
        
        if len(data) < 50:
            print(f"Warning: Only {len(data)} data points available. Some indicators may not work properly.")
        
        try:
            # Moving Averages
            data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            if len(data) >= 50:
                data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            
            # Exponential Moving Averages
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # MACD
            data['MACD'] = ta.trend.macd(data['Close'])
            data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
            data['MACD_histogram'] = ta.trend.macd_diff(data['Close'])
            
            # RSI
            if len(data) >= 14:
                data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # Bollinger Bands
            data['BB_upper'] = ta.volatility.bollinger_hband(data['Close'])
            data['BB_lower'] = ta.volatility.bollinger_lband(data['Close'])
            data['BB_middle'] = ta.volatility.bollinger_mavg(data['Close'])
            data['BB_width'] = data['BB_upper'] - data['BB_lower']
            
            # Stochastic Oscillator
            if len(data) >= 14:
                data['Stoch_k'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
                data['Stoch_d'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # Average True Range (ATR)
            if len(data) >= 14:
                data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Volume indicators
            data['Volume_SMA'] = ta.volume.volume_sma(data['Close'], data['Volume'], window=20)
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
        except Exception as e:
            print(f"Warning: Error adding some technical indicators: {e}")
        
        return data
    
    def add_price_features(self, df):
        """
        Add price-based features
        
        Args:
            df (pandas.DataFrame): Stock data
        
        Returns:
            pandas.DataFrame: Data with price features
        """
        data = df.copy()
        
        # Price changes
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_2d'] = data['Close'].pct_change(periods=2)
        data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
        
        # High-Low ratio
        data['HL_Ratio'] = data['High'] / data['Low']
        
        # Open-Close ratio
        data['OC_Ratio'] = data['Open'] / data['Close']
        
        # Price position within the day's range
        data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Gap analysis
        data['Gap'] = data['Open'] / data['Close'].shift(1) - 1
        
        # Rolling statistics
        for window in [5, 10, 20]:
            data[f'Close_Mean_{window}d'] = data['Close'].rolling(window=window).mean()
            data[f'Close_Std_{window}d'] = data['Close'].rolling(window=window).std()
            data[f'Volume_Mean_{window}d'] = data['Volume'].rolling(window=window).mean()
            data[f'High_Max_{window}d'] = data['High'].rolling(window=window).max()
            data[f'Low_Min_{window}d'] = data['Low'].rolling(window=window).min()
        
        return data
    
    def add_lagged_features(self, df, lags=[1, 2, 3, 5, 10]):
        """
        Add lagged features
        
        Args:
            df (pandas.DataFrame): Stock data
            lags (list): List of lag periods
        
        Returns:
            pandas.DataFrame: Data with lagged features
        """
        data = df.copy()
        
        features_to_lag = ['Close', 'Volume', 'RSI', 'MACD']
        
        for feature in features_to_lag:
            if feature in data.columns:
                for lag in lags:
                    data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
        
        return data
    
    def create_target_variable(self, df, target_days=1, target_type='price_change'):
        """
        Create target variable for prediction
        
        Args:
            df (pandas.DataFrame): Stock data
            target_days (int): Number of days ahead to predict
            target_type (str): Type of target ('price_change', 'direction', 'price')
        
        Returns:
            pandas.DataFrame: Data with target variable
        """
        data = df.copy()
        
        if target_type == 'price_change':
            # Percentage change in closing price
            data['Target'] = data['Close'].pct_change(periods=target_days).shift(-target_days)
        
        elif target_type == 'direction':
            # Binary: 1 if price goes up, 0 if down
            price_change = data['Close'].pct_change(periods=target_days).shift(-target_days)
            data['Target'] = (price_change > 0).astype(int)
        
        elif target_type == 'price':
            # Future closing price
            data['Target'] = data['Close'].shift(-target_days)
        
        return data
    
    def prepare_features(self, df, target_days=1, target_type='price_change'):
        """
        Complete feature engineering pipeline
        
        Args:
            df (pandas.DataFrame): Raw stock data
            target_days (int): Number of days ahead to predict
            target_type (str): Type of target variable
        
        Returns:
            pandas.DataFrame: Fully processed data
        """
        print("Adding technical indicators...")
        data = self.add_technical_indicators(df)
        
        print("Adding price features...")
        data = self.add_price_features(data)
        
        print("Adding lagged features...")
        data = self.add_lagged_features(data)
        
        print("Creating target variable...")
        data = self.create_target_variable(data, target_days, target_type)
        
        # Remove rows with NaN values
        initial_rows = len(data)
        data = data.dropna()
        final_rows = len(data)
        
        print(f"Removed {initial_rows - final_rows} rows with missing values")
        print(f"Final dataset shape: {data.shape}")
        
        return data

if __name__ == "__main__":
    # Example usage
    from data_collector import StockDataCollector
    
    # Load some sample data
    collector = StockDataCollector()
    data = collector.fetch_stock_data('AAPL', period='1y')
    
    if data is not None:
        # Create feature engineering instance
        fe = FeatureEngineering()
        
        # Process the data
        processed_data = fe.prepare_features(data, target_days=1, target_type='direction')
        
        print("\nFeature columns:")
        print(processed_data.columns.tolist())
        
        print(f"\nData shape: {processed_data.shape}")
        print(f"Target distribution:")
        print(processed_data['Target'].value_counts())