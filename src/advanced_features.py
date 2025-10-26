import pandas as pd
import numpy as np
from scipy import stats


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering including candlestick patterns,
    market regime detection, and sophisticated technical indicators
    """

    def __init__(self):
        pass

    def add_candlestick_patterns(self, df):
        """
        Add candlestick pattern recognition features

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with candlestick pattern features
        """
        data = df.copy()

        # Body and shadow calculations
        data['Body'] = abs(data['Close'] - data['Open'])
        data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
        data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        data['Body_Range_Ratio'] = data['Body'] / (data['High'] - data['Low'] + 1e-10)

        # Doji (small body)
        data['Doji'] = (data['Body'] < (data['High'] - data['Low']) * 0.1).astype(int)

        # Hammer and Hanging Man
        data['Hammer'] = (
            (data['Lower_Shadow'] > 2 * data['Body']) &
            (data['Upper_Shadow'] < data['Body'])
        ).astype(int)

        # Shooting Star
        data['Shooting_Star'] = (
            (data['Upper_Shadow'] > 2 * data['Body']) &
            (data['Lower_Shadow'] < data['Body'])
        ).astype(int)

        # Engulfing patterns
        prev_body = data['Body'].shift(1)
        prev_close = data['Close'].shift(1)
        prev_open = data['Open'].shift(1)

        data['Bullish_Engulfing'] = (
            (data['Close'] > data['Open']) &
            (prev_close < prev_open) &
            (data['Body'] > prev_body)
        ).astype(int)

        data['Bearish_Engulfing'] = (
            (data['Close'] < data['Open']) &
            (prev_close > prev_open) &
            (data['Body'] > prev_body)
        ).astype(int)

        # Morning Star / Evening Star (simplified)
        data['Morning_Star'] = (
            (data['Close'] > data['Open']) &
            (data['Close'].shift(1) < data['Open'].shift(1)) &
            (data['Close'].shift(2) < data['Open'].shift(2)) &
            (data['Body'].shift(1) < data['Body'] * 0.5)
        ).astype(int)

        data['Evening_Star'] = (
            (data['Close'] < data['Open']) &
            (data['Close'].shift(1) > data['Open'].shift(1)) &
            (data['Close'].shift(2) > data['Open'].shift(2)) &
            (data['Body'].shift(1) < data['Body'] * 0.5)
        ).astype(int)

        # Three White Soldiers / Three Black Crows
        data['Three_White_Soldiers'] = (
            (data['Close'] > data['Open']) &
            (data['Close'].shift(1) > data['Open'].shift(1)) &
            (data['Close'].shift(2) > data['Open'].shift(2)) &
            (data['Close'] > data['Close'].shift(1)) &
            (data['Close'].shift(1) > data['Close'].shift(2))
        ).astype(int)

        data['Three_Black_Crows'] = (
            (data['Close'] < data['Open']) &
            (data['Close'].shift(1) < data['Open'].shift(1)) &
            (data['Close'].shift(2) < data['Open'].shift(2)) &
            (data['Close'] < data['Close'].shift(1)) &
            (data['Close'].shift(1) < data['Close'].shift(2))
        ).astype(int)

        # Marubozu (long body with little/no shadow)
        data['Bullish_Marubozu'] = (
            (data['Close'] > data['Open']) &
            (data['Body_Range_Ratio'] > 0.9)
        ).astype(int)

        data['Bearish_Marubozu'] = (
            (data['Close'] < data['Open']) &
            (data['Body_Range_Ratio'] > 0.9)
        ).astype(int)

        return data

    def detect_market_regime(self, df, window=50):
        """
        Detect market regime (trending vs ranging, bullish vs bearish)

        Args:
            df: DataFrame with price data
            window: Lookback window for regime detection

        Returns:
            DataFrame with regime features
        """
        data = df.copy()

        # Trend strength using ADX-like calculation
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()

        # Directional movement
        up_move = data['High'] - data['High'].shift(1)
        down_move = data['Low'].shift(1) - data['Low']

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=data.index).rolling(window=14).mean()
        minus_dm = pd.Series(minus_dm, index=data.index).rolling(window=14).mean()

        plus_di = 100 * plus_dm / (atr + 1e-10)
        minus_di = 100 * minus_dm / (atr + 1e-10)

        data['Trend_Strength'] = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        data['Trend_Direction'] = np.sign(plus_di - minus_di)

        # Market regime classification
        data['Regime_Trending'] = (data['Trend_Strength'] > 0.25).astype(int)
        data['Regime_Bullish'] = (data['Trend_Direction'] > 0).astype(int)

        # Volatility regime
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std()
        vol_percentile = volatility.rolling(window=window*2).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0.5
        )
        data['Regime_High_Volatility'] = (vol_percentile > 0.75).astype(int)
        data['Regime_Low_Volatility'] = (vol_percentile < 0.25).astype(int)

        # Volume regime
        volume_sma = data['Volume'].rolling(window=window).mean()
        data['Regime_High_Volume'] = (data['Volume'] > volume_sma * 1.5).astype(int)

        return data

    def add_market_correlation_features(self, df, market_data=None):
        """
        Add features based on correlation with market indices

        Args:
            df: DataFrame with stock data
            market_data: Optional DataFrame with market index data (e.g., SPY)

        Returns:
            DataFrame with correlation features
        """
        data = df.copy()

        if market_data is not None:
            # Align dates
            common_dates = data.index.intersection(market_data.index)
            stock_returns = data.loc[common_dates, 'Close'].pct_change()
            market_returns = market_data.loc[common_dates, 'Close'].pct_change()

            # Rolling correlation
            rolling_corr = stock_returns.rolling(window=30).corr(market_returns)
            data.loc[common_dates, 'Market_Correlation'] = rolling_corr

            # Beta (systematic risk)
            def calculate_beta(x, y):
                if len(x) < 2 or len(y) < 2:
                    return 1.0
                covariance = np.cov(x, y)[0, 1]
                market_variance = np.var(y)
                return covariance / (market_variance + 1e-10) if market_variance > 0 else 1.0

            rolling_beta = pd.Series(index=common_dates, dtype=float)
            for i in range(30, len(common_dates)):
                idx_range = common_dates[i-30:i]
                rolling_beta.loc[common_dates[i]] = calculate_beta(
                    stock_returns.loc[idx_range].values,
                    market_returns.loc[idx_range].values
                )

            data.loc[common_dates, 'Market_Beta'] = rolling_beta

            # Relative strength
            stock_perf = (data.loc[common_dates, 'Close'] / data.loc[common_dates, 'Close'].shift(20) - 1)
            market_perf = (market_data.loc[common_dates, 'Close'] / market_data.loc[common_dates, 'Close'].shift(20) - 1)
            data.loc[common_dates, 'Relative_Strength'] = stock_perf - market_perf

        return data

    def add_momentum_features(self, df):
        """
        Add advanced momentum indicators

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with momentum features
        """
        data = df.copy()

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            data[f'ROC_{period}'] = (data['Close'] / data['Close'].shift(period) - 1) * 100

        # Momentum
        for period in [5, 10, 20]:
            data[f'Momentum_{period}'] = data['Close'] - data['Close'].shift(period)

        # Williams %R
        for period in [14, 28]:
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            data[f'Williams_R_{period}'] = -100 * (highest_high - data['Close']) / (highest_high - lowest_low + 1e-10)

        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            data[f'CCI_{period}'] = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)

        # Money Flow Index (MFI)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        positive_mf = pd.Series(positive_flow, index=data.index).rolling(window=14).sum()
        negative_mf = pd.Series(negative_flow, index=data.index).rolling(window=14).sum()

        data['MFI'] = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))

        return data

    def add_volatility_features(self, df):
        """
        Add advanced volatility indicators

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with volatility features
        """
        data = df.copy()

        # Historical volatility
        returns = data['Close'].pct_change()
        for period in [10, 20, 30]:
            data[f'Historical_Volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)

        # Parkinson's volatility (uses High-Low range)
        for period in [10, 20]:
            hl_ratio = np.log(data['High'] / data['Low'])
            data[f'Parkinson_Volatility_{period}'] = np.sqrt(
                hl_ratio.rolling(window=period).apply(lambda x: (x**2).sum() / (4 * len(x) * np.log(2)))
            ) * np.sqrt(252)

        # Keltner Channels
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        ema_tp = typical_price.ewm(span=20).mean()
        atr = (data['High'] - data['Low']).rolling(window=10).mean()

        data['Keltner_Upper'] = ema_tp + 2 * atr
        data['Keltner_Lower'] = ema_tp - 2 * atr
        data['Keltner_Position'] = (data['Close'] - data['Keltner_Lower']) / (data['Keltner_Upper'] - data['Keltner_Lower'] + 1e-10)

        # Chaikin Volatility
        hl_range = data['High'] - data['Low']
        hl_ema = hl_range.ewm(span=10).mean()
        data['Chaikin_Volatility'] = (hl_ema - hl_ema.shift(10)) / (hl_ema.shift(10) + 1e-10) * 100

        return data

    def add_volume_analysis_features(self, df):
        """
        Add advanced volume analysis features

        Args:
            df: DataFrame with volume data

        Returns:
            DataFrame with volume features
        """
        data = df.copy()

        # Volume Rate of Change
        for period in [5, 10]:
            data[f'Volume_ROC_{period}'] = (data['Volume'] / data['Volume'].shift(period) - 1) * 100

        # Accumulation/Distribution Line
        clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'] + 1e-10)
        data['AD_Line'] = (clv * data['Volume']).cumsum()

        # Chaikin Money Flow
        mf_volume = clv * data['Volume']
        data['Chaikin_MF'] = mf_volume.rolling(window=20).sum() / (data['Volume'].rolling(window=20).sum() + 1e-10)

        # Volume Price Trend
        data['VPT'] = (data['Volume'] * data['Close'].pct_change()).cumsum()

        # Ease of Movement
        distance = (data['High'] + data['Low']) / 2 - (data['High'].shift(1) + data['Low'].shift(1)) / 2
        box_ratio = (data['Volume'] / 1e6) / (data['High'] - data['Low'] + 1e-10)
        data['EMV'] = distance / (box_ratio + 1e-10)
        data['EMV_MA'] = data['EMV'].rolling(window=14).mean()

        return data

    def prepare_advanced_features(self, df, market_data=None):
        """
        Complete advanced feature engineering pipeline

        Args:
            df: DataFrame with OHLCV data
            market_data: Optional market index data

        Returns:
            DataFrame with all advanced features
        """
        print("Adding candlestick patterns...")
        data = self.add_candlestick_patterns(df)

        print("Detecting market regimes...")
        data = self.detect_market_regime(data)

        print("Adding momentum features...")
        data = self.add_momentum_features(data)

        print("Adding volatility features...")
        data = self.add_volatility_features(data)

        print("Adding volume analysis features...")
        data = self.add_volume_analysis_features(data)

        if market_data is not None:
            print("Adding market correlation features...")
            data = self.add_market_correlation_features(data, market_data)

        print(f"Advanced feature engineering complete! Added features, total columns: {data.shape[1]}")

        return data


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../src')
    from data_collector import StockDataCollector

    # Load data
    print("Loading stock data...")
    collector = StockDataCollector()
    data = collector.fetch_stock_data('AAPL', period='1y')

    if data is not None:
        # Create advanced features
        advanced_fe = AdvancedFeatureEngineering()
        enhanced_data = advanced_fe.prepare_advanced_features(data)

        print("\nSample of new features:")
        print(enhanced_data[['Doji', 'Hammer', 'Regime_Trending', 'ROC_20', 'MFI']].tail(10))

        print(f"\nTotal features created: {enhanced_data.shape[1]}")
        print(f"Data shape: {enhanced_data.shape}")
