import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class StockDataCollector:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_stock_data(self, symbol, period="2y", interval="1d"):
        """
        Fetch stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Period to fetch data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data found for symbol: {symbol}")
                return None
            
            # Add symbol column
            data['Symbol'] = symbol
            
            print(f"Successfully fetched {len(data)} rows of data for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def save_data(self, data, symbol, filename=None):
        """
        Save stock data to CSV file
        
        Args:
            data (pandas.DataFrame): Stock data
            symbol (str): Stock symbol
            filename (str): Optional custom filename
        """
        if data is None or data.empty:
            print("No data to save")
            return
        
        if filename is None:
            filename = f"{symbol}_stock_data.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath)
        print(f"Data saved to: {filepath}")
    
    def get_multiple_stocks(self, symbols, period="2y", interval="1d"):
        """
        Fetch data for multiple stock symbols
        
        Args:
            symbols (list): List of stock symbols
            period (str): Period to fetch data
            interval (str): Data interval
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.fetch_stock_data(symbol, period, interval)
            if data is not None:
                stock_data[symbol] = data
                self.save_data(data, symbol)
        
        return stock_data
    
    def get_stock_info(self, symbol):
        """
        Get basic information about a stock
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            print(f"Error getting info for {symbol}: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()
    
    # Popular tech stocks for demonstration
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    print("Fetching stock data for tech companies...")
    stock_data = collector.get_multiple_stocks(tech_stocks, period="1y", interval="1d")
    
    # Display basic info about collected data
    for symbol, data in stock_data.items():
        if data is not None:
            print(f"\n{symbol} Data Summary:")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"Number of trading days: {len(data)}")
            print(f"Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")