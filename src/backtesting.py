import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class Backtester:
    """
    Advanced backtesting framework for stock trading strategies
    """

    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.001):
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital in dollars
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.portfolio_values = []
        self.results = None

    def run_backtest(self, data, predictions, prediction_type='direction',
                    stop_loss=None, take_profit=None, position_size=1.0):
        """
        Run backtest on historical data

        Args:
            data: DataFrame with OHLCV data (must have 'Close' column and DatetimeIndex)
            predictions: Array of predictions (1 for buy, 0 for sell/hold)
            prediction_type: 'direction' for classification, 'price_change' for regression
            stop_loss: Stop loss percentage (e.g., 0.05 for 5%)
            take_profit: Take profit percentage (e.g., 0.10 for 10%)
            position_size: Fraction of capital to use per trade (0-1)

        Returns:
            dict: Backtest results
        """
        capital = self.initial_capital
        position = 0  # Number of shares held
        entry_price = 0
        trades = []
        portfolio_values = []

        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]

            # Track portfolio value
            portfolio_value = capital + (position * current_price)
            portfolio_values.append({
                'date': current_date,
                'value': portfolio_value,
                'capital': capital,
                'position': position,
                'price': current_price
            })

            # Check stop loss and take profit
            if position > 0:
                price_change = (current_price - entry_price) / entry_price

                # Stop loss hit
                if stop_loss and price_change <= -stop_loss:
                    sell_price = current_price * (1 - self.slippage)
                    proceeds = position * sell_price
                    commission_cost = proceeds * self.commission
                    capital += proceeds - commission_cost

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': sell_price,
                        'shares': position,
                        'profit': proceeds - commission_cost - (position * entry_price),
                        'return': price_change,
                        'exit_reason': 'stop_loss'
                    })

                    position = 0
                    continue

                # Take profit hit
                if take_profit and price_change >= take_profit:
                    sell_price = current_price * (1 - self.slippage)
                    proceeds = position * sell_price
                    commission_cost = proceeds * self.commission
                    capital += proceeds - commission_cost

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': sell_price,
                        'shares': position,
                        'profit': proceeds - commission_cost - (position * entry_price),
                        'return': price_change,
                        'exit_reason': 'take_profit'
                    })

                    position = 0
                    continue

            # Trading logic based on predictions
            if i >= len(predictions):
                continue

            pred = predictions[i]

            # Buy signal (no current position, prediction is bullish)
            if position == 0 and pred == 1:
                # Calculate position size
                invest_amount = capital * position_size
                buy_price = current_price * (1 + self.slippage)
                shares_to_buy = invest_amount / buy_price
                commission_cost = invest_amount * self.commission

                # Execute buy
                position = shares_to_buy
                entry_price = buy_price
                entry_date = current_date
                capital -= (invest_amount + commission_cost)

            # Sell signal (have position, prediction is bearish)
            elif position > 0 and pred == 0:
                sell_price = current_price * (1 - self.slippage)
                proceeds = position * sell_price
                commission_cost = proceeds * self.commission
                capital += proceeds - commission_cost

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': sell_price,
                    'shares': position,
                    'profit': proceeds - commission_cost - (position * entry_price),
                    'return': (sell_price - entry_price) / entry_price,
                    'exit_reason': 'signal'
                })

                position = 0

        # Close any remaining position at the end
        if position > 0:
            final_price = data['Close'].iloc[-1]
            sell_price = final_price * (1 - self.slippage)
            proceeds = position * sell_price
            commission_cost = proceeds * self.commission
            capital += proceeds - commission_cost

            trades.append({
                'entry_date': entry_date,
                'exit_date': data.index[-1],
                'entry_price': entry_price,
                'exit_price': sell_price,
                'shares': position,
                'profit': proceeds - commission_cost - (position * entry_price),
                'return': (sell_price - entry_price) / entry_price,
                'exit_reason': 'end_of_data'
            })

        # Store results
        self.trades = pd.DataFrame(trades)
        self.portfolio_values = pd.DataFrame(portfolio_values)

        # Calculate metrics
        final_value = self.portfolio_values['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        self.results = self._calculate_metrics(total_return, final_value)
        return self.results

    def _calculate_metrics(self, total_return, final_value):
        """
        Calculate performance metrics

        Args:
            total_return: Total return over period
            final_value: Final portfolio value

        Returns:
            dict: Performance metrics
        """
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100
        }

        if len(self.trades) > 0:
            # Trade statistics
            metrics['num_trades'] = len(self.trades)
            metrics['winning_trades'] = len(self.trades[self.trades['profit'] > 0])
            metrics['losing_trades'] = len(self.trades[self.trades['profit'] < 0])
            metrics['win_rate'] = metrics['winning_trades'] / metrics['num_trades'] if metrics['num_trades'] > 0 else 0

            # Profit statistics
            metrics['total_profit'] = self.trades['profit'].sum()
            metrics['avg_profit'] = self.trades['profit'].mean()
            metrics['avg_profit_pct'] = self.trades['return'].mean() * 100
            metrics['best_trade'] = self.trades['profit'].max()
            metrics['worst_trade'] = self.trades['profit'].min()

            # Returns
            returns = self.portfolio_values['value'].pct_change().dropna()
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            metrics['max_drawdown'] = self._calculate_max_drawdown()
            metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100

            # Profit factor
            gross_profit = self.trades[self.trades['profit'] > 0]['profit'].sum()
            gross_loss = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf

        else:
            # No trades executed
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0

        return metrics

    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            float: Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0

        # Annualize returns (assuming daily data)
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = np.sqrt(252) * (excess_returns.mean() / returns.std())
        return sharpe

    def _calculate_max_drawdown(self):
        """
        Calculate maximum drawdown

        Returns:
            float: Maximum drawdown as a fraction
        """
        portfolio_values = self.portfolio_values['value'].values
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        return drawdown.min()

    def get_trade_summary(self):
        """
        Get detailed trade summary

        Returns:
            DataFrame: Trade details
        """
        return self.trades

    def plot_results(self, buy_and_hold_data=None):
        """
        Plot backtest results

        Args:
            buy_and_hold_data: Optional DataFrame with buy-and-hold comparison data
        """
        if self.portfolio_values is None or len(self.portfolio_values) == 0:
            print("No backtest results to plot!")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Portfolio value over time
        axes[0].plot(self.portfolio_values['date'], self.portfolio_values['value'], label='Strategy', linewidth=2)

        # Add buy and hold comparison if provided
        if buy_and_hold_data is not None:
            initial_price = buy_and_hold_data['Close'].iloc[0]
            shares = self.initial_capital / initial_price
            buy_hold_values = shares * buy_and_hold_data['Close']
            axes[0].plot(buy_and_hold_data.index, buy_hold_values, label='Buy & Hold', linewidth=2, alpha=0.7)

        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        portfolio_values = self.portfolio_values['value'].values
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        axes[1].fill_between(self.portfolio_values['date'], drawdown, 0, alpha=0.5, color='red')
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)

        # Cumulative returns
        if len(self.trades) > 0:
            cumulative_returns = (1 + self.trades['return']).cumprod() - 1
            axes[2].plot(range(len(cumulative_returns)), cumulative_returns * 100, linewidth=2)
            axes[2].set_title('Cumulative Trade Returns', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Return (%)')
            axes[2].set_xlabel('Trade Number')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """
        Print backtest summary
        """
        if self.results is None:
            print("No backtest results available!")
            return

        print("="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"\nCapital:")
        print(f"  Initial Capital:        ${self.results['initial_capital']:,.2f}")
        print(f"  Final Value:            ${self.results['final_value']:,.2f}")
        print(f"  Total Return:           {self.results['total_return_pct']:.2f}%")

        if self.results['num_trades'] > 0:
            print(f"\nTrade Statistics:")
            print(f"  Total Trades:           {self.results['num_trades']}")
            print(f"  Winning Trades:         {self.results['winning_trades']}")
            print(f"  Losing Trades:          {self.results['losing_trades']}")
            print(f"  Win Rate:               {self.results['win_rate']*100:.2f}%")

            print(f"\nProfit Analysis:")
            print(f"  Total Profit:           ${self.results['total_profit']:,.2f}")
            print(f"  Average Profit/Trade:   ${self.results['avg_profit']:,.2f}")
            print(f"  Average Return/Trade:   {self.results['avg_profit_pct']:.2f}%")
            print(f"  Best Trade:             ${self.results['best_trade']:,.2f}")
            print(f"  Worst Trade:            ${self.results['worst_trade']:,.2f}")
            print(f"  Profit Factor:          {self.results['profit_factor']:.2f}")

            print(f"\nRisk Metrics:")
            print(f"  Sharpe Ratio:           {self.results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:           {self.results['max_drawdown_pct']:.2f}%")
        else:
            print("\nNo trades executed during backtest period")

        print("="*60)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../src')
    from data_collector import StockDataCollector
    from feature_engineering import FeatureEngineering
    from ml_models import StockPredictor

    # Load data
    print("Loading stock data...")
    collector = StockDataCollector()
    data = collector.fetch_stock_data('AAPL', period='2y')

    if data is not None:
        # Feature engineering
        print("\nProcessing features...")
        fe = FeatureEngineering()
        processed_data = fe.prepare_features(data, target_days=1, target_type='direction')

        # Train model
        print("\nTraining model...")
        predictor = StockPredictor()
        X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
        predictor.train_model(X_train, y_train, 'random_forest', 'classification')

        # Make predictions on test set
        predictions = predictor.predict(processed_data[predictor.feature_columns])

        # Align data and predictions
        test_data = data.iloc[len(data)-len(predictions):]

        # Run backtest
        print("\n" + "="*60)
        print("Running Backtest")
        print("="*60)

        backtester = Backtester(
            initial_capital=10000,
            commission=0.001,  # 0.1%
            slippage=0.001     # 0.1%
        )

        results = backtester.run_backtest(
            test_data,
            predictions,
            prediction_type='direction',
            stop_loss=0.05,      # 5% stop loss
            take_profit=0.10,    # 10% take profit
            position_size=0.95   # Use 95% of capital
        )

        # Print results
        backtester.print_summary()

        # Plot results
        backtester.plot_results(buy_and_hold_data=test_data)
