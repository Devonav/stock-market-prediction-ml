import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np


class StockVisualization:
    """
    Interactive visualization dashboard using Plotly
    """

    def __init__(self):
        self.theme = 'plotly_dark'

    def plot_candlestick(self, data, predictions=None, title="Stock Price Analysis"):
        """
        Create interactive candlestick chart with optional predictions

        Args:
            data: DataFrame with OHLCV data
            predictions: Optional predictions array
            title: Chart title
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(title, 'Volume', 'Indicators')
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # Add moving averages if available
        ma_columns = [col for col in data.columns if 'SMA' in col or 'EMA' in col]
        colors = ['blue', 'orange', 'purple', 'cyan', 'magenta']
        for i, ma_col in enumerate(ma_columns[:5]):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[ma_col],
                    name=ma_col,
                    line=dict(width=1, color=colors[i % len(colors)])
                ),
                row=1, col=1
            )

        # Bollinger Bands if available
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_upper'],
                    name='BB Upper',
                    line=dict(width=1, dash='dash', color='gray'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_lower'],
                    name='BB Lower',
                    line=dict(width=1, dash='dash', color='gray'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.2)',
                    showlegend=False
                ),
                row=1, col=1
            )

        # Prediction markers
        if predictions is not None and len(predictions) == len(data):
            buy_signals = data.index[predictions == 1]
            sell_signals = data.index[predictions == 0]

            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals,
                        y=data.loc[buy_signals, 'Low'] * 0.99,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(symbol='triangle-up', size=10, color='lime')
                    ),
                    row=1, col=1
                )

            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals,
                        y=data.loc[sell_signals, 'High'] * 1.01,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(symbol='triangle-down', size=10, color='red')
                    ),
                    row=1, col=1
                )

        # Volume bars
        colors_volume = ['red' if close < open else 'green'
                        for close, open in zip(data['Close'], data['Open'])]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors_volume,
                showlegend=False
            ),
            row=2, col=1
        )

        # RSI if available
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )

            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        # Update layout
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(
            height=900,
            showlegend=True,
            template=self.theme,
            hovermode='x unified'
        )

        fig.show()

    def plot_feature_importance(self, feature_importance, top_n=20, title="Feature Importance"):
        """
        Plot feature importance

        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            title: Chart title
        """
        if feature_importance is None or len(feature_importance) == 0:
            print("No feature importance data available!")
            return

        # Get top features
        top_features = feature_importance.head(top_n).sort_values('importance', ascending=True)

        fig = go.Figure(
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            template=self.theme
        )

        fig.show()

    def plot_prediction_performance(self, y_true, y_pred, dates=None, title="Prediction Performance"):
        """
        Plot actual vs predicted values

        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional date index
            title: Chart title
        """
        if dates is None:
            dates = list(range(len(y_true)))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_true,
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_pred,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            template=self.theme,
            hovermode='x unified'
        )

        fig.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels=['Down', 'Up'], title="Confusion Matrix"):
        """
        Plot confusion matrix for classification

        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            labels: Class labels
            title: Chart title
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=True
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500,
            template=self.theme
        )

        fig.show()

    def plot_backtest_results(self, portfolio_values, trades=None, title="Backtest Results"):
        """
        Plot backtest results with portfolio value and trades

        Args:
            portfolio_values: DataFrame with portfolio value over time
            trades: Optional DataFrame with trade history
            title: Chart title
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'Drawdown')
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_values['date'],
                y=portfolio_values['value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Initial capital line
        initial_value = portfolio_values['value'].iloc[0]
        fig.add_hline(
            y=initial_value,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Initial Capital",
            row=1, col=1
        )

        # Trade markers
        if trades is not None and len(trades) > 0:
            # Entry points
            fig.add_trace(
                go.Scatter(
                    x=trades['entry_date'],
                    y=[portfolio_values[portfolio_values['date'] == d]['value'].iloc[0]
                       if len(portfolio_values[portfolio_values['date'] == d]) > 0 else initial_value
                       for d in trades['entry_date']],
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=12, color='green')
                ),
                row=1, col=1
            )

            # Exit points
            fig.add_trace(
                go.Scatter(
                    x=trades['exit_date'],
                    y=[portfolio_values[portfolio_values['date'] == d]['value'].iloc[0]
                       if len(portfolio_values[portfolio_values['date'] == d]) > 0 else initial_value
                       for d in trades['exit_date']],
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=12, color='red')
                ),
                row=1, col=1
            )

        # Drawdown
        portfolio_vals = portfolio_values['value'].values
        cummax = np.maximum.accumulate(portfolio_vals)
        drawdown = (portfolio_vals - cummax) / cummax * 100

        fig.add_trace(
            go.Scatter(
                x=portfolio_values['date'],
                y=drawdown,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=800,
            template=self.theme,
            hovermode='x unified',
            showlegend=True
        )

        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.show()

    def plot_model_comparison(self, comparison_results, metric='accuracy', title="Model Comparison"):
        """
        Compare multiple models

        Args:
            comparison_results: Dictionary of model results
            metric: Metric to compare
            title: Chart title
        """
        models = list(comparison_results.keys())
        values = [result['metrics'][metric] for result in comparison_results.values()]

        fig = go.Figure(
            go.Bar(
                x=models,
                y=values,
                marker=dict(
                    color=values,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'{v:.4f}' for v in values],
                textposition='auto'
            )
        )

        fig.update_layout(
            title=f"{title} - {metric.upper()}",
            xaxis_title='Model',
            yaxis_title=metric.title(),
            height=500,
            template=self.theme
        )

        fig.show()

    def plot_correlation_heatmap(self, data, features=None, title="Feature Correlation Heatmap"):
        """
        Plot correlation heatmap

        Args:
            data: DataFrame with features
            features: List of features to include (if None, use all numeric)
            title: Chart title
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to reasonable number of features
        if len(features) > 30:
            print(f"Too many features ({len(features)}), showing top 30 by variance")
            variances = data[features].var().sort_values(ascending=False)
            features = variances.head(30).index.tolist()

        corr_matrix = data[features].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 8},
                showscale=True
            )
        )

        fig.update_layout(
            title=title,
            height=800,
            width=800,
            template=self.theme
        )

        fig.show()

    def create_dashboard(self, data, predictions=None, model_results=None,
                        backtest_results=None, title="Stock Analysis Dashboard"):
        """
        Create comprehensive dashboard

        Args:
            data: Stock data DataFrame
            predictions: Model predictions
            model_results: Model evaluation results
            backtest_results: Backtesting results
            title: Dashboard title
        """
        print(f"Creating {title}...")

        # 1. Candlestick with predictions
        print("Plotting price chart...")
        self.plot_candlestick(data, predictions, title=f"{title} - Price Analysis")

        # 2. Correlation heatmap
        if data.shape[1] > 5:
            print("Plotting correlation heatmap...")
            self.plot_correlation_heatmap(data, title=f"{title} - Feature Correlations")

        # 3. Model comparison if available
        if model_results:
            print("Plotting model comparison...")
            metric = 'accuracy' if 'accuracy' in list(model_results.values())[0]['metrics'] else 'r2'
            self.plot_model_comparison(model_results, metric=metric, title=f"{title} - Model Performance")

        # 4. Backtest results if available
        if backtest_results:
            print("Plotting backtest results...")
            self.plot_backtest_results(
                backtest_results['portfolio_values'],
                backtest_results.get('trades'),
                title=f"{title} - Backtest"
            )

        print("Dashboard creation complete!")


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
    data = collector.fetch_stock_data('AAPL', period='6mo')

    if data is not None:
        # Feature engineering
        print("Processing features...")
        fe = FeatureEngineering()
        processed_data = fe.prepare_features(data, target_days=1, target_type='direction')

        # Train model
        print("Training model...")
        predictor = StockPredictor()
        X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data)
        predictor.train_model(X_train, y_train, 'random_forest', 'classification')

        # Make predictions
        predictions = predictor.predict(processed_data[predictor.feature_columns])

        # Create visualizations
        viz = StockVisualization()

        # Plot candlestick with predictions
        viz.plot_candlestick(data.tail(100), predictions[-100:], title="AAPL Stock Analysis")

        # Plot feature importance
        feature_importance = predictor.get_feature_importance(top_n=20)
        if feature_importance is not None:
            viz.plot_feature_importance(feature_importance, title="Top 20 Important Features")
