import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import StockDataCollector
from feature_engineering import FeatureEngineering
from advanced_features import AdvancedFeatureEngineering
from ml_models import StockPredictor, compare_models
from deep_learning_models import DeepLearningPredictor
from backtesting import Backtester
from visualization import StockVisualization

# Page configuration
st.set_page_config(
    page_title="Stock Market Prediction AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .success-box {
        background-color: #dcfce7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #22c55e;
    }
    .warning-box {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stocks.png", width=80)
    st.title("🎯 Configuration")

    # Stock Selection
    st.subheader("📊 Stock Selection")
    stock_symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, TSLA, MSFT)")

    popular_stocks = st.selectbox(
        "Or choose popular stock:",
        ["Custom", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"]
    )

    if popular_stocks != "Custom":
        stock_symbol = popular_stocks

    # Time Period
    st.subheader("📅 Time Period")
    period = st.selectbox(
        "Historical Data Period",
        ["6mo", "1y", "2y", "5y", "max"],
        index=2,
        help="More data = better patterns, but slower"
    )

    # Model Configuration
    st.subheader("🤖 Model Settings")
    model_type = st.selectbox(
        "Select Model",
        ["xgboost", "lightgbm", "random_forest", "ensemble", "logistic_regression"],
        help="XGBoost and LightGBM usually perform best"
    )

    # Prediction Settings
    st.subheader("🎯 Prediction Settings")
    target_type = st.selectbox(
        "Prediction Type",
        ["direction", "price_change", "price"],
        help="Direction = UP/DOWN, Price Change = %, Price = Actual value"
    )

    target_days = st.slider(
        "Days Ahead",
        min_value=1,
        max_value=10,
        value=1,
        help="Predict 1-10 days in the future"
    )

    # Advanced Features
    st.subheader("⚙️ Advanced Options")
    use_advanced_features = st.checkbox("Use Advanced Features", value=False, help="Candlestick patterns, market regimes")
    compare_all_models = st.checkbox("Compare All Models", value=False, help="Train and compare multiple models")
    run_backtest = st.checkbox("Run Backtesting", value=False, help="Simulate trading strategy")

    # Run Button
    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", type="primary")

# Main Content
st.markdown('<h1 class="main-header">📈 Stock Market Prediction AI</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Machine Learning & Deep Learning")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Predictions",
    "📈 Visualizations",
    "🔍 Model Comparison",
    "💰 Backtesting",
    "📚 About"
])

# Run Analysis
if run_button:
    with st.spinner(f'🔄 Analyzing {stock_symbol}...'):
        try:
            # Step 1: Collect Data
            st.info(f"📥 Collecting {stock_symbol} data for period: {period}")
            collector = StockDataCollector()
            data = collector.fetch_stock_data(stock_symbol, period=period)

            if data is None or len(data) == 0:
                st.error(f"❌ Could not fetch data for {stock_symbol}. Please check the symbol and try again.")
                st.stop()

            st.session_state.data = data
            st.success(f"✅ Collected {len(data)} trading days ({data.index.min().date()} to {data.index.max().date()})")

            # Step 2: Feature Engineering
            st.info("🔧 Engineering features...")
            fe = FeatureEngineering()
            processed_data = fe.prepare_features(data, target_days=target_days, target_type=target_type)

            if use_advanced_features:
                st.info("🚀 Adding advanced features (candlestick patterns, market regimes)...")
                advanced_fe = AdvancedFeatureEngineering()
                advanced_data = advanced_fe.prepare_advanced_features(data)
                # Merge advanced features
                for col in advanced_data.columns:
                    if col not in processed_data.columns and col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        processed_data[col] = advanced_data[col]
                processed_data = processed_data.dropna()

            st.session_state.processed_data = processed_data
            st.success(f"✅ Created {processed_data.shape[1]-1} features from {processed_data.shape[0]} samples")

            # Step 3: Train Model(s)
            if compare_all_models:
                st.info("🤖 Training and comparing all models...")
                predictor = StockPredictor()
                X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data, time_series_split=True)

                task_type = 'classification' if target_type == 'direction' else 'regression'
                results = compare_models(X_train, X_test, y_train, y_test, task_type=task_type,
                                       feature_columns=predictor.feature_columns, scaler=predictor.scaler)

                st.session_state.model_results = results

                # Get best model
                comparison_df = pd.DataFrame({model: result['metrics'] for model, result in results.items()}).T
                if task_type == 'classification':
                    best_model_name = comparison_df['accuracy'].idxmax()
                else:
                    best_model_name = comparison_df['r2'].idxmax()

                st.session_state.model = results[best_model_name]['model']
                st.success(f"✅ Best model: {best_model_name}")

            else:
                st.info(f"🤖 Training {model_type} model...")
                predictor = StockPredictor()
                X_train, X_test, y_train, y_test = predictor.prepare_data(processed_data, time_series_split=True)

                task_type = 'classification' if target_type == 'direction' else 'regression'
                predictor.train_model(X_train, y_train, model_type, task_type)

                st.session_state.model = predictor
                st.success(f"✅ Model trained successfully!")

            # Step 4: Make Predictions
            st.info("🔮 Making predictions...")
            model = st.session_state.model
            predictions = model.predict(processed_data[model.feature_columns])
            st.session_state.predictions = predictions

            st.success("🎉 Analysis Complete!")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

# TAB 1: Predictions
with tab1:
    if st.session_state.data is not None and st.session_state.predictions is not None:
        st.header("📊 Prediction Results")

        # Current prediction
        col1, col2, col3 = st.columns(3)

        latest_prediction = st.session_state.predictions[-1]
        data = st.session_state.data
        latest_close = data['Close'].iloc[-1]

        with col1:
            st.metric(
                label="📍 Current Price",
                value=f"${latest_close:.2f}",
                delta=f"{data['Close'].pct_change().iloc[-1]*100:.2f}%"
            )

        with col2:
            if target_type == 'direction':
                pred_text = "📈 UP" if latest_prediction == 1 else "📉 DOWN"
                st.metric(label="🔮 Prediction (Next Day)", value=pred_text)
            else:
                st.metric(
                    label="🔮 Predicted Change",
                    value=f"{latest_prediction:.2f}%"
                )

        with col3:
            if st.session_state.processed_data is not None:
                target_dist = st.session_state.processed_data['Target'].mean()
                if target_type == 'direction':
                    st.metric(
                        label="📊 Historical Up Days",
                        value=f"{target_dist*100:.1f}%"
                    )

        st.markdown("---")

        # Recent predictions table
        st.subheader("📋 Recent Predictions (Last 20 Days)")

        recent_data = st.session_state.processed_data.tail(20)
        recent_preds = st.session_state.predictions[-20:]

        results_df = pd.DataFrame({
            'Date': recent_data.index.strftime('%Y-%m-%d'),
            'Close': recent_data['Close'].values,
            'Actual': recent_data['Target'].values,
            'Predicted': recent_preds
        })

        if target_type == 'direction':
            results_df['Actual'] = results_df['Actual'].map({1: '📈 UP', 0: '📉 DOWN'})
            results_df['Predicted'] = results_df['Predicted'].map({1: '📈 UP', 0: '📉 DOWN'})
            results_df['Correct'] = ['✅' if a == p else '❌' for a, p in zip(
                recent_data['Target'].values, recent_preds
            )]

            accuracy = (recent_data['Target'].values == recent_preds).mean()
            st.info(f"🎯 Recent Accuracy: **{accuracy*100:.1f}%**")

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Model Performance
        if st.session_state.model is not None:
            st.markdown("---")
            st.subheader("📊 Model Performance Metrics")

            model = st.session_state.model
            processed_data = st.session_state.processed_data

            # Prepare test set
            X_train, X_test, y_train, y_test = model.prepare_data(processed_data, time_series_split=True)
            metrics = model.evaluate_model(X_test, y_test)

            if target_type == 'direction':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%")
                with col2:
                    st.metric("🎪 Precision", f"{metrics['precision']*100:.2f}%")
                with col3:
                    st.metric("📡 Recall", f"{metrics['recall']*100:.2f}%")
                with col4:
                    st.metric("⚖️ F1-Score", f"{metrics['f1_score']*100:.2f}%")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📉 MSE", f"{metrics['mse']:.4f}")
                with col2:
                    st.metric("📊 RMSE", f"{metrics['rmse']:.4f}")
                with col3:
                    st.metric("📏 MAE", f"{metrics['mae']:.4f}")
                with col4:
                    st.metric("📈 R²", f"{metrics['r2']:.4f}")

    else:
        st.info("👈 Configure settings in the sidebar and click '🚀 Run Analysis' to see predictions!")
        st.markdown("""
        ### Quick Start Guide:
        1. Enter a stock symbol (e.g., AAPL, TSLA, MSFT)
        2. Choose your preferred model (XGBoost recommended)
        3. Select prediction settings
        4. Click **Run Analysis**

        ### Model Recommendations:
        - **XGBoost**: Best overall accuracy (55-60%)
        - **LightGBM**: Fast training, good accuracy (54-58%)
        - **Ensemble**: Combines multiple models (56-59%)
        """)

# TAB 2: Visualizations
with tab2:
    if st.session_state.data is not None:
        st.header("📈 Interactive Visualizations")

        data = st.session_state.data
        predictions = st.session_state.predictions

        # Candlestick chart
        st.subheader("🕯️ Candlestick Chart with Predictions")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price', 'Volume', 'RSI')
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add moving averages if available
        processed_data = st.session_state.processed_data
        if processed_data is not None:
            if 'SMA_20' in processed_data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=processed_data['SMA_20'],
                              name='SMA 20', line=dict(color='orange', width=1)),
                    row=1, col=1
                )
            if 'SMA_50' in processed_data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=processed_data['SMA_50'],
                              name='SMA 50', line=dict(color='blue', width=1)),
                    row=1, col=1
                )

        # Buy/Sell signals
        if predictions is not None and target_type == 'direction':
            aligned_data = data.iloc[-len(predictions):]
            buy_signals = aligned_data.index[predictions == 1]
            sell_signals = aligned_data.index[predictions == 0]

            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals,
                        y=aligned_data.loc[buy_signals, 'Low'] * 0.995,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(symbol='triangle-up', size=10, color='green')
                    ),
                    row=1, col=1
                )

            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals,
                        y=aligned_data.loc[sell_signals, 'High'] * 1.005,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(symbol='triangle-down', size=10, color='red')
                    ),
                    row=1, col=1
                )

        # Volume
        colors = ['red' if close < open else 'green'
                 for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )

        # RSI if available
        if processed_data is not None and 'RSI' in processed_data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=processed_data['RSI'], name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        if st.session_state.model is not None:
            st.markdown("---")
            st.subheader("🎯 Feature Importance")

            feature_importance = st.session_state.model.get_feature_importance(top_n=20)

            if feature_importance is not None:
                fig = go.Figure(
                    go.Bar(
                        x=feature_importance['importance'],
                        y=feature_importance['feature'],
                        orientation='h',
                        marker=dict(color=feature_importance['importance'], colorscale='Viridis')
                    )
                )
                fig.update_layout(
                    title='Top 20 Most Important Features',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type")

    else:
        st.info("👈 Run analysis first to see visualizations!")

# TAB 3: Model Comparison
with tab3:
    if st.session_state.model_results is not None:
        st.header("🔍 Model Comparison Results")

        results = st.session_state.model_results
        comparison_df = pd.DataFrame({model: result['metrics'] for model, result in results.items()}).T

        # Display table
        st.subheader("📊 Performance Metrics")
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        # Best model
        if target_type == 'direction':
            best_model = comparison_df['accuracy'].idxmax()
            best_score = comparison_df.loc[best_model, 'accuracy']
            metric_name = 'Accuracy'
        else:
            best_model = comparison_df['r2'].idxmax()
            best_score = comparison_df.loc[best_model, 'r2']
            metric_name = 'R²'

        st.success(f"🏆 **Best Model:** {best_model} ({metric_name}: {best_score:.4f})")

        # Comparison chart
        st.subheader("📊 Visual Comparison")

        metric_to_plot = 'accuracy' if target_type == 'direction' else 'r2'

        fig = go.Figure(
            go.Bar(
                x=comparison_df.index,
                y=comparison_df[metric_to_plot],
                marker=dict(
                    color=comparison_df[metric_to_plot],
                    colorscale='Blues',
                    showscale=True
                ),
                text=[f'{v:.2%}' if target_type == 'direction' else f'{v:.4f}'
                      for v in comparison_df[metric_to_plot]],
                textposition='auto'
            )
        )
        fig.update_layout(
            title=f'Model Comparison - {metric_name}',
            xaxis_title='Model',
            yaxis_title=metric_name,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Enable 'Compare All Models' in the sidebar and run analysis to see model comparison!")

# TAB 4: Backtesting
with tab4:
    if run_backtest and st.session_state.data is not None and st.session_state.predictions is not None:
        st.header("💰 Backtesting Results")

        st.info("🔄 Running backtest simulation...")

        # Backtesting parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
        with col2:
            stop_loss = st.slider("Stop Loss (%)", 0.0, 20.0, 5.0) / 100
        with col3:
            take_profit = st.slider("Take Profit (%)", 0.0, 50.0, 10.0) / 100

        commission = 0.001  # 0.1%
        slippage = 0.001    # 0.1%

        # Run backtest
        backtester = Backtester(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )

        data = st.session_state.data
        predictions = st.session_state.predictions

        # Align data
        test_data = data.iloc[-len(predictions):]

        results = backtester.run_backtest(
            test_data,
            predictions,
            prediction_type=target_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=0.95
        )

        # Display results
        st.subheader("📈 Performance Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💵 Final Value",
                f"${results['final_value']:,.2f}",
                f"{results['total_return_pct']:.2f}%"
            )

        with col2:
            if 'num_trades' in results:
                st.metric("🔄 Total Trades", results['num_trades'])

        with col3:
            if 'win_rate' in results:
                st.metric("🎯 Win Rate", f"{results['win_rate']*100:.1f}%")

        with col4:
            if 'sharpe_ratio' in results:
                st.metric("📊 Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

        # Portfolio value chart
        if backtester.portfolio_values is not None and len(backtester.portfolio_values) > 0:
            st.subheader("💼 Portfolio Value Over Time")

            portfolio_df = backtester.portfolio_values

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['value'],
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            # Initial capital line
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital",
                row=1, col=1
            )

            # Drawdown
            portfolio_vals = portfolio_df['value'].values
            cummax = np.maximum.accumulate(portfolio_vals)
            drawdown = (portfolio_vals - cummax) / cummax * 100

            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=drawdown,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )

            fig.update_layout(height=600, showlegend=True)
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

        # Trade history
        if backtester.trades is not None and len(backtester.trades) > 0:
            st.subheader("📋 Trade History")

            trades_df = backtester.trades.copy()
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
            trades_df['profit'] = trades_df['profit'].round(2)
            trades_df['return'] = (trades_df['return'] * 100).round(2)

            st.dataframe(trades_df, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Enable 'Run Backtesting' in the sidebar and run analysis to see backtest results!")
        st.markdown("""
        ### Backtesting Features:
        - **Realistic simulation** with commission and slippage
        - **Risk management** with stop-loss and take-profit
        - **Performance metrics**: Sharpe ratio, max drawdown, win rate
        - **Trade-by-trade analysis**
        - **Portfolio visualization**
        """)

# TAB 5: About
with tab5:
    st.header("📚 About This Application")

    st.markdown("""
    ## 🤖 Stock Market Prediction AI

    This application uses advanced machine learning and deep learning techniques to predict stock price movements.

    ### 🎯 Features

    #### Machine Learning Models
    - **XGBoost**: Gradient boosting framework (55-60% accuracy)
    - **LightGBM**: Fast gradient boosting (54-58% accuracy)
    - **Random Forest**: Ensemble of decision trees (50-55% accuracy)
    - **Ensemble**: Combines multiple models (56-59% accuracy)

    #### Feature Engineering
    - **100+ technical indicators**
    - **Candlestick patterns**: Doji, Hammer, Engulfing, etc.
    - **Market regime detection**: Trending vs ranging markets
    - **Advanced momentum**: RSI, MACD, Stochastic, CCI, MFI
    - **Volatility indicators**: Bollinger Bands, ATR, Keltner Channels
    - **Volume analysis**: OBV, Chaikin Money Flow, A/D Line

    #### Backtesting
    - Realistic trading simulation
    - Commission and slippage modeling
    - Risk management (stop-loss, take-profit)
    - Comprehensive performance metrics

    ### ⚠️ Important Disclaimers

    1. **Not Financial Advice**: This tool is for educational purposes only
    2. **No Guarantees**: Past performance does not guarantee future results
    3. **Use at Your Own Risk**: Always do your own research
    4. **Market Volatility**: Stock markets are inherently unpredictable
    5. **Paper Trade First**: Test strategies before using real money

    ### 📊 Performance Expectations

    - **Good Accuracy**: 55-60% for daily direction prediction
    - **Excellent Accuracy**: 60%+ (rare)
    - **Random Baseline**: 50% (coin flip)

    ### 🛠️ Technology Stack

    - **Frontend**: Streamlit
    - **ML**: XGBoost, LightGBM, Scikit-learn
    - **Deep Learning**: TensorFlow, Keras
    - **Visualization**: Plotly
    - **Data**: yfinance, pandas, numpy

    ### 📖 How to Use

    1. **Select a stock** symbol in the sidebar
    2. **Choose your model** (XGBoost recommended)
    3. **Configure settings** (period, prediction type, etc.)
    4. **Click Run Analysis** to see results
    5. **Explore different tabs** for visualizations and analysis

    ### 💡 Tips for Best Results

    - Use **2-5 years** of data for better patterns
    - **XGBoost and LightGBM** typically perform best
    - Enable **Compare All Models** to find the best model for each stock
    - Always use **backtesting** to verify strategy performance
    - Set appropriate **stop-losses** to manage risk

    ### 📞 Support

    For questions or issues, please check the project documentation.

    ---

    **Remember**: Always invest responsibly and never risk more than you can afford to lose! 💰
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p><strong>Stock Market Prediction AI</strong> | Built with ❤️ using Streamlit & Machine Learning</p>
    <p>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
