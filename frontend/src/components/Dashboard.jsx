import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  BarChart3,
  Settings,
  Activity,
  DollarSign,
  Target,
  Share2
} from 'lucide-react';
import { AnimatedCard, MetricCard } from './AnimatedCard';
import { AnimatedButton } from './AnimatedButton';
import { AnimatedInput, AnimatedSelect } from './AnimatedInput';
import StockPrediction from './StockPrediction';
import ModelComparison from './ModelComparison';
import BacktestSimulator from './BacktestSimulator';
import SentimentDashboard from './SentimentDashboard';
import PortfolioDashboard from './PortfolioDashboard';
import Marquee from './Marquee';
import Dock from './Dock';
import { stockTickerService, DEFAULT_TICKER_SYMBOLS } from '../services/stockTicker';
import { stockAPI, socket } from '../services/api';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('predictions');
  const [symbol, setSymbol] = useState('AAPL');
  const [period, setPeriod] = useState('2y');
  const [modelType, setModelType] = useState('xgboost');
  const [targetType, setTargetType] = useState('direction');
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [tickerData, setTickerData] = useState([]);
  const [tickerLoading, setTickerLoading] = useState(true);

  const tabs = [
    { id: 'predictions', label: 'Predictions', icon: TrendingUp },
    { id: 'sentiment', label: 'Sentiment', icon: Share2 },
    { id: 'comparison', label: 'Model Comparison', icon: BarChart3 },
    { id: 'backtest', label: 'Backtesting', icon: DollarSign },
    { id: 'portfolio', label: 'Portfolio', icon: Target },
  ];

  const popularStocks = [
    { value: 'AAPL', label: 'Apple (AAPL)' },
    { value: 'TSLA', label: 'Tesla (TSLA)' },
    { value: 'MSFT', label: 'Microsoft (MSFT)' },
    { value: 'GOOGL', label: 'Google (GOOGL)' },
    { value: 'AMZN', label: 'Amazon (AMZN)' },
    { value: 'NVDA', label: 'NVIDIA (NVDA)' },
    { value: 'META', label: 'Meta (META)' },
  ];

  const periods = [
    { value: '5d', label: '1 Week' },
    { value: '1mo', label: '1 Month' },
    { value: '3mo', label: '3 Months' },
    { value: '6mo', label: '6 Months' },
    { value: '1y', label: '1 Year' },
    { value: '2y', label: '2 Years' },
    { value: '5y', label: '5 Years' },
    { value: 'max', label: 'Maximum' },
  ];

  const models = [
    { value: 'xgboost', label: 'XGBoost (Best)' },
    { value: 'lightgbm', label: 'LightGBM (Fast)' },
    { value: 'random_forest', label: 'Random Forest' },
    { value: 'ensemble', label: 'Ensemble (Combo)' },
    { value: 'lstm', label: 'LSTM (Deep Learning)' },
    { value: 'logistic_regression', label: 'Logistic Regression' },
  ];

  const targets = [
    { value: 'direction', label: 'Direction (Up/Down)' },
    { value: 'price_change', label: 'Price Change (%)' },
    { value: 'price', label: 'Actual Price' },
  ];

  // Fallback static data if API fails
  const fallbackTickerData = [
    { symbol: 'AAPL', price: '178.50', percentChange: '2.3', arrow: '↑', colorClass: 'text-green-400', isPositive: true },
    { symbol: 'TSLA', price: '242.30', percentChange: '1.8', arrow: '↑', colorClass: 'text-green-400', isPositive: true },
    { symbol: 'MSFT', price: '405.20', percentChange: '0.5', arrow: '↓', colorClass: 'text-red-400', isPositive: false },
    { symbol: 'NVDA', price: '495.80', percentChange: '3.2', arrow: '↑', colorClass: 'text-green-400', isPositive: true },
    { symbol: 'GOOGL', price: '142.65', percentChange: '0.9', arrow: '↑', colorClass: 'text-green-400', isPositive: true },
    { symbol: 'AMZN', price: '178.25', percentChange: '1.4', arrow: '↑', colorClass: 'text-green-400', isPositive: true },
    { symbol: 'META', price: '485.90', percentChange: '1.1', arrow: '↓', colorClass: 'text-red-400', isPositive: false }
  ];

  // Fetch live stock ticker data
  const fetchTickerData = async () => {
    try {
      console.log('Fetching ticker data...');
      const quotes = await stockTickerService.getMultipleQuotes(DEFAULT_TICKER_SYMBOLS);
      console.log('Quotes received:', quotes);

      const formattedData = quotes
        .map(quote => stockTickerService.formatForMarquee(quote))
        .filter(data => data !== null);

      console.log('Formatted data:', formattedData);

      if (formattedData.length > 0) {
        setTickerData(formattedData);
      } else {
        console.warn('No data received, using fallback');
        setTickerData(fallbackTickerData);
      }
      setTickerLoading(false);
    } catch (error) {
      console.error('Error fetching ticker data:', error);
      console.log('Using fallback data');
      setTickerData(fallbackTickerData);
      setTickerLoading(false);
    }
  };

  // Initial fetch and real-time updates
  useEffect(() => {
    fetchTickerData();

    // Listen for real-time updates
    socket.on('price_update', (data) => {
      setTickerData(prevData => {
        return prevData.map(item => {
          if (item.symbol === data.symbol) {
            const isPositive = data.change >= 0;
            return {
              ...item,
              price: data.price.toFixed(2),
              percentChange: data.change_pct.toFixed(2),
              arrow: isPositive ? '↑' : '↓',
              colorClass: isPositive ? 'text-emerald-400' : 'text-red-400',
              isPositive
            };
          }
          return item;
        });
      });
    });

    return () => {
      socket.off('price_update');
    };
  }, []);

  // Fetch sentiment when tab is active
  useEffect(() => {
    if (activeTab === 'sentiment' && symbol) {
      const fetchSentiment = async () => {
        setSentimentLoading(true);
        try {
          const data = await stockAPI.getSentiment(symbol);
          setSentimentData(data);
        } catch (error) {
          console.error("Error fetching sentiment:", error);
        } finally {
          setSentimentLoading(false);
        }
      };
      fetchSentiment();
    }
  }, [activeTab, symbol]);

  const dockItems = tabs.map(tab => ({
    icon: <tab.icon className="w-6 h-6" />,
    label: tab.label,
    onClick: () => setActiveTab(tab.id),
    className: activeTab === tab.id ? 'active' : ''
  }));

  return (
    <div className="min-h-screen p-6 pb-32">
      {/* Stock Ticker Marquee - Live Data */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/10"
      >
        {tickerLoading ? (
          <div className="text-center text-slate-400 text-sm py-2">
            Loading live market data...
          </div>
        ) : tickerData.length > 0 ? (
          <Marquee speed={30} pauseOnHover={true}>
            <div className="flex gap-8 items-center text-sm font-semibold py-2">
              {tickerData.map((stock, index) => (
                <span key={index} className="flex items-center gap-2">
                  <span className="text-white">{stock.symbol}</span>
                  <span className={stock.colorClass}>${stock.price}</span>
                  <span className={`${stock.colorClass} text-xs`}>
                    {stock.arrow} {stock.percentChange}%
                  </span>
                </span>
              ))}
            </div>
          </Marquee>
        ) : (
          <div className="text-center text-slate-400 text-sm py-2">
            Market data unavailable
          </div>
        )}
      </motion.div>

      <div className="max-w-7xl mx-auto mt-20 space-y-8">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4 py-8"
        >
          <h1 className="text-6xl font-bold tracking-tight">
            <span className="text-gradient">StockPredict</span>
          </h1>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto font-light">
            Advanced market analysis powered by machine learning and sentiment analysis
          </p>
        </motion.header>

        {/* Configuration Panel */}
        <AnimatedCard delay={0.1} className="glass-panel">
          <div className="flex items-center gap-2 mb-6">
            <Settings className="w-6 h-6 text-blue-400" />
            <h2 className="text-2xl font-bold text-slate-100">Configuration</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Stock Symbol</label>
              <div className="relative">
                <Target className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  className="glass-input w-full pl-10"
                  placeholder="e.g., AAPL"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Time Period</label>
              <div className="relative">
                <Activity className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                <select
                  value={period}
                  onChange={(e) => setPeriod(e.target.value)}
                  className="glass-input w-full pl-10 appearance-none cursor-pointer"
                >
                  {periods.map(p => (
                    <option key={p.value} value={p.value} className="bg-slate-800">{p.label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Model Type</label>
              <div className="relative">
                <BarChart3 className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  className="glass-input w-full pl-10 appearance-none cursor-pointer"
                >
                  {models.map(m => (
                    <option key={m.value} value={m.value} className="bg-slate-800">{m.label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Prediction Target</label>
              <div className="relative">
                <TrendingUp className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                <select
                  value={targetType}
                  onChange={(e) => setTargetType(e.target.value)}
                  className="glass-input w-full pl-10 appearance-none cursor-pointer"
                >
                  {targets.map(t => (
                    <option key={t.value} value={t.value} className="bg-slate-800">{t.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </AnimatedCard>

        {/* Tab Content */}
        <div className="min-h-[600px]">
          {activeTab === 'predictions' && (
            <StockPrediction
              symbol={symbol}
              period={period}
              modelType={modelType}
              targetType={targetType}
              predictionData={predictionData}
              setPredictionData={setPredictionData}
            />
          )}

          {activeTab === 'sentiment' && (
            <SentimentDashboard
              sentimentData={sentimentData}
              loading={sentimentLoading}
            />
          )}

          {activeTab === 'comparison' && (
            <ModelComparison
              symbol={symbol}
              period={period}
              targetType={targetType}
            />
          )}

          {activeTab === 'backtest' && (
            <BacktestSimulator
              symbol={symbol}
              modelType={modelType}
              targetType={targetType}
              predictionData={predictionData}
            />
          )}

          {activeTab === 'portfolio' && (
            <PortfolioDashboard />
          )}
        </div>
      </div>

      {/* Dock Navigation */}
      <Dock items={dockItems} />

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="mt-12 text-center text-slate-500 pb-8"
      >
        <p className="font-semibold">Stock Market Prediction AI</p>
        <p className="text-sm mt-1">
          For educational purposes only. Not financial advice.
        </p>
      </motion.div>
    </div>
  );
};

export default Dashboard;
