import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  BarChart3,
  Settings,
  Activity,
  DollarSign,
  Target
} from 'lucide-react';
import { AnimatedCard, MetricCard } from './AnimatedCard';
import { AnimatedButton } from './AnimatedButton';
import { AnimatedInput, AnimatedSelect } from './AnimatedInput';
import StockPrediction from './StockPrediction';
import ModelComparison from './ModelComparison';
import BacktestSimulator from './BacktestSimulator';
import Marquee from './Marquee';
import Dock from './Dock';
import { stockTickerService, DEFAULT_TICKER_SYMBOLS } from '../services/stockTicker';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('predictions');
  const [symbol, setSymbol] = useState('AAPL');
  const [period, setPeriod] = useState('2y');
  const [modelType, setModelType] = useState('xgboost');
  const [targetType, setTargetType] = useState('direction');
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [tickerData, setTickerData] = useState([]);
  const [tickerLoading, setTickerLoading] = useState(true);

  const tabs = [
    { id: 'predictions', label: 'Predictions', icon: TrendingUp },
    { id: 'comparison', label: 'Model Comparison', icon: BarChart3 },
    { id: 'backtest', label: 'Backtesting', icon: DollarSign },
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

  // Initial fetch and auto-refresh every 30 seconds
  useEffect(() => {
    fetchTickerData();
    const interval = setInterval(fetchTickerData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

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
        className="mb-4 bg-slate-900 text-white py-3 px-4 rounded-lg shadow-lg"
      >
        {tickerLoading ? (
          <div className="text-center text-slate-400 text-sm py-1">
            Loading live market data...
          </div>
        ) : tickerData.length > 0 ? (
          <Marquee speed={30} pauseOnHover={true}>
            <div className="flex gap-8 items-center text-sm font-semibold">
              {tickerData.map((stock, index) => (
                <span key={index} className="flex items-center gap-2">
                  <span className="text-slate-400">{stock.symbol}</span>
                  <span className={stock.colorClass}>${stock.price}</span>
                  <span className={`${stock.colorClass} text-xs`}>
                    {stock.arrow} {stock.percentChange}%
                  </span>
                </span>
              ))}
            </div>
          </Marquee>
        ) : (
          <div className="text-center text-slate-400 text-sm py-1">
            Market data unavailable
          </div>
        )}
      </motion.div>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between mb-2">
          <div>
            <h1 className="text-5xl font-bold text-slate-900 mb-2">
              Stock Market Prediction AI
            </h1>
            <p className="text-slate-600 text-lg">
              Powered by Machine Learning & Deep Learning
            </p>
          </div>
          <div className="animate-float">
            <Activity className="w-16 h-16 text-blue-600" />
          </div>
        </div>
      </motion.div>

      {/* Configuration Panel */}
      <AnimatedCard delay={0.1} className="mb-8">
        <div className="flex items-center gap-2 mb-6">
          <Settings className="w-6 h-6 text-blue-600" />
          <h2 className="text-2xl font-bold text-slate-800">Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <AnimatedInput
            label="Stock Symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="AAPL"
            icon={Target}
          />

          <AnimatedSelect
            label="Quick Select"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            options={popularStocks}
          />

          <AnimatedSelect
            label="Time Period"
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            options={periods}
          />

          <AnimatedSelect
            label="Model Type"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            options={models}
          />
        </div>

        <div className="mt-6 flex items-center gap-4">
          <AnimatedSelect
            label="Prediction Type"
            value={targetType}
            onChange={(e) => setTargetType(e.target.value)}
            options={targets}
            className="flex-1"
          />
        </div>
      </AnimatedCard>

      {/* Dock Navigation */}
      <Dock items={dockItems} magnification={80} distance={150} />

      {/* Tab Content */}
      <div className="animate-fade-in">
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
      </div>

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
