import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Activity, BarChart3 } from 'lucide-react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area
} from 'recharts';
import './AdvancedCandlestickChart.css';

const AdvancedCandlestickChart = ({ symbol = 'AAPL', period = '6mo' }) => {
  const [loading, setLoading] = useState(true);
  const [chartData, setChartData] = useState([]);
  const [indicators, setIndicators] = useState({
    sma20: true,
    sma50: true,
    volume: true,
    bollinger: false
  });

  // Fetch stock data
  useEffect(() => {
    const fetchStockData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`http://localhost:5000/api/stock-data?symbol=${symbol}&period=${period}`);

        if (!response.ok) {
          generateDemoData();
          return;
        }

        const data = await response.json();

        // Transform data for Recharts
        const transformed = data.candles.map((candle, index) => {
          const date = new Date(candle.time * 1000);
          return {
            date: date.toLocaleDateString(),
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: data.volume[index]?.value || 0,
            range: [candle.low, candle.high],
            color: candle.close > candle.open ? '#22c55e' : '#ef4444'
          };
        });

        // Calculate indicators
        const withIndicators = calculateIndicators(transformed);
        setChartData(withIndicators);
      } catch (error) {
        console.error('Error fetching stock data:', error);
        generateDemoData();
      } finally {
        setLoading(false);
      }
    };

    fetchStockData();
  }, [symbol, period]);

  // Generate demo data
  const generateDemoData = () => {
    const data = [];
    let basePrice = 150;
    const daysAgo = period === '1mo' ? 30 : period === '6mo' ? 180 : 365;

    for (let i = daysAgo; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);

      const open = basePrice + (Math.random() - 0.5) * 10;
      const close = open + (Math.random() - 0.5) * 8;
      const high = Math.max(open, close) + Math.random() * 5;
      const low = Math.min(open, close) - Math.random() * 5;

      data.push({
        date: date.toLocaleDateString(),
        open,
        high,
        low,
        close,
        volume: Math.random() * 10000000 + 5000000,
        range: [low, high],
        color: close > open ? '#22c55e' : '#ef4444'
      });

      basePrice = close;
    }

    const withIndicators = calculateIndicators(data);
    setChartData(withIndicators);
  };

  // Calculate technical indicators
  const calculateIndicators = (data) => {
    if (data.length === 0) return data;

    // Calculate SMA
    const calculateSMA = (period) => {
      return data.map((item, index) => {
        if (index < period - 1) return null;
        const sum = data.slice(index - period + 1, index + 1)
          .reduce((acc, val) => acc + val.close, 0);
        return sum / period;
      });
    };

    // Calculate Bollinger Bands
    const calculateBollinger = (period = 20, stdDev = 2) => {
      const sma = calculateSMA(period);
      const upper = [];
      const lower = [];

      data.forEach((item, index) => {
        if (index < period - 1) {
          upper.push(null);
          lower.push(null);
        } else {
          const slice = data.slice(index - period + 1, index + 1);
          const mean = sma[index];
          const variance = slice.reduce((acc, val) =>
            acc + Math.pow(val.close - mean, 2), 0) / period;
          const std = Math.sqrt(variance);

          upper.push(mean + (stdDev * std));
          lower.push(mean - (stdDev * std));
        }
      });

      return { upper, lower };
    };

    const sma20 = calculateSMA(20);
    const sma50 = calculateSMA(50);
    const bollinger = calculateBollinger();

    return data.map((item, index) => ({
      ...item,
      sma20: sma20[index],
      sma50: sma50[index],
      bollingerUpper: bollinger.upper[index],
      bollingerLower: bollinger.lower[index]
    }));
  };

  const toggleIndicator = (indicator) => {
    setIndicators(prev => ({ ...prev, [indicator]: !prev[indicator] }));
  };

  // Custom candlestick shape
  const CustomCandlestick = (props) => {
    const { x, y, width, height, low, high, open, close } = props;
    const isUp = close > open;
    const color = isUp ? '#22c55e' : '#ef4444';
    const candleWidth = Math.max(width * 0.6, 2);
    const wickX = x + width / 2;

    return (
      <g>
        {/* Wick */}
        <line
          x1={wickX}
          y1={y}
          x2={wickX}
          y2={y + height}
          stroke={color}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x + (width - candleWidth) / 2}
          y={isUp ? y + height * ((high - close) / (high - low)) : y + height * ((high - open) / (high - low))}
          width={candleWidth}
          height={Math.abs(height * ((close - open) / (high - low)))}
          fill={color}
          stroke={color}
        />
      </g>
    );
  };

  if (loading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="advanced-chart-container loading"
      >
        <div className="loading-spinner">
          <Activity className="w-12 h-12 text-indigo-600 animate-spin" />
          <p className="text-slate-600 mt-4">Loading chart data...</p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="advanced-chart-container"
    >
      {/* Chart Header */}
      <div className="chart-header">
        <div className="chart-title">
          <TrendingUp className="w-6 h-6 text-indigo-600" />
          <h3 className="text-xl font-bold text-slate-800">{symbol} - Advanced Chart</h3>
        </div>

        {/* Indicator Controls */}
        <div className="indicator-controls">
          <button
            onClick={() => toggleIndicator('sma20')}
            className={`indicator-btn ${indicators.sma20 ? 'active' : ''}`}
          >
            <span className="indicator-color bg-blue-500"></span>
            SMA 20
          </button>
          <button
            onClick={() => toggleIndicator('sma50')}
            className={`indicator-btn ${indicators.sma50 ? 'active' : ''}`}
          >
            <span className="indicator-color bg-purple-500"></span>
            SMA 50
          </button>
          <button
            onClick={() => toggleIndicator('bollinger')}
            className={`indicator-btn ${indicators.bollinger ? 'active' : ''}`}
          >
            <span className="indicator-color bg-amber-500"></span>
            Bollinger
          </button>
          <button
            onClick={() => toggleIndicator('volume')}
            className={`indicator-btn ${indicators.volume ? 'active' : ''}`}
          >
            <BarChart3 className="w-4 h-4" />
            Volume
          </button>
        </div>
      </div>

      {/* Price Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12, fill: '#64748b' }}
            tickFormatter={(value) => {
              const parts = value.split('/');
              return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : value;
            }}
          />
          <YAxis
            domain={['auto', 'auto']}
            tick={{ fontSize: 12, fill: '#64748b' }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '2px solid #e2e8f0',
              borderRadius: '0.5rem',
              padding: '0.75rem'
            }}
            formatter={(value) => `$${Number(value).toFixed(2)}`}
          />
          <Legend />

          {/* Bollinger Bands */}
          {indicators.bollinger && (
            <>
              <Area
                type="monotone"
                dataKey="bollingerUpper"
                stroke="#f59e0b"
                fill="none"
                strokeWidth={1}
                strokeDasharray="3 3"
                name="BB Upper"
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="bollingerLower"
                stroke="#f59e0b"
                fill="none"
                strokeWidth={1}
                strokeDasharray="3 3"
                name="BB Lower"
                dot={false}
              />
            </>
          )}

          {/* Moving Averages */}
          {indicators.sma20 && (
            <Line
              type="monotone"
              dataKey="sma20"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="SMA 20"
            />
          )}
          {indicators.sma50 && (
            <Line
              type="monotone"
              dataKey="sma50"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              name="SMA 50"
            />
          )}

          {/* Close Price Line */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#6366f1"
            strokeWidth={2}
            dot={false}
            name="Close Price"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Volume Chart */}
      {indicators.volume && (
        <ResponsiveContainer width="100%" height={150}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12, fill: '#64748b' }}
              tickFormatter={(value) => {
                const parts = value.split('/');
                return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : value;
              }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#64748b' }}
              tickFormatter={(value) => `${(value / 1000000).toFixed(0)}M`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '2px solid #e2e8f0',
                borderRadius: '0.5rem',
                padding: '0.75rem'
              }}
              formatter={(value) => `${(value / 1000000).toFixed(2)}M`}
            />
            <Bar dataKey="volume" fill="#6366f1" opacity={0.6} name="Volume" />
          </ComposedChart>
        </ResponsiveContainer>
      )}

      {/* Chart Info */}
      <div className="chart-info">
        <div className="info-item">
          <span className="info-label">Period:</span>
          <span className="info-value">{period}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Data Points:</span>
          <span className="info-value">{chartData.length}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Indicators:</span>
          <span className="info-value">{Object.values(indicators).filter(Boolean).length} active</span>
        </div>
      </div>
    </motion.div>
  );
};

export default AdvancedCandlestickChart;
