import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { AnimatedCard } from './AnimatedCard';
import { stockAPI } from '../services/api';
import { motion } from 'framer-motion';

const CandlestickChart = ({ symbol, period }) => {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        setLoading(true);
        const result = await stockAPI.getChartData(symbol, period);
        setChartData(result);
      } catch (error) {
        console.error('Error fetching chart data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  }, [symbol, period]);

  if (loading) {
    return (
      <AnimatedCard>
        <div className="h-96 flex items-center justify-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full"
          />
        </div>
      </AnimatedCard>
    );
  }

  if (!chartData || !chartData.data) {
    return (
      <AnimatedCard>
        <div className="h-96 flex items-center justify-center text-slate-500">
          No chart data available
        </div>
      </AnimatedCard>
    );
  }

  const data = chartData.data;

  const candlestickTrace = {
    x: data.map((d) => d.date),
    open: data.map((d) => d.open),
    high: data.map((d) => d.high),
    low: data.map((d) => d.low),
    close: data.map((d) => d.close),
    type: 'candlestick',
    name: symbol,
    increasing: { line: { color: '#10b981' } },
    decreasing: { line: { color: '#ef4444' } },
  };

  const volumeTrace = {
    x: data.map((d) => d.date),
    y: data.map((d) => d.volume),
    type: 'bar',
    name: 'Volume',
    marker: {
      color: data.map((d) =>
        d.close >= d.open ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'
      ),
    },
    yaxis: 'y2',
  };

  const traces = [candlestickTrace, volumeTrace];

  // Add SMA lines if available
  if (data[0].sma_20) {
    traces.push({
      x: data.map((d) => d.date),
      y: data.map((d) => d.sma_20),
      type: 'scatter',
      mode: 'lines',
      name: 'SMA 20',
      line: { color: '#f59e0b', width: 2 },
    });
  }

  if (data[0].sma_50) {
    traces.push({
      x: data.map((d) => d.date),
      y: data.map((d) => d.sma_50),
      type: 'scatter',
      mode: 'lines',
      name: 'SMA 50',
      line: { color: '#3b82f6', width: 2 },
    });
  }

  const layout = {
    title: `${symbol} Stock Price`,
    xaxis: {
      title: 'Date',
      rangeslider: { visible: false },
    },
    yaxis: {
      title: 'Price ($)',
      side: 'left',
    },
    yaxis2: {
      title: 'Volume',
      overlaying: 'y',
      side: 'right',
      showgrid: false,
    },
    height: 500,
    hovermode: 'x unified',
    plot_bgcolor: '#f8fafc',
    paper_bgcolor: 'white',
    font: { family: 'system-ui' },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
  };

  return (
    <AnimatedCard delay={0.6}>
      <h3 className="text-xl font-bold text-slate-800 mb-4">
        Candlestick Chart with Indicators
      </h3>
      <Plot data={traces} layout={layout} config={config} className="w-full" />
    </AnimatedCard>
  );
};

export default CandlestickChart;
