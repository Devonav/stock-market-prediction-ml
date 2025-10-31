import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  DollarSign,
  TrendingUp,
  Target,
  Activity,
  Zap,
  AlertCircle
} from 'lucide-react';
import Plot from 'react-plotly.js';
import { AnimatedCard, MetricCard } from './AnimatedCard';
import { AnimatedButton } from './AnimatedButton';
import { stockAPI } from '../services/api';

const BacktestSimulator = ({ symbol, modelType, targetType, predictionData }) => {
  const [loading, setLoading] = useState(false);
  const [backtestResults, setBacktestResults] = useState(null);
  const [error, setError] = useState(null);

  // Backtest parameters
  const [initialCapital, setInitialCapital] = useState(10000);
  const [stopLoss, setStopLoss] = useState(5);
  const [takeProfit, setTakeProfit] = useState(10);

  const runBacktest = async () => {
    if (!predictionData) {
      setError('Please run a prediction first before backtesting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await stockAPI.runBacktest({
        symbol,
        model_type: modelType,
        target_type: targetType,
        initial_capital: initialCapital,
        stop_loss: stopLoss / 100,
        take_profit: takeProfit / 100,
        commission: 0.001,
        slippage: 0.001,
      });

      setBacktestResults(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <AnimatedCard>
        <div className="flex items-center gap-2 mb-6">
          <DollarSign className="w-6 h-6 text-green-600" />
          <h2 className="text-2xl font-bold text-slate-800">
            Backtesting Simulator
          </h2>
        </div>

        {!predictionData && (
          <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-lg mb-6">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5" />
              <div>
                <p className="font-semibold text-amber-800">
                  Prediction Required
                </p>
                <p className="text-amber-700 text-sm">
                  Please run a prediction first in the "Predictions" tab before
                  running a backtest.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Initial Capital ($)
            </label>
            <input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              className="input-field"
              step="1000"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Stop Loss (%)
            </label>
            <input
              type="number"
              value={stopLoss}
              onChange={(e) => setStopLoss(Number(e.target.value))}
              className="input-field"
              step="0.5"
              min="0"
              max="20"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Take Profit (%)
            </label>
            <input
              type="number"
              value={takeProfit}
              onChange={(e) => setTakeProfit(Number(e.target.value))}
              className="input-field"
              step="1"
              min="0"
              max="50"
            />
          </div>
        </div>

        <AnimatedButton
          onClick={runBacktest}
          loading={loading}
          icon={Zap}
          disabled={!predictionData}
        >
          Run Backtest
        </AnimatedButton>
      </AnimatedCard>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg"
        >
          <p className="text-red-700 font-semibold">Error: {error}</p>
        </motion.div>
      )}

      {/* Results */}
      {backtestResults && !loading && (
        <>
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="Final Value"
              value={`$${backtestResults.final_value.toFixed(2)}`}
              delta={`${backtestResults.total_return_pct.toFixed(2)}%`}
              icon={DollarSign}
              trend={
                backtestResults.total_return_pct > 0
                  ? 'up'
                  : backtestResults.total_return_pct < 0
                  ? 'down'
                  : 'neutral'
              }
              delay={0.1}
            />

            <MetricCard
              title="Total Trades"
              value={backtestResults.num_trades}
              icon={Activity}
              trend="neutral"
              delay={0.2}
            />

            <MetricCard
              title="Win Rate"
              value={`${(backtestResults.win_rate * 100).toFixed(1)}%`}
              icon={Target}
              trend={backtestResults.win_rate > 0.5 ? 'up' : 'down'}
              delay={0.3}
            />

            <MetricCard
              title="Sharpe Ratio"
              value={backtestResults.sharpe_ratio.toFixed(2)}
              delta={
                backtestResults.sharpe_ratio > 1
                  ? 'Good'
                  : backtestResults.sharpe_ratio > 0.5
                  ? 'Moderate'
                  : 'Poor'
              }
              icon={TrendingUp}
              trend={
                backtestResults.sharpe_ratio > 1
                  ? 'up'
                  : backtestResults.sharpe_ratio > 0.5
                  ? 'neutral'
                  : 'down'
              }
              delay={0.4}
            />
          </div>

          {/* Portfolio Value Chart */}
          {backtestResults.portfolio_values &&
            backtestResults.portfolio_values.length > 0 && (
              <AnimatedCard delay={0.5}>
                <h3 className="text-xl font-bold text-slate-800 mb-4">
                  Portfolio Value Over Time
                </h3>
                <PortfolioChart
                  data={backtestResults.portfolio_values}
                  initialCapital={initialCapital}
                />
              </AnimatedCard>
            )}

          {/* Trade History */}
          {backtestResults.trades && backtestResults.trades.length > 0 && (
            <AnimatedCard delay={0.6}>
              <h3 className="text-xl font-bold text-slate-800 mb-4">
                Trade History
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b-2 border-slate-200">
                      <th className="text-left py-3 px-4 font-semibold text-slate-700">
                        Entry Date
                      </th>
                      <th className="text-left py-3 px-4 font-semibold text-slate-700">
                        Exit Date
                      </th>
                      <th className="text-left py-3 px-4 font-semibold text-slate-700">
                        Entry Price
                      </th>
                      <th className="text-left py-3 px-4 font-semibold text-slate-700">
                        Exit Price
                      </th>
                      <th className="text-right py-3 px-4 font-semibold text-slate-700">
                        Profit
                      </th>
                      <th className="text-right py-3 px-4 font-semibold text-slate-700">
                        Return
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtestResults.trades.map((trade, index) => (
                      <motion.tr
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                      >
                        <td className="py-3 px-4 text-slate-600">
                          {new Date(trade.entry_date).toLocaleDateString()}
                        </td>
                        <td className="py-3 px-4 text-slate-600">
                          {new Date(trade.exit_date).toLocaleDateString()}
                        </td>
                        <td className="py-3 px-4 text-slate-800 font-semibold">
                          ${trade.entry_price.toFixed(2)}
                        </td>
                        <td className="py-3 px-4 text-slate-800 font-semibold">
                          ${trade.exit_price.toFixed(2)}
                        </td>
                        <td
                          className={`py-3 px-4 text-right font-bold ${
                            trade.profit > 0
                              ? 'text-green-600'
                              : trade.profit < 0
                              ? 'text-red-600'
                              : 'text-slate-600'
                          }`}
                        >
                          ${trade.profit.toFixed(2)}
                        </td>
                        <td
                          className={`py-3 px-4 text-right font-bold ${
                            trade.return > 0
                              ? 'text-green-600'
                              : trade.return < 0
                              ? 'text-red-600'
                              : 'text-slate-600'
                          }`}
                        >
                          {trade.return.toFixed(2)}%
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </AnimatedCard>
          )}
        </>
      )}

      {/* Initial State */}
      {!backtestResults && !loading && (
        <AnimatedCard>
          <div className="text-center py-12">
            <DollarSign className="w-20 h-20 text-green-300 mx-auto mb-4 animate-pulse-slow" />
            <h3 className="text-2xl font-bold text-slate-700 mb-2">
              Backtest Your Strategy
            </h3>
            <p className="text-slate-500">
              Configure parameters above and click "Run Backtest" to simulate
              trading performance
            </p>
          </div>
        </AnimatedCard>
      )}
    </div>
  );
};

const PortfolioChart = ({ data, initialCapital }) => {
  const dates = data.map((d) => d.date);
  const values = data.map((d) => d.value);

  // Calculate drawdown
  const cumMax = [];
  let maxVal = -Infinity;
  values.forEach((val) => {
    maxVal = Math.max(maxVal, val);
    cumMax.push(maxVal);
  });

  const drawdown = values.map((val, i) =>
    ((val - cumMax[i]) / cumMax[i]) * 100
  );

  const portfolioTrace = {
    x: dates,
    y: values,
    type: 'scatter',
    mode: 'lines',
    name: 'Portfolio Value',
    line: { color: '#3b82f6', width: 3 },
    fill: 'tonexty',
  };

  const initialLine = {
    x: dates,
    y: Array(dates.length).fill(initialCapital),
    type: 'scatter',
    mode: 'lines',
    name: 'Initial Capital',
    line: { color: '#64748b', width: 2, dash: 'dash' },
  };

  const drawdownTrace = {
    x: dates,
    y: drawdown,
    type: 'scatter',
    mode: 'lines',
    name: 'Drawdown',
    line: { color: '#ef4444', width: 2 },
    fill: 'tozeroy',
    yaxis: 'y2',
  };

  const layout = {
    title: 'Portfolio Performance',
    xaxis: { title: 'Date' },
    yaxis: { title: 'Value ($)', side: 'left' },
    yaxis2: {
      title: 'Drawdown (%)',
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
    <Plot
      data={[initialLine, portfolioTrace, drawdownTrace]}
      layout={layout}
      config={config}
      className="w-full"
    />
  );
};

export default BacktestSimulator;
