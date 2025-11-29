import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Zap,
  CheckCircle,
  XCircle,
  BarChart
} from 'lucide-react';
import { AnimatedCard, MetricCard, PulseCard } from './AnimatedCard';
import { AnimatedButton } from './AnimatedButton';
import { stockAPI } from '../services/api';
import CandlestickChart from './CandlestickChart';
import FeatureImportanceChart from './FeatureImportanceChart';
import { ShimmerCard, ShimmerChart } from './Shimmer';
import AdvancedCandlestickChart from './AdvancedCandlestickChart';

const StockPrediction = ({
  symbol,
  period,
  modelType,
  targetType,
  predictionData,
  setPredictionData
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runPrediction = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await stockAPI.predict({
        symbol,
        period,
        model_type: modelType,
        target_type: targetType,
        target_days: 1,
        use_advanced_features: false
      });

      setPredictionData(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderPrediction = () => {
    if (!predictionData) return null;

    const { latest_prediction, target_type } = predictionData;

    if (target_type === 'direction') {
      const isUp = latest_prediction === 1;
      return {
        text: isUp ? 'UP' : 'DOWN',
        icon: isUp ? TrendingUp : TrendingDown,
        trend: isUp ? 'up' : 'down',
        color: isUp ? 'text-emerald-400' : 'text-red-400',
        bgColor: isUp ? 'bg-emerald-500' : 'bg-red-500'
      };
    } else {
      return {
        text: `${latest_prediction.toFixed(2)}%`,
        icon: Activity,
        trend: 'neutral',
        color: 'text-blue-400',
        bgColor: 'bg-blue-500'
      };
    }
  };

  const prediction = renderPrediction();

  return (
    <div className="space-y-6">
      {/* Action Button */}
      <AnimatedCard>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-slate-100 mb-2">
              Ready to Analyze {symbol}?
            </h3>
            <p className="text-slate-400">
              Using {modelType} model to predict {targetType} for the next trading day
            </p>
          </div>
          <AnimatedButton
            onClick={runPrediction}
            loading={loading}
            icon={Zap}
          >
            Run Analysis
          </AnimatedButton>
        </div>
      </AnimatedCard>

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-red-500/20 border-l-4 border-red-500 p-4 rounded-lg backdrop-blur-sm"
        >
          <p className="text-red-200 font-semibold">Error: {error}</p>
        </motion.div>
      )}

      {/* Loading State with Shimmer */}
      {loading && (
        <>
          {/* Shimmer Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <ShimmerCard />
            <ShimmerCard />
            <ShimmerCard />
            <ShimmerCard />
          </div>

          {/* Shimmer Performance Card */}
          <ShimmerCard />

          {/* Shimmer Table */}
          <ShimmerCard />

          {/* Shimmer Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ShimmerChart />
            <ShimmerChart />
          </div>
        </>
      )}

      {/* Results */}
      {predictionData && !loading && (
        <>
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="Current Price"
              value={`$${predictionData.latest_price.toFixed(2)}`}
              icon={Activity}
              trend="neutral"
              delay={0.1}
            />

            {prediction && (
              <PulseCard>
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-400 mb-2">
                      Next Day Prediction
                    </p>
                    <div className="flex items-center gap-3">
                      <div className={`p-3 rounded-lg ${prediction.bgColor} bg-opacity-20`}>
                        <prediction.icon className={`w-8 h-8 ${prediction.color}`} />
                      </div>
                      <p className={`text-4xl font-bold ${prediction.color}`}>
                        {prediction.text}
                      </p>
                    </div>
                  </div>
                </div>
              </PulseCard>
            )}

            <MetricCard
              title="Model Accuracy"
              value={`${(predictionData.metrics.accuracy * 100).toFixed(1)}%`}
              delta={
                predictionData.metrics.accuracy > 0.55
                  ? '🎯 Good Performance'
                  : '⚠️ Moderate Performance'
              }
              icon={Target}
              trend={predictionData.metrics.accuracy > 0.55 ? 'up' : 'neutral'}
              delay={0.2}
            />

            <MetricCard
              title="Total Features"
              value={predictionData.total_features}
              delta={`${predictionData.total_samples} samples`}
              icon={Activity}
              trend="neutral"
              delay={0.3}
            />
          </div>

          {/* Model Performance Details */}
          <AnimatedCard delay={0.4}>
            <h3 className="text-xl font-bold text-slate-100 mb-4 flex items-center gap-2">
              <BarChart className="w-6 h-6 text-blue-400" />
              Model Performance Metrics
            </h3>

            {targetType === 'direction' ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">Accuracy</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {(predictionData.metrics.accuracy * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">Precision</p>
                  <p className="text-2xl font-bold text-emerald-400">
                    {(predictionData.metrics.precision * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">Recall</p>
                  <p className="text-2xl font-bold text-teal-400">
                    {(predictionData.metrics.recall * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">F1-Score</p>
                  <p className="text-2xl font-bold text-indigo-400">
                    {(predictionData.metrics.f1_score * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">R² Score</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {predictionData.metrics.r2.toFixed(4)}
                  </p>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">RMSE</p>
                  <p className="text-2xl font-bold text-emerald-400">
                    {predictionData.metrics.rmse.toFixed(4)}
                  </p>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">MAE</p>
                  <p className="text-2xl font-bold text-teal-400">
                    {predictionData.metrics.mae.toFixed(4)}
                  </p>
                </div>
                <div className="text-center p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-slate-400 mb-1">MSE</p>
                  <p className="text-2xl font-bold text-indigo-400">
                    {predictionData.metrics.mse.toFixed(4)}
                  </p>
                </div>
              </div>
            )}
          </AnimatedCard>

          {/* Recent Predictions Table */}
          <AnimatedCard delay={0.5}>
            <h3 className="text-xl font-bold text-slate-100 mb-4">
              Recent Predictions (Last 20 Days)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 font-semibold text-slate-300">
                      Date
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-300">
                      Close Price
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-300">
                      Actual
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-300">
                      Predicted
                    </th>
                    {targetType === 'direction' && (
                      <th className="text-center py-3 px-4 font-semibold text-slate-300">
                        Result
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {predictionData.recent_predictions.map((row, index) => {
                    const isCorrect =
                      targetType === 'direction' && row.actual === row.predicted;
                    return (
                      <motion.tr
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="border-b border-white/5 hover:bg-white/5 transition-colors"
                      >
                        <td className="py-3 px-4 text-slate-200 font-medium">
                          {new Date(row.date).toLocaleDateString()}
                        </td>
                        <td className="py-3 px-4 font-semibold text-slate-100">
                          ${row.close.toFixed(2)}
                        </td>
                        <td className="py-3 px-4">
                          {targetType === 'direction' ? (
                            <span
                              className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-semibold ${row.actual === 1
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-red-500/20 text-red-400'
                                }`}
                            >
                              {row.actual === 1 ? (
                                <>
                                  <TrendingUp className="w-4 h-4" /> UP
                                </>
                              ) : (
                                <>
                                  <TrendingDown className="w-4 h-4" /> DOWN
                                </>
                              )}
                            </span>
                          ) : (
                            <span className="text-slate-300">
                              {row.actual.toFixed(2)}%
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          {targetType === 'direction' ? (
                            <span
                              className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-semibold ${row.predicted === 1
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-red-500/20 text-red-400'
                                }`}
                            >
                              {row.predicted === 1 ? (
                                <>
                                  <TrendingUp className="w-4 h-4" /> UP
                                </>
                              ) : (
                                <>
                                  <TrendingDown className="w-4 h-4" /> DOWN
                                </>
                              )}
                            </span>
                          ) : (
                            <span className="text-slate-300">
                              {row.predicted.toFixed(2)}%
                            </span>
                          )}
                        </td>
                        {targetType === 'direction' && (
                          <td className="py-3 px-4 text-center">
                            {isCorrect ? (
                              <CheckCircle className="w-5 h-5 text-emerald-500 mx-auto" />
                            ) : (
                              <XCircle className="w-5 h-5 text-red-500 mx-auto" />
                            )}
                          </td>
                        )}
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </AnimatedCard>

          {/* Advanced Chart */}
          <AdvancedCandlestickChart symbol={symbol} period={period} />

          {/* Feature Importance */}
          {predictionData.feature_importance && (
            <FeatureImportanceChart
              data={predictionData.feature_importance}
            />
          )}
        </>
      )}

      {/* Initial State */}
      {!predictionData && !loading && (
        <AnimatedCard>
          <div className="text-center py-12">
            <Activity className="w-20 h-20 text-blue-400 mx-auto mb-4 animate-pulse-slow" />
            <h3 className="text-2xl font-bold text-slate-100 mb-2">
              Ready to Predict
            </h3>
            <p className="text-slate-400">
              Click "Run Analysis" to start predicting {symbol} stock movement
            </p>
          </div>
        </AnimatedCard>
      )}
    </div>
  );
};

export default StockPrediction;
