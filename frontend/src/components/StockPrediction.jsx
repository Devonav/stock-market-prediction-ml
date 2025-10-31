import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Zap,
  CheckCircle,
  XCircle
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
        color: isUp ? 'text-green-600' : 'text-red-600',
        bgColor: isUp ? 'bg-green-100' : 'bg-red-100'
      };
    } else {
      return {
        text: `${latest_prediction.toFixed(2)}%`,
        icon: Activity,
        trend: 'neutral',
        color: 'text-blue-600',
        bgColor: 'bg-blue-100'
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
            <h3 className="text-xl font-bold text-slate-800 mb-2">
              Ready to Analyze {symbol}?
            </h3>
            <p className="text-slate-600">
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
          className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg"
        >
          <p className="text-red-700 font-semibold">Error: {error}</p>
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
                    <p className="text-sm font-medium text-slate-600 mb-2">
                      Next Day Prediction
                    </p>
                    <div className="flex items-center gap-3">
                      <div className={`p-3 rounded-lg ${prediction.bgColor}`}>
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
            <h3 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
              <BarChart className="w-6 h-6 text-blue-600" />
              Model Performance Metrics
            </h3>

            {targetType === 'direction' ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">Accuracy</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {(predictionData.metrics.accuracy * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">Precision</p>
                  <p className="text-2xl font-bold text-green-600">
                    {(predictionData.metrics.precision * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">Recall</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {(predictionData.metrics.recall * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-indigo-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">F1-Score</p>
                  <p className="text-2xl font-bold text-indigo-600">
                    {(predictionData.metrics.f1_score * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">R² Score</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {predictionData.metrics.r2.toFixed(4)}
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">RMSE</p>
                  <p className="text-2xl font-bold text-green-600">
                    {predictionData.metrics.rmse.toFixed(4)}
                  </p>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">MAE</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {predictionData.metrics.mae.toFixed(4)}
                  </p>
                </div>
                <div className="text-center p-4 bg-indigo-50 rounded-lg">
                  <p className="text-sm text-slate-600 mb-1">MSE</p>
                  <p className="text-2xl font-bold text-indigo-600">
                    {predictionData.metrics.mse.toFixed(4)}
                  </p>
                </div>
              </div>
            )}
          </AnimatedCard>

          {/* Recent Predictions Table */}
          <AnimatedCard delay={0.5}>
            <h3 className="text-xl font-bold text-slate-800 mb-4">
              Recent Predictions (Last 20 Days)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b-2 border-slate-200">
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">
                      Date
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">
                      Close Price
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">
                      Actual
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">
                      Predicted
                    </th>
                    {targetType === 'direction' && (
                      <th className="text-center py-3 px-4 font-semibold text-slate-700">
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
                        className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                      >
                        <td className="py-3 px-4 text-slate-600">
                          {new Date(row.date).toLocaleDateString()}
                        </td>
                        <td className="py-3 px-4 font-semibold text-slate-800">
                          ${row.close.toFixed(2)}
                        </td>
                        <td className="py-3 px-4">
                          {targetType === 'direction' ? (
                            <span
                              className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-semibold ${
                                row.actual === 1
                                  ? 'bg-green-100 text-green-700'
                                  : 'bg-red-100 text-red-700'
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
                            <span className="text-slate-700">
                              {row.actual.toFixed(2)}%
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          {targetType === 'direction' ? (
                            <span
                              className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-semibold ${
                                row.predicted === 1
                                  ? 'bg-green-100 text-green-700'
                                  : 'bg-red-100 text-red-700'
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
                            <span className="text-slate-700">
                              {row.predicted.toFixed(2)}%
                            </span>
                          )}
                        </td>
                        {targetType === 'direction' && (
                          <td className="py-3 px-4 text-center">
                            {isCorrect ? (
                              <CheckCircle className="w-5 h-5 text-green-600 mx-auto" />
                            ) : (
                              <XCircle className="w-5 h-5 text-red-600 mx-auto" />
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
            <Activity className="w-20 h-20 text-blue-300 mx-auto mb-4 animate-pulse-slow" />
            <h3 className="text-2xl font-bold text-slate-700 mb-2">
              Ready to Predict
            </h3>
            <p className="text-slate-500">
              Click "Run Analysis" to start predicting {symbol} stock movement
            </p>
          </div>
        </AnimatedCard>
      )}
    </div>
  );
};

const BarChart = ({ className }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

export default StockPrediction;
