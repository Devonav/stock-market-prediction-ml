import { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Zap, Trophy } from 'lucide-react';
import Plot from 'react-plotly.js';
import { AnimatedCard } from './AnimatedCard';
import { AnimatedButton } from './AnimatedButton';
import { stockAPI } from '../services/api';

const ModelComparison = ({ symbol, period, targetType }) => {
  const [loading, setLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [error, setError] = useState(null);

  const runComparison = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await stockAPI.compareModels({
        symbol,
        period,
        target_type: targetType,
        target_days: 1,
        use_advanced_features: false,
      });

      setComparisonData(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const getBestModel = () => {
    if (!comparisonData) return null;

    const { comparison } = comparisonData;
    const metricKey = targetType === 'direction' ? 'accuracy' : 'r2';

    let bestModel = null;
    let bestScore = -Infinity;

    Object.entries(comparison).forEach(([model, data]) => {
      const score = data.metrics[metricKey];
      if (score > bestScore) {
        bestScore = score;
        bestModel = model;
      }
    });

    return { model: bestModel, score: bestScore };
  };

  const best = getBestModel();

  return (
    <div className="space-y-6">
      {/* Action Button */}
      <AnimatedCard>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-slate-800 mb-2">
              Compare All Models for {symbol}
            </h3>
            <p className="text-slate-600">
              Train and compare multiple ML models to find the best performer
            </p>
          </div>
          <AnimatedButton onClick={runComparison} loading={loading} icon={Zap}>
            Run Comparison
          </AnimatedButton>
        </div>
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
      {comparisonData && !loading && (
        <>
          {/* Best Model */}
          {best && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <Trophy className="w-10 h-10 text-amber-600" />
                <div>
                  <p className="text-sm font-medium text-slate-600">
                    Best Model
                  </p>
                  <p className="text-2xl font-bold text-slate-900">
                    {best.model}
                  </p>
                  <p className="text-lg text-slate-700 mt-1">
                    {targetType === 'direction'
                      ? `Accuracy: ${(best.score * 100).toFixed(2)}%`
                      : `R² Score: ${best.score.toFixed(4)}`}
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Comparison Table */}
          <AnimatedCard delay={0.2}>
            <h3 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-blue-600" />
              Model Performance Comparison
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b-2 border-slate-200">
                    <th className="text-left py-3 px-4 font-semibold text-slate-700">
                      Model
                    </th>
                    {targetType === 'direction' ? (
                      <>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          Accuracy
                        </th>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          Precision
                        </th>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          Recall
                        </th>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          F1-Score
                        </th>
                      </>
                    ) : (
                      <>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          R²
                        </th>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          RMSE
                        </th>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          MAE
                        </th>
                        <th className="text-center py-3 px-4 font-semibold text-slate-700">
                          MSE
                        </th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(comparisonData.comparison).map(
                    ([model, data], index) => (
                      <motion.tr
                        key={model}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`border-b border-slate-100 hover:bg-slate-50 transition-colors ${
                          best.model === model ? 'bg-blue-50' : ''
                        }`}
                      >
                        <td className="py-3 px-4 font-semibold text-slate-800">
                          {model}
                          {best.model === model && (
                            <span className="ml-2 text-amber-600">👑</span>
                          )}
                        </td>
                        {targetType === 'direction' ? (
                          <>
                            <td className="py-3 px-4 text-center text-blue-600 font-semibold">
                              {(data.metrics.accuracy * 100).toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-center text-green-600 font-semibold">
                              {(data.metrics.precision * 100).toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-center text-purple-600 font-semibold">
                              {(data.metrics.recall * 100).toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-center text-indigo-600 font-semibold">
                              {(data.metrics.f1_score * 100).toFixed(2)}%
                            </td>
                          </>
                        ) : (
                          <>
                            <td className="py-3 px-4 text-center text-blue-600 font-semibold">
                              {data.metrics.r2.toFixed(4)}
                            </td>
                            <td className="py-3 px-4 text-center text-green-600 font-semibold">
                              {data.metrics.rmse.toFixed(4)}
                            </td>
                            <td className="py-3 px-4 text-center text-purple-600 font-semibold">
                              {data.metrics.mae.toFixed(4)}
                            </td>
                            <td className="py-3 px-4 text-center text-indigo-600 font-semibold">
                              {data.metrics.mse.toFixed(4)}
                            </td>
                          </>
                        )}
                      </motion.tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          </AnimatedCard>

          {/* Comparison Chart */}
          <AnimatedCard delay={0.4}>
            <h3 className="text-xl font-bold text-slate-800 mb-4">
              Visual Comparison
            </h3>
            <ComparisonChart
              comparison={comparisonData.comparison}
              targetType={targetType}
            />
          </AnimatedCard>
        </>
      )}

      {/* Initial State */}
      {!comparisonData && !loading && (
        <AnimatedCard>
          <div className="text-center py-12">
            <BarChart3 className="w-20 h-20 text-blue-300 mx-auto mb-4 animate-pulse-slow" />
            <h3 className="text-2xl font-bold text-slate-700 mb-2">
              Compare Models
            </h3>
            <p className="text-slate-500">
              Click "Run Comparison" to compare all available models for {symbol}
            </p>
          </div>
        </AnimatedCard>
      )}
    </div>
  );
};

const ComparisonChart = ({ comparison, targetType }) => {
  const models = Object.keys(comparison);
  const metricKey = targetType === 'direction' ? 'accuracy' : 'r2';
  const values = models.map((model) => comparison[model].metrics[metricKey]);

  const trace = {
    x: models,
    y: values,
    type: 'bar',
    marker: {
      color: values,
      colorscale: 'Blues',
      showscale: true,
    },
    text: values.map((v) =>
      targetType === 'direction' ? `${(v * 100).toFixed(2)}%` : v.toFixed(4)
    ),
    textposition: 'auto',
  };

  const layout = {
    title: `Model Comparison - ${
      targetType === 'direction' ? 'Accuracy' : 'R² Score'
    }`,
    xaxis: { title: 'Model' },
    yaxis: {
      title: targetType === 'direction' ? 'Accuracy' : 'R² Score',
    },
    height: 400,
    plot_bgcolor: '#f8fafc',
    paper_bgcolor: 'white',
    font: { family: 'system-ui' },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
  };

  return <Plot data={[trace]} layout={layout} config={config} className="w-full" />;
};

export default ModelComparison;
