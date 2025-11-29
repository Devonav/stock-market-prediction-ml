import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { TrendingUp, Activity, DollarSign, Target } from 'lucide-react';
import { AnimatedCard, MetricCard } from './AnimatedCard';

const PerformanceMetrics = ({ portfolio, history }) => {
    if (!portfolio) return null;

    const { total_value, total_pl, total_pl_pct, cash, holdings_value } = portfolio;

    // Prepare chart data
    const dates = history.map(h => h.date);
    const values = history.map(h => h.total_value);

    // Calculate drawdown if we have history
    let drawdown = [];
    if (values.length > 0) {
        let maxVal = -Infinity;
        const cumMax = [];
        values.forEach(v => {
            maxVal = Math.max(maxVal, v);
            cumMax.push(maxVal);
        });
        drawdown = values.map((v, i) => ((v - cumMax[i]) / cumMax[i]) * 100);
    }

    return (
        <div className="space-y-6">
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                    title="Total Portfolio Value"
                    value={`$${total_value.toFixed(2)}`}
                    delta={`${total_pl_pct.toFixed(2)}%`}
                    icon={DollarSign}
                    trend={total_pl >= 0 ? 'up' : 'down'}
                    delay={0.1}
                />

                <MetricCard
                    title="Total P&L"
                    value={`$${total_pl.toFixed(2)}`}
                    delta={total_pl >= 0 ? 'Profit' : 'Loss'}
                    icon={TrendingUp}
                    trend={total_pl >= 0 ? 'up' : 'down'}
                    delay={0.2}
                />

                <MetricCard
                    title="Cash Balance"
                    value={`$${cash.toFixed(2)}`}
                    icon={Activity}
                    trend="neutral"
                    delay={0.3}
                />

                <MetricCard
                    title="Invested Amount"
                    value={`$${holdings_value.toFixed(2)}`}
                    icon={Target}
                    trend="neutral"
                    delay={0.4}
                />
            </div>

            {/* Charts */}
            {history.length > 0 ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <AnimatedCard delay={0.5}>
                        <h3 className="text-xl font-bold text-slate-100 mb-4">Portfolio Growth</h3>
                        <Plot
                            data={[
                                {
                                    x: dates,
                                    y: values,
                                    type: 'scatter',
                                    mode: 'lines',
                                    line: { color: '#3b82f6', width: 3 },
                                    fill: 'tozeroy',
                                    name: 'Value'
                                }
                            ]}
                            layout={{
                                autosize: true,
                                height: 350,
                                margin: { l: 50, r: 20, t: 20, b: 40 },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { color: '#94a3b8' },
                                xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
                                yaxis: { gridcolor: 'rgba(255,255,255,0.1)', title: 'Value ($)' }
                            }}
                            useResizeHandler={true}
                            className="w-full"
                        />
                    </AnimatedCard>

                    <AnimatedCard delay={0.6}>
                        <h3 className="text-xl font-bold text-slate-100 mb-4">Drawdown Analysis</h3>
                        <Plot
                            data={[
                                {
                                    x: dates,
                                    y: drawdown,
                                    type: 'scatter',
                                    mode: 'lines',
                                    line: { color: '#ef4444', width: 2 },
                                    fill: 'tozeroy',
                                    name: 'Drawdown'
                                }
                            ]}
                            layout={{
                                autosize: true,
                                height: 350,
                                margin: { l: 50, r: 20, t: 20, b: 40 },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { color: '#94a3b8' },
                                xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
                                yaxis: { gridcolor: 'rgba(255,255,255,0.1)', title: 'Drawdown (%)' }
                            }}
                            useResizeHandler={true}
                            className="w-full"
                        />
                    </AnimatedCard>
                </div>
            ) : (
                <AnimatedCard>
                    <div className="text-center py-12 text-slate-400">
                        <Activity className="w-16 h-16 mx-auto mb-4 opacity-50" />
                        <p>No history available yet. Start trading to see performance charts!</p>
                    </div>
                </AnimatedCard>
            )}
        </div>
    );
};

export default PerformanceMetrics;
