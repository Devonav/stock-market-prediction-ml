import Plot from 'react-plotly.js';
import { AnimatedCard } from './AnimatedCard';

const FeatureImportanceChart = ({ data }) => {
  if (!data || data.length === 0) {
    return null;
  }

  const trace = {
    x: data.map((d) => d.importance),
    y: data.map((d) => d.feature),
    type: 'bar',
    orientation: 'h',
    marker: {
      color: data.map((d) => d.importance),
      colorscale: 'Viridis',
      showscale: true,
    },
  };

  const layout = {
    title: 'Top 20 Most Important Features',
    xaxis: { title: 'Importance' },
    yaxis: { title: 'Feature', automargin: true },
    height: 600,
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
    <AnimatedCard delay={0.7}>
      <h3 className="text-xl font-bold text-slate-800 mb-4">
        Feature Importance
      </h3>
      <Plot data={[trace]} layout={layout} config={config} className="w-full" />
    </AnimatedCard>
  );
};

export default FeatureImportanceChart;
