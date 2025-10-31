import './Shimmer.css';

const Shimmer = ({
  width = '100%',
  height = '100%',
  className = '',
  borderRadius = '0.75rem'
}) => {
  return (
    <div
      className={`shimmer ${className}`}
      style={{
        width,
        height,
        borderRadius
      }}
    >
      <div className="shimmer-wave" />
    </div>
  );
};

export const ShimmerCard = ({ className = '' }) => (
  <div className={`shimmer-card ${className}`}>
    <Shimmer height="200px" />
    <div className="shimmer-card-content">
      <Shimmer height="24px" width="60%" />
      <Shimmer height="16px" width="40%" />
      <Shimmer height="16px" width="80%" />
    </div>
  </div>
);

export const ShimmerChart = ({ className = '' }) => (
  <div className={`shimmer-chart ${className}`}>
    <Shimmer height="100%" />
  </div>
);

export default Shimmer;
