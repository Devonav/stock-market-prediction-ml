import './GridPattern.css';

const GridPattern = ({
  width = 40,
  height = 40,
  x = 0,
  y = 0,
  strokeDasharray = '0',
  className = '',
  squares = []
}) => {
  return (
    <svg
      className={`grid-pattern ${className}`}
      aria-hidden="true"
    >
      <defs>
        <pattern
          id="grid-pattern"
          width={width}
          height={height}
          x={x}
          y={y}
          patternUnits="userSpaceOnUse"
        >
          <path
            d={`M.5 ${height}V.5H${width}`}
            fill="none"
            stroke="currentColor"
            strokeDasharray={strokeDasharray}
          />
        </pattern>
      </defs>
      <rect width="100%" height="100%" strokeWidth={0} fill="url(#grid-pattern)" />
      {squares && squares.map(([x, y], index) => (
        <rect
          key={index}
          width={width - 1}
          height={height - 1}
          x={x * width + 1}
          y={y * height + 1}
          strokeWidth={0}
          fill="currentColor"
          className="grid-square"
        />
      ))}
    </svg>
  );
};

export default GridPattern;
