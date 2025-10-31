import { motion } from 'framer-motion';

export const AnimatedCard = ({
  children,
  className = '',
  delay = 0,
  variant = 'default',
  hover = true
}) => {
  const variants = {
    default: 'card',
    gradient: 'card-gradient',
    metric: 'metric-card'
  };

  const cardClass = variants[variant] || variants.default;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={hover ? { scale: 1.02, y: -5 } : {}}
      className={`${cardClass} ${className}`}
    >
      {children}
    </motion.div>
  );
};

export const MetricCard = ({
  title,
  value,
  delta,
  icon: Icon,
  trend = 'neutral',
  delay = 0
}) => {
  const trendColors = {
    up: 'border-green-500',
    down: 'border-red-500',
    neutral: 'border-blue-500'
  };

  const deltaColors = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-slate-600'
  };

  return (
    <AnimatedCard variant="metric" delay={delay} className={trendColors[trend]}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-600 mb-2">{title}</p>
          <p className="text-3xl font-bold text-slate-900">{value}</p>
          {delta && (
            <p className={`text-sm font-semibold mt-2 ${deltaColors[trend]}`}>
              {delta}
            </p>
          )}
        </div>
        {Icon && (
          <div className="ml-4">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <Icon className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        )}
      </div>
    </AnimatedCard>
  );
};

export const PulseCard = ({ children, className = '' }) => {
  return (
    <motion.div
      animate={{
        boxShadow: [
          '0 4px 6px -1px rgb(0 0 0 / 0.1)',
          '0 20px 25px -5px rgb(0 0 0 / 0.1)',
          '0 4px 6px -1px rgb(0 0 0 / 0.1)',
        ],
      }}
      transition={{
        duration: 2,
        repeat: Infinity,
        repeatType: 'reverse',
      }}
      className={`card ${className}`}
    >
      {children}
    </motion.div>
  );
};
