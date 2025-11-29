import { motion } from 'framer-motion';

export const AnimatedCard = ({
  children,
  className = '',
  delay = 0,
  variant = 'default',
  hover = true
}) => {
  const variants = {
    default: 'glass-card',
    gradient: 'glass-card', // Unified look
    metric: 'glass-panel'
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
    up: 'border-l-4 border-green-500',
    down: 'border-l-4 border-red-500',
    neutral: 'border-l-4 border-blue-500'
  };

  const deltaColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-slate-400'
  };

  return (
    <AnimatedCard variant="metric" delay={delay} className={trendColors[trend]}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-400 mb-2">{title}</p>
          <p className="text-3xl font-bold text-white">{value}</p>
          {delta && (
            <p className={`text-sm font-semibold mt-2 ${deltaColors[trend]}`}>
              {delta}
            </p>
          )}
        </div>
        {Icon && (
          <div className="ml-4">
            <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-sm border border-white/10">
              <Icon className="w-6 h-6 text-indigo-400" />
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
