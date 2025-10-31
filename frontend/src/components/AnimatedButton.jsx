import { motion } from 'framer-motion';

export const AnimatedButton = ({
  children,
  onClick,
  variant = 'primary',
  loading = false,
  disabled = false,
  icon: Icon,
  className = ''
}) => {
  const variants = {
    primary: 'btn-primary',
    secondary: 'btn-secondary'
  };

  const buttonClass = variants[variant] || variants.primary;

  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.05 }}
      whileTap={{ scale: disabled ? 1 : 0.95 }}
      onClick={onClick}
      disabled={disabled || loading}
      className={`${buttonClass} ${className} flex items-center justify-center gap-2 ${
        (disabled || loading) ? 'opacity-50 cursor-not-allowed' : ''
      }`}
    >
      {loading ? (
        <>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
          />
          <span>Processing...</span>
        </>
      ) : (
        <>
          {Icon && <Icon className="w-5 h-5" />}
          {children}
        </>
      )}
    </motion.button>
  );
};

export const IconButton = ({ icon: Icon, onClick, className = '', tooltip }) => {
  return (
    <motion.button
      whileHover={{ scale: 1.1, rotate: 5 }}
      whileTap={{ scale: 0.9 }}
      onClick={onClick}
      className={`p-3 rounded-lg bg-white shadow-md hover:shadow-lg border border-slate-200 transition-all ${className}`}
      title={tooltip}
    >
      <Icon className="w-5 h-5 text-slate-700" />
    </motion.button>
  );
};
