import { motion } from 'framer-motion';
import { useState } from 'react';

export const AnimatedInput = ({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  icon: Icon,
  className = ''
}) => {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`relative ${className}`}
    >
      {label && (
        <label className="block text-sm font-semibold text-slate-300 mb-2">
          {label}
        </label>
      )}
      <div className="relative">
        {Icon && (
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
            <Icon className={`w-5 h-5 transition-colors ${isFocused ? 'text-blue-400' : 'text-slate-400'
              }`} />
          </div>
        )}
        <motion.input
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          className={`glass-input w-full ${Icon ? 'pl-11' : ''}`}
          whileFocus={{ scale: 1.01 }}
        />
      </div>
    </motion.div>
  );
};

export const AnimatedSelect = ({
  label,
  value,
  onChange,
  options,
  className = ''
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={className}
    >
      {label && (
        <label className="block text-sm font-semibold text-slate-300 mb-2">
          {label}
        </label>
      )}
      <select
        value={value}
        onChange={onChange}
        className="glass-input w-full cursor-pointer appearance-none"
      >
        {options.map((option, index) => (
          <option key={index} value={option.value} className="bg-slate-800 text-white">
            {option.label}
          </option>
        ))}
      </select>
    </motion.div>
  );
};
