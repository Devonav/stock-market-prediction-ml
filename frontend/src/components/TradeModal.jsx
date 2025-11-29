import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, DollarSign, AlertCircle } from 'lucide-react';
import { AnimatedButton } from './AnimatedButton';

const TradeModal = ({ isOpen, onClose, symbol, currentPrice, onTrade, holdings }) => {
    const [action, setAction] = useState('buy');
    const [quantity, setQuantity] = useState(1);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (isOpen) {
            setQuantity(1);
            setError(null);
            setAction('buy');
        }
    }, [isOpen]);

    const handleSubmit = async () => {
        if (quantity <= 0) {
            setError('Quantity must be greater than 0');
            return;
        }

        if (action === 'sell') {
            const currentHolding = holdings.find(h => h.symbol === symbol);
            if (!currentHolding || currentHolding.quantity < quantity) {
                setError(`Insufficient shares. You own ${currentHolding?.quantity || 0} shares.`);
                return;
            }
        }

        try {
            await onTrade(symbol, action, quantity);
            onClose();
        } catch (err) {
            setError(err.message || 'Trade failed');
        }
    };

    const total = quantity * currentPrice;

    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                    />

                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="relative w-full max-w-md bg-slate-900 border border-white/10 rounded-2xl shadow-2xl overflow-hidden"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-white/10 bg-white/5">
                            <h3 className="text-xl font-bold text-white flex items-center gap-2">
                                <DollarSign className="w-5 h-5 text-emerald-400" />
                                Trade {symbol}
                            </h3>
                            <button
                                onClick={onClose}
                                className="text-slate-400 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Body */}
                        <div className="p-6 space-y-6">
                            {/* Price Info */}
                            <div className="flex justify-between items-center p-4 bg-white/5 rounded-xl border border-white/5">
                                <span className="text-slate-400">Current Price</span>
                                <span className="text-2xl font-bold text-white">
                                    ${currentPrice.toFixed(2)}
                                </span>
                            </div>

                            {/* Action Toggle */}
                            <div className="grid grid-cols-2 gap-2 p-1 bg-slate-800 rounded-lg">
                                <button
                                    onClick={() => setAction('buy')}
                                    className={`py-2 px-4 rounded-md font-semibold transition-all ${action === 'buy'
                                            ? 'bg-emerald-500 text-white shadow-lg'
                                            : 'text-slate-400 hover:text-slate-200'
                                        }`}
                                >
                                    Buy
                                </button>
                                <button
                                    onClick={() => setAction('sell')}
                                    className={`py-2 px-4 rounded-md font-semibold transition-all ${action === 'sell'
                                            ? 'bg-red-500 text-white shadow-lg'
                                            : 'text-slate-400 hover:text-slate-200'
                                        }`}
                                >
                                    Sell
                                </button>
                            </div>

                            {/* Quantity Input */}
                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                    Quantity
                                </label>
                                <input
                                    type="number"
                                    min="1"
                                    value={quantity}
                                    onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 0))}
                                    className="w-full bg-slate-800 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                                />
                            </div>

                            {/* Total Calculation */}
                            <div className="flex justify-between items-center pt-4 border-t border-white/10">
                                <span className="text-slate-300 font-medium">Estimated Total</span>
                                <span className="text-2xl font-bold text-blue-400">
                                    ${total.toFixed(2)}
                                </span>
                            </div>

                            {/* Error Message */}
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20"
                                >
                                    <AlertCircle className="w-4 h-4" />
                                    {error}
                                </motion.div>
                            )}

                            {/* Submit Button */}
                            <AnimatedButton
                                onClick={handleSubmit}
                                className={`w-full justify-center ${action === 'buy' ? 'bg-emerald-500 hover:bg-emerald-600' : 'bg-red-500 hover:bg-red-600'
                                    }`}
                            >
                                {action === 'buy' ? 'Place Buy Order' : 'Place Sell Order'}
                            </AnimatedButton>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
};

export default TradeModal;
