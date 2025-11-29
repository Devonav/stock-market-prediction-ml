import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Plus, History, PieChart, TrendingUp, TrendingDown } from 'lucide-react';
import { AnimatedCard } from './AnimatedCard';
import { AnimatedButton } from './AnimatedButton';
import TradeModal from './TradeModal';
import PerformanceMetrics from './PerformanceMetrics';
import { stockAPI, socket } from '../services/api';

const PortfolioDashboard = () => {
    const [portfolio, setPortfolio] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isTradeModalOpen, setIsTradeModalOpen] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
    const [currentPrice, setCurrentPrice] = useState(0);

    const fetchPortfolio = async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            // Add 5s timeout
            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Request timed out - Backend may be unreachable')), 5000)
            );

            const data = await Promise.race([
                stockAPI.getPortfolio(),
                timeoutPromise
            ]);

            setPortfolio(data);
            setError(null);
        } catch (err) {
            setError(err.message || 'Failed to load portfolio data');
            console.error(err);
        } finally {
            if (!silent) setLoading(false);
        }
    };

    useEffect(() => {
        fetchPortfolio();

        const handlePriceUpdate = () => {
            fetchPortfolio(true);
        };

        socket.on('price_update', handlePriceUpdate);
        return () => {
            socket.off('price_update', handlePriceUpdate);
        };
    }, []);

    const handleTradeClick = async (symbol) => {
        setSelectedSymbol(symbol);
        // Fetch current price for the symbol
        try {
            const data = await stockAPI.getStockData(symbol, '1d');
            setCurrentPrice(data.latest_price);
            setIsTradeModalOpen(true);
        } catch (err) {
            console.error('Failed to fetch price', err);
        }
    };

    const executeTrade = async (symbol, action, quantity) => {
        try {
            await stockAPI.executeTrade(symbol, action, quantity);
            await fetchPortfolio(); // Refresh data
        } catch (err) {
            throw new Error(err.response?.data?.error || 'Trade failed');
        }
    };

    if (loading && !portfolio) {
        return (
            <div className="flex items-center justify-center h-64">
                <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-200">
                {error}
            </div>
        );
    }

    if (!portfolio) {
        return (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
                <p>No portfolio data available.</p>
                <AnimatedButton
                    onClick={fetchPortfolio}
                    icon={RefreshCw}
                    className="mt-4 bg-blue-500 hover:bg-blue-600"
                >
                    Retry
                </AnimatedButton>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header Actions */}
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-slate-100">My Portfolio</h2>
                <div className="flex gap-3">
                    <AnimatedButton
                        onClick={() => handleTradeClick('AAPL')} // Default to AAPL for now
                        icon={Plus}
                        className="bg-emerald-500 hover:bg-emerald-600"
                    >
                        New Trade
                    </AnimatedButton>
                    <AnimatedButton
                        onClick={fetchPortfolio}
                        icon={RefreshCw}
                        className="bg-slate-700 hover:bg-slate-600"
                    >
                        Refresh
                    </AnimatedButton>
                </div>
            </div>

            {/* Performance Overview */}
            <PerformanceMetrics
                portfolio={portfolio}
                history={portfolio?.history || []}
            />

            {/* Holdings Table */}
            <AnimatedCard delay={0.2}>
                <h3 className="text-xl font-bold text-slate-100 mb-4 flex items-center gap-2">
                    <PieChart className="w-5 h-5 text-blue-400" />
                    Current Holdings
                </h3>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-white/10">
                                <th className="text-left py-3 px-4 font-semibold text-slate-300">Symbol</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Shares</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Avg Cost</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Current Price</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Market Value</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Return</th>
                                <th className="text-center py-3 px-4 font-semibold text-slate-300">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {portfolio?.holdings?.length > 0 ? (
                                portfolio.holdings.map((holding, index) => (
                                    <motion.tr
                                        key={holding.symbol}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: index * 0.05 }}
                                        className="border-b border-white/5 hover:bg-white/5 transition-colors"
                                    >
                                        <td className="py-3 px-4 font-bold text-white">{holding.symbol}</td>
                                        <td className="py-3 px-4 text-right text-slate-300">{holding.quantity}</td>
                                        <td className="py-3 px-4 text-right text-slate-300">${holding.avg_cost.toFixed(2)}</td>
                                        <td className="py-3 px-4 text-right text-slate-300">${holding.current_price.toFixed(2)}</td>
                                        <td className="py-3 px-4 text-right font-semibold text-white">
                                            ${holding.market_value.toFixed(2)}
                                        </td>
                                        <td className={`py-3 px-4 text-right font-bold ${holding.unrealized_pl >= 0 ? 'text-emerald-400' : 'text-red-400'
                                            }`}>
                                            {holding.unrealized_pl >= 0 ? '+' : ''}{holding.unrealized_pl_pct.toFixed(2)}%
                                        </td>
                                        <td className="py-3 px-4 text-center">
                                            <button
                                                onClick={() => handleTradeClick(holding.symbol)}
                                                className="text-sm bg-blue-500/20 text-blue-400 px-3 py-1 rounded hover:bg-blue-500/30 transition-colors"
                                            >
                                                Trade
                                            </button>
                                        </td>
                                    </motion.tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="7" className="py-8 text-center text-slate-400">
                                        No holdings yet. Start trading to build your portfolio!
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </AnimatedCard>

            {/* Transaction History */}
            <AnimatedCard delay={0.3}>
                <h3 className="text-xl font-bold text-slate-100 mb-4 flex items-center gap-2">
                    <History className="w-5 h-5 text-purple-400" />
                    Recent Transactions
                </h3>
                <div className="overflow-x-auto max-h-64 overflow-y-auto custom-scrollbar">
                    <table className="w-full">
                        <thead className="sticky top-0 bg-slate-900/90 backdrop-blur-sm z-10">
                            <tr className="border-b border-white/10">
                                <th className="text-left py-3 px-4 font-semibold text-slate-300">Date</th>
                                <th className="text-left py-3 px-4 font-semibold text-slate-300">Symbol</th>
                                <th className="text-center py-3 px-4 font-semibold text-slate-300">Action</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Quantity</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Price</th>
                                <th className="text-right py-3 px-4 font-semibold text-slate-300">Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            {portfolio?.transactions?.length > 0 ? (
                                [...portfolio.transactions].reverse().map((tx, index) => (
                                    <tr key={index} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                        <td className="py-3 px-4 text-slate-400 text-sm">
                                            {new Date(tx.date).toLocaleDateString()} {new Date(tx.date).toLocaleTimeString()}
                                        </td>
                                        <td className="py-3 px-4 font-semibold text-white">{tx.symbol}</td>
                                        <td className="py-3 px-4 text-center">
                                            <span className={`px-2 py-1 rounded text-xs font-bold uppercase ${tx.action === 'buy' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                                                }`}>
                                                {tx.action}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-right text-slate-300">{tx.quantity}</td>
                                        <td className="py-3 px-4 text-right text-slate-300">${tx.price.toFixed(2)}</td>
                                        <td className="py-3 px-4 text-right font-semibold text-white">${tx.total.toFixed(2)}</td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="6" className="py-8 text-center text-slate-400">
                                        No transactions yet.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </AnimatedCard>

            {/* Trade Modal */}
            <TradeModal
                isOpen={isTradeModalOpen}
                onClose={() => setIsTradeModalOpen(false)}
                symbol={selectedSymbol}
                currentPrice={currentPrice}
                onTrade={executeTrade}
                holdings={portfolio?.holdings || []}
            />
        </div>
    );
};

export default PortfolioDashboard;
