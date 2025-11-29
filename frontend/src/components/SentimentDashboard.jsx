import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, MessageCircle, Newspaper, Share2 } from 'lucide-react';
import { AnimatedCard, MetricCard } from './AnimatedCard';

const SentimentDashboard = ({ sentimentData, loading }) => {
    if (loading) {
        return (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-pulse">
                <div className="h-64 bg-white/5 rounded-xl backdrop-blur-sm"></div>
                <div className="h-64 bg-white/5 rounded-xl backdrop-blur-sm"></div>
            </div>
        );
    }

    if (!sentimentData) return null;

    const getSentimentColor = (score) => {
        if (score >= 0.2) return 'text-emerald-500';
        if (score <= -0.2) return 'text-red-500';
        return 'text-amber-500';
    };

    const getSentimentBg = (score) => {
        if (score >= 0.2) return 'bg-emerald-500';
        if (score <= -0.2) return 'bg-red-500';
        return 'bg-amber-500';
    };

    const getSentimentIcon = (score) => {
        if (score >= 0.2) return <TrendingUp className="w-8 h-8 text-emerald-500" />;
        if (score <= -0.2) return <TrendingDown className="w-8 h-8 text-red-500" />;
        return <Minus className="w-8 h-8 text-amber-500" />;
    };

    return (
        <div className="space-y-6">
            {/* Main Sentiment Score */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <AnimatedCard delay={0.1} className="md:col-span-2">
                    <h3 className="text-xl font-bold text-slate-100 mb-4 flex items-center gap-2">
                        <Share2 className="w-5 h-5 text-blue-400" />
                        Market Sentiment Analysis
                    </h3>

                    <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                        <div className="text-center flex-1">
                            <div className="text-sm text-slate-400 mb-2">Overall Sentiment Score</div>
                            <div className={`text-5xl font-bold mb-2 ${getSentimentColor(sentimentData.score)}`}>
                                {sentimentData.score > 0 ? '+' : ''}{sentimentData.score}
                            </div>
                            <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold text-white ${getSentimentBg(sentimentData.score)}`}>
                                {sentimentData.label}
                            </div>
                        </div>

                        <div className="flex-1 w-full">
                            {/* Simple Gauge Visualization */}
                            <div className="relative h-4 bg-white/10 rounded-full overflow-hidden">
                                <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-white/30 z-10"></div>
                                <motion.div
                                    initial={{ width: '0%' }}
                                    animate={{
                                        width: `${Math.abs(sentimentData.score * 50)}%`,
                                        left: sentimentData.score >= 0 ? '50%' : `calc(50% - ${Math.abs(sentimentData.score * 50)}%)`
                                    }}
                                    className={`absolute top-0 bottom-0 ${getSentimentBg(sentimentData.score)}`}
                                />
                            </div>
                            <div className="flex justify-between text-xs text-slate-400 mt-2">
                                <span>Bearish (-1.0)</span>
                                <span>Neutral (0.0)</span>
                                <span>Bullish (+1.0)</span>
                            </div>
                        </div>
                    </div>
                </AnimatedCard>

                <MetricCard
                    title="Social Volume"
                    value={sentimentData.social_analysis.mentions.toLocaleString()}
                    icon={MessageCircle}
                    trend={sentimentData.social_analysis.trending ? "+15% Trending" : "Normal Activity"}
                    trendUp={sentimentData.social_analysis.trending}
                    delay={0.2}
                />
            </div>

            {/* News & Social Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <AnimatedCard delay={0.3}>
                    <h3 className="text-lg font-bold text-slate-100 mb-4 flex items-center gap-2">
                        <Newspaper className="w-5 h-5 text-slate-400" />
                        Recent News Headlines
                    </h3>
                    <div className="space-y-4">
                        {sentimentData.news_analysis.top_headlines.map((headline, idx) => (
                            <div key={idx} className="p-3 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-colors">
                                <p className="text-slate-300 font-medium text-sm">{headline}</p>
                            </div>
                        ))}
                    </div>
                    <div className="mt-4 text-xs text-slate-500 text-right">
                        Based on {sentimentData.news_analysis.article_count} analyzed articles
                    </div>
                </AnimatedCard>

                <AnimatedCard delay={0.4}>
                    <h3 className="text-lg font-bold text-slate-100 mb-4 flex items-center gap-2">
                        <MessageCircle className="w-5 h-5 text-teal-400" />
                        Social Media Sentiment
                    </h3>

                    <div className="space-y-6">
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-400">News Sentiment</span>
                                <span className="font-semibold text-slate-100">{sentimentData.news_analysis.score}</span>
                            </div>
                            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${(sentimentData.news_analysis.score + 1) * 50}%` }}
                                    className="h-full bg-blue-500"
                                />
                            </div>
                        </div>

                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-400">Social Sentiment</span>
                                <span className="font-semibold text-slate-100">{sentimentData.social_analysis.score}</span>
                            </div>
                            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${(sentimentData.social_analysis.score + 1) * 50}%` }}
                                    className="h-full bg-teal-500"
                                />
                            </div>
                        </div>

                        <div className="p-4 bg-blue-500/20 border border-blue-500/30 rounded-lg text-sm text-blue-200">
                            <p>
                                <strong>Insight:</strong>
                                {sentimentData.score > 0.3
                                    ? " Strong positive sentiment across both news and social channels suggests bullish momentum."
                                    : sentimentData.score < -0.3
                                        ? " Negative sentiment indicates potential downside risk or market fear."
                                        : " Mixed signals from news and social media suggest market indecision."}
                            </p>
                        </div>
                    </div>
                </AnimatedCard>
            </div>
        </div>
    );
};

export default SentimentDashboard;
