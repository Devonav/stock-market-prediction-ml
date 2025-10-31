// Free stock ticker service using Finnhub API
// Get your free API key at: https://finnhub.io/register

// Option 1: Use environment variable (create .env file with VITE_FINNHUB_API_KEY=your_key)
// Option 2: Replace 'YOUR_API_KEY_HERE' below with your actual API key
const FINNHUB_API_KEY = import.meta.env.VITE_FINNHUB_API_KEY || 'YOUR_API_KEY_HERE';
const FINNHUB_BASE_URL = 'https://finnhub.io/api/v1';

// Fallback to a public proxy if CORS issues occur
const PROXY_URL = 'https://api.allorigins.win/raw?url=';

export const stockTickerService = {
  /**
   * Fetch real-time quote for a single stock
   */
  async getQuote(symbol) {
    try {
      const url = `${FINNHUB_BASE_URL}/quote?symbol=${symbol}&token=${FINNHUB_API_KEY}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to fetch ${symbol}`);
      }

      const data = await response.json();

      return {
        symbol,
        price: data.c, // current price
        change: data.d, // change
        percentChange: data.dp, // percent change
        high: data.h, // high price of the day
        low: data.l, // low price of the day
        open: data.o, // open price of the day
        previousClose: data.pc, // previous close price
      };
    } catch (error) {
      console.error(`Error fetching ${symbol}:`, error);
      return null;
    }
  },

  /**
   * Fetch quotes for multiple stocks
   */
  async getMultipleQuotes(symbols) {
    const promises = symbols.map(symbol => this.getQuote(symbol));
    const results = await Promise.all(promises);
    return results.filter(result => result !== null);
  },

  /**
   * Format stock data for marquee display
   */
  formatForMarquee(stockData) {
    if (!stockData || !stockData.price) return null;

    const isPositive = stockData.change >= 0;
    const arrow = isPositive ? '↑' : '↓';
    const colorClass = isPositive ? 'text-green-400' : 'text-red-400';

    return {
      symbol: stockData.symbol,
      price: stockData.price.toFixed(2),
      change: stockData.change.toFixed(2),
      percentChange: Math.abs(stockData.percentChange).toFixed(1),
      arrow,
      colorClass,
      isPositive
    };
  }
};

// Popular stock symbols to track
export const DEFAULT_TICKER_SYMBOLS = [
  'AAPL',  // Apple
  'TSLA',  // Tesla
  'MSFT',  // Microsoft
  'NVDA',  // NVIDIA
  'GOOGL', // Google
  'AMZN',  // Amazon
  'META'   // Meta
];
