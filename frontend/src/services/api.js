import axios from 'axios';
import { io } from 'socket.io-client';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Initialize Socket.IO
export const socket = io('http://localhost:5000', {
  transports: ['websocket'],
  autoConnect: true
});

socket.on('connect', () => {
  console.log('Connected to WebSocket server');
});

socket.on('disconnect', () => {
  console.log('Disconnected from WebSocket server');
});

export const stockAPI = {
  // Get stock data
  getStockData: async (symbol, period = '2y') => {
    const response = await api.post('/stock/data', { symbol, period });
    return response.data;
  },

  // Make prediction
  predict: async (params) => {
    const response = await api.post('/predict', params);
    return response.data;
  },

  // Compare models
  compareModels: async (params) => {
    const response = await api.post('/compare-models', params);
    return response.data;
  },

  // Run backtest
  runBacktest: async (params) => {
    const response = await api.post('/backtest', params);
    return response.data;
  },

  // Get chart data
  getChartData: async (symbol, period = '6mo') => {
    const response = await api.post('/chart-data', { symbol, period });
    return response.data;
  },

  // Get sentiment analysis
  getSentiment: async (symbol) => {
    const response = await api.get('/sentiment', {
      params: { symbol }
    });
    return response.data;
  },

  // Portfolio Management
  getPortfolio: async () => {
    const response = await api.get('/portfolio');
    return response.data;
  },

  executeTrade: async (symbol, action, quantity) => {
    const response = await api.post('/portfolio/trade', { symbol, action, quantity });
    return response.data;
  },

  resetPortfolio: async () => {
    const response = await api.post('/portfolio/reset');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
