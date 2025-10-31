import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
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

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
