from flask_socketio import SocketIO, emit
import threading
import time
import random
import json

class WebSocketServer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketServer, cls).__new__(cls)
            cls._instance.socketio = None
            cls._instance.thread = None
            cls._instance.stop_thread = False
            # Initialize with default prices to avoid startup delay
            cls._instance.current_prices = {
                'AAPL': 178.50, 'TSLA': 242.30, 'MSFT': 405.20,
                'GOOGL': 142.65, 'AMZN': 178.25, 'NVDA': 495.80, 'META': 485.90
            }
        return cls._instance

    def init_app(self, app):
        """Initialize SocketIO with Flask app"""
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('connection_response', {'data': 'Connected to Stocker Real-Time API'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')

        @self.socketio.on('subscribe_ticker')
        def handle_subscribe(data):
            symbol = data.get('symbol')
            print(f'Client subscribed to {symbol}')
            # In a real app, we would add this client to a room for this symbol

    def start_broadcasting(self):
        """Start the background thread for data broadcasting"""
        if self.thread is None:
            self.stop_thread = False
            self.thread = threading.Thread(target=self._broadcast_loop)
            self.thread.daemon = True
            self.thread.start()
            print("WebSocket broadcasting started")

    def _broadcast_loop(self):
        """Background loop to simulate real-time data"""
        symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
        
        # Base prices for simulation
        prices = {
            'AAPL': 178.50, 'TSLA': 242.30, 'MSFT': 405.20,
            'GOOGL': 142.65, 'AMZN': 178.25, 'NVDA': 495.80, 'META': 485.90
        }

        while not self.stop_thread:
            # Simulate random price updates
            for symbol in symbols:
                # Random fluctuation between -0.5% and +0.5%
                change_pct = random.uniform(-0.005, 0.005)
                current_price = prices[symbol]
                new_price = current_price * (1 + change_pct)
                prices[symbol] = new_price
                
                # Update shared state
                self.current_prices[symbol] = new_price
                
                update_data = {
                    'symbol': symbol,
                    'price': round(new_price, 2),
                    'change': round(new_price - current_price, 2),
                    'change_pct': round(change_pct * 100, 2),
                    'timestamp': time.time()
                }
                
                if self.socketio:
                    self.socketio.emit('price_update', update_data)
            
            # Broadcast every 2 seconds
            time.sleep(2)

    def get_latest_prices(self):
        """Get the current simulated prices"""
        return self.current_prices.copy()

    def run(self, app, **kwargs):
        """Run the SocketIO app"""
        if self.socketio:
            self.start_broadcasting()
            self.socketio.run(app, **kwargs)
