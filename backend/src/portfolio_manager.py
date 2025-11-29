import json
import os
from datetime import datetime
import pandas as pd

class PortfolioManager:
    def __init__(self, initial_capital=100000.0, storage_file='portfolio.json'):
        self.initial_capital = initial_capital
        self.storage_file = storage_file
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self):
        """Load portfolio from JSON file or create new if not exists"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading portfolio: {e}")
        
        # Default empty portfolio
        return {
            'cash': self.initial_capital,
            'holdings': {},  # symbol -> {quantity, avg_cost}
            'transactions': [],  # list of trade records
            'history': []  # daily value snapshots
        }

    def _save_portfolio(self):
        """Save portfolio to JSON file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.portfolio, f, indent=4)
        except Exception as e:
            print(f"Error saving portfolio: {e}")

    def get_portfolio_state(self, current_prices=None):
        """
        Get current portfolio state including total value and P&L
        
        Args:
            current_prices (dict): Dictionary mapping symbols to current prices
        """
        cash = self.portfolio['cash']
        holdings_value = 0
        holdings_list = []

        for symbol, data in self.portfolio['holdings'].items():
            quantity = data['quantity']
            avg_cost = data['avg_cost']
            current_price = current_prices.get(symbol, avg_cost) if current_prices else avg_cost
            
            market_value = quantity * current_price
            holdings_value += market_value
            
            unrealized_pl = market_value - (quantity * avg_cost)
            unrealized_pl_pct = (unrealized_pl / (quantity * avg_cost)) * 100 if avg_cost > 0 else 0

            holdings_list.append({
                'symbol': symbol,
                'quantity': quantity,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pl': unrealized_pl,
                'unrealized_pl_pct': unrealized_pl_pct
            })

        total_value = cash + holdings_value
        total_pl = total_value - self.initial_capital
        total_pl_pct = (total_pl / self.initial_capital) * 100

        return {
            'cash': cash,
            'holdings_value': holdings_value,
            'total_value': total_value,
            'total_pl': total_pl,
            'total_pl_pct': total_pl_pct,
            'holdings': holdings_list,
            'transactions': self.portfolio['transactions'][-50:]  # Last 50 transactions
        }

    def execute_trade(self, symbol, action, quantity, price, date=None):
        """
        Execute a buy or sell trade
        
        Args:
            symbol (str): Stock symbol
            action (str): 'buy' or 'sell'
            quantity (int): Number of shares
            price (float): Execution price
            date (str): Transaction date (ISO format)
        
        Returns:
            dict: Transaction result or error
        """
        if quantity <= 0:
            return {'success': False, 'error': 'Quantity must be positive'}
        
        if price <= 0:
            return {'success': False, 'error': 'Price must be positive'}

        date = date or datetime.now().isoformat()
        total_cost = quantity * price

        if action.lower() == 'buy':
            if self.portfolio['cash'] < total_cost:
                return {'success': False, 'error': 'Insufficient funds'}
            
            # Update cash
            self.portfolio['cash'] -= total_cost
            
            # Update holdings
            if symbol not in self.portfolio['holdings']:
                self.portfolio['holdings'][symbol] = {'quantity': 0, 'avg_cost': 0}
            
            current_holding = self.portfolio['holdings'][symbol]
            new_quantity = current_holding['quantity'] + quantity
            # Calculate new average cost
            total_prev_cost = current_holding['quantity'] * current_holding['avg_cost']
            new_avg_cost = (total_prev_cost + total_cost) / new_quantity
            
            self.portfolio['holdings'][symbol] = {
                'quantity': new_quantity,
                'avg_cost': new_avg_cost
            }

        elif action.lower() == 'sell':
            if symbol not in self.portfolio['holdings'] or self.portfolio['holdings'][symbol]['quantity'] < quantity:
                return {'success': False, 'error': 'Insufficient shares'}
            
            # Update cash
            self.portfolio['cash'] += total_cost
            
            # Update holdings
            current_holding = self.portfolio['holdings'][symbol]
            new_quantity = current_holding['quantity'] - quantity
            
            if new_quantity == 0:
                del self.portfolio['holdings'][symbol]
            else:
                current_holding['quantity'] = new_quantity
                # Avg cost doesn't change on sell

        else:
            return {'success': False, 'error': 'Invalid action'}

        # Record transaction
        transaction = {
            'date': date,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'total': total_cost
        }
        self.portfolio['transactions'].append(transaction)
        
        self._save_portfolio()
        return {'success': True, 'transaction': transaction}

    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.portfolio = {
            'cash': self.initial_capital,
            'holdings': {},
            'transactions': [],
            'history': []
        }
        self._save_portfolio()
        return self.get_portfolio_state()
