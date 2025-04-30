import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime

class Game:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, window_size=10):
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.current_idx = window_size
        self.position = 0
        self.entry_price = 0
        self.position_hold_time = 0
        self.max_position_hold_time = 100  # Maximum number of steps to hold a position
        self.profit_threshold = 0.02  # 2% profit target
        self.loss_threshold = -0.01  # 1% stop loss
        
        # Initialize MT5 if not already initialized
        if not mt5.initialize():
            print("MT5 initialization failed:", mt5.last_error())
            quit()
            
        # Fetch initial data
        self._fetch_data()
        
    def _fetch_data(self):
        rates = mt5.copy_rates_from(self.symbol, self.timeframe, datetime.now(), 1000)
        if rates is None or len(rates) == 0:
            print("No data available")
            return False
        self.df = pd.DataFrame(rates)
        self.df['time'] = pd.to_datetime(self.df['time'], unit='s')
        self.df.set_index('time', inplace=True)
        return True
        
    def get_state(self):
        # Get the window of data
        window = self.df['close'].iloc[self.current_idx-self.window_size:self.current_idx].values
        
        # Normalize the data
        state = (window - window.mean()) / (window.std() + 1e-6)
        return state
    
    def step(self, action):
        """
        action: 0 (HOLD), 1 (SELL), 2 (BUY)
        """
        # Get current price
        current_price = self.df['close'].iloc[self.current_idx]
        
        # Calculate reward based on action and position
        reward = 0
        if action == 2 and self.position <= 0:  # BUY
            self.position = 1
            self.entry_price = current_price
            self.position_hold_time = 0
            reward = 0  # No immediate reward for entering position
        elif action == 1 and self.position >= 0:  # SELL
            self.position = -1
            self.entry_price = current_price
            self.position_hold_time = 0
            reward = 0  # No immediate reward for entering position
        elif action == 0:  # HOLD
            if self.position > 0:  # Long position
                reward = (current_price - self.entry_price) / self.entry_price
            elif self.position < 0:  # Short position
                reward = (self.entry_price - current_price) / self.entry_price
            self.position_hold_time += 1
        
        # Move to next time step
        self.current_idx += 1
        
        # Check if we need more data
        if self.current_idx >= len(self.df) - 1:
            if not self._fetch_data():
                return self.get_state(), reward, True
            self.current_idx = self.window_size
        
        # Get next state
        next_state = self.get_state()
        
        # Check if episode should end
        done = False
        
        # End episode if we've held a position too long
        if self.position != 0 and self.position_hold_time >= self.max_position_hold_time:
            done = True
            print(f"Episode ended: Position held too long ({self.position_hold_time} steps)")
        
        # End episode if we've hit profit target
        if reward >= self.profit_threshold:
            done = True
            print(f"Episode ended: Profit target reached ({reward*100:.2f}%)")
        
        # End episode if we've hit stop loss
        if reward <= self.loss_threshold:
            done = True
            print(f"Episode ended: Stop loss hit ({reward*100:.2f}%)")
        
        return next_state, reward, done
    
    def reset(self):
        self.current_idx = self.window_size
        self.position = 0
        self.entry_price = 0
        self.position_hold_time = 0
        if not self._fetch_data():
            return None
        return self.get_state()
    
    def close(self):
        mt5.shutdown() 