import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import torch

print("Starting script...")

# --- Replace this with your model path and logic ---
class DummyRLModel:
    def predict(self, state):
        # Placeholder logic: Random action
        print("Generating prediction...")
        return np.random.choice([0, 1, 2])  # 0: Hold, 1: Sell, 2: Buy

# Initialize MT5
print("Initializing MT5...")
if not mt5.initialize():
    print("MT5 initialization failed:", mt5.last_error())
    quit()
print("MT5 initialized successfully")

symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
print(f"Fetching data for {symbol}...")
rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
print(f"Data fetched successfully. Shape: {df.shape}")

# Dummy Game wrapper
class SimpleGame:
    def __init__(self, bars, model):
        self.bars = bars
        self.model = model
        self.current_idx = 50  # skip some bars for indicators
        self.position = 0
        self.entry_price = 0
        print("Game initialized")

    def get_state(self):
        # Example state using normalized close prices
        window = self.bars['close'].iloc[self.current_idx-10:self.current_idx].values
        state = (window - window.mean()) / (window.std() + 1e-6)
        return state

    def step(self):
        print("Getting state...")
        state = self.get_state()
        print("Predicting action...")
        action = self.model.predict(state)

        # Save the action to file
        print("Saving action to file...")
        with open("C:\\Users\\Public\\Documents\\rl_action.txt", "w") as f:
            if action == 2:
                f.write("BUY")
            elif action == 1:
                f.write("SELL")
            else:
                f.write("HOLD")

        print("Action:", ["HOLD", "SELL", "BUY"][action])

# Initialize model and run decision loop once (or loop this)
print("Initializing model...")
model = DummyRLModel()
game = SimpleGame(df, model)
print("Running step...")
game.step()

print("Shutting down MT5...")
mt5.shutdown()
print("Script completed")