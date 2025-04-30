import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import torch
import torch.nn as nn

# --- Replace with your actual model class and checkpoint ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Load your trained model ---
model = DQN(input_dim=30, output_dim=3)  # Adjust input_dim based on your state shape
model.load_state_dict(torch.load("dqn_model.pt"))
model.eval()

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed:", mt5.last_error())
    quit()

symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# --- Generate state vector from market data ---
def get_state(data):
    # Example: normalize last 30 close prices
    closes = data['close'].iloc[-30:].values
    state = (closes - closes.mean()) / (closes.std() + 1e-6)
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

state = get_state(df)

# --- Predict action ---
with torch.no_grad():
    q_values = model(state)
    action = torch.argmax(q_values).item()

# --- Save action to file for EA to read ---
with open("C:/Users/Public/Documents/rl_action.txt", "w") as f:
    if action == 2:
        f.write("BUY")
    elif action == 1:
        f.write("SELL")
    else:
        f.write("HOLD")

print("Action:", ["HOLD", "SELL", "BUY"][action])

mt5.shutdown()