import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import argparse

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

def get_state(data):
    # Example: normalize last 30 close prices
    closes = data['close'].iloc[-30:].values
    state = (closes - closes.mean()) / (closes.std() + 1e-6)
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate trading signals for backtesting RL strategy')
    
    # Trading parameters
    parser.add_argument('--symbol', type=str, default='EURUSD',
                      help='Trading symbol (default: EURUSD)')
    parser.add_argument('--timeframe', type=str, default='M1',
                      help='Timeframe (M1, M5, M15, M30, H1, H4, D1) (default: M1)')
    
    # Date range
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                      help='Start date in YYYY-MM-DD format (default: 2023-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                      help='End date in YYYY-MM-DD format (default: 2024-01-01)')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, default='dqn_model.pt',
                      help='Path to the trained model file (default: dqn_model.pt)')
    parser.add_argument('--input-dim', type=int, default=30,
                      help='Input dimension for the model (default: 30)')
    parser.add_argument('--output-dim', type=int, default=3,
                      help='Output dimension for the model (default: 3)')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, 
                      default='C:/Users/Public/Documents/backtest_signals.csv',
                      help='Path to save the generated signals (default: C:/Users/Public/Documents/backtest_signals.csv)')
    
    return parser.parse_args()

def get_timeframe_enum(timeframe_str):
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    return timeframes.get(timeframe_str.upper(), mt5.TIMEFRAME_M1)

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return

    # Load model
    model = DQN(input_dim=args.input_dim, output_dim=args.output_dim)
    try:
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        mt5.shutdown()
        return

    # Convert dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        mt5.shutdown()
        return

    # Get timeframe
    timeframe = get_timeframe_enum(args.timeframe)

    # Get historical data
    try:
        rates = mt5.copy_rates_range(args.symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            print("No data received from MT5")
            mt5.shutdown()
            return
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
    except Exception as e:
        print(f"Error getting historical data: {e}")
        mt5.shutdown()
        return

    # Generate signals
    signals = []
    window_size = args.input_dim

    print(f"Generating signals for {args.symbol} from {start_date} to {end_date}")
    print(f"Total candles: {len(df)}")

    for i in range(window_size, len(df)):
        # Get data window
        window_data = df.iloc[i-window_size:i+1]
        
        # Get state and predict
        state = get_state(window_data)
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        
        # Convert action to signal
        signal = ["HOLD", "SELL", "BUY"][action]
        
        # Store signal with timestamp
        signals.append({
            'time': window_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'signal': signal
        })

    # Save signals to CSV
    try:
        signals_df = pd.DataFrame(signals)
        signals_df.to_csv(args.output_file, index=False)
        print(f"Generated {len(signals)} signals")
        print(f"Signals saved to: {args.output_file}")
    except Exception as e:
        print(f"Error saving signals: {e}")

    mt5.shutdown()

if __name__ == "__main__":
    main() 