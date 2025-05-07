import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_from_csv import CSVGameEnv, Monitor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate trading signals for backtesting RL strategy')
    
    # Data parameters
    parser.add_argument('--data-file', type=str, required=True,
                      help='Path to the CSV data file for backtesting')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained PPO model zip file')
    parser.add_argument('--vec-normalize-path', type=str, required=True,
                      help='Path to the vector normalization file (.pkl)')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, default='backtest_signals.csv',
                      help='Path to save the generated signals (default: backtest_signals.csv)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load data
    try:
        df = pd.read_csv(args.data_file)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create environment
    base_env = CSVGameEnv(csv_path=args.data_file, window_size=30)
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])
    
    # Load vector normalization
    try:
        env = VecNormalize.load(args.vec_normalize_path, env)
        env.training = False  # Disable training mode
        env.norm_reward = False  # Don't normalize rewards during evaluation
    except Exception as e:
        print(f"Error loading vector normalization: {e}")
        return

    # Load model
    try:
        model = PPO.load(args.model_path, env=env, device='cpu')  # Force CPU usage
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Generate signals
    signals = []
    obs = env.reset()
    done = False
    last_balance = None

    print(f"Generating signals for {args.data_file}")
    print(f"Total candles: {len(df)}")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Convert action to signal
        signal = ["HOLD", "SELL", "BUY"][action[0]]
        
        # Get current timestamp and balance
        try:
            current_idx = env.envs[0].env.current_idx
            if current_idx < len(df):
                timestamp = df.index[current_idx]
                current_balance = env.envs[0].env.balance
                
                # Store signal with timestamp
                signals.append({
                    'time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'signal': signal,
                    'balance': current_balance
                })
                
                # Log if this is a new day
                if timestamp.hour == 0 and timestamp.minute == 0:
                    print(f"Processing new day: {timestamp.date()}")
                
                # If done, check reason and handle accordingly
                if done:
                    if info.get('reason') == "End of data":
                        print(f"Reached end of data at {timestamp}")
                        break
                    else:
                        print(f"Environment reset at {timestamp} - Reason: {info.get('reason')}")
                        print(f"Previous balance: {current_balance}")
                        # Reset environment but preserve the current index
                        obs = env.reset()
                        # Set the balance to the last known balance
                        env.envs[0].env.balance = current_balance
                        print(f"Reset complete - New balance: {env.envs[0].env.balance}")
                        done = False
                
                last_balance = current_balance
                
        except Exception as e:
            print(f"Error during signal generation: {e}")
            print("Environment structure:", type(env), type(env.envs[0]), type(env.envs[0].env))
            break

    # Save signals to CSV
    try:
        signals_df = pd.DataFrame(signals)
        signals_df.to_csv(args.output_file, index=False)
        print(f"Generated {len(signals)} signals")
        print(f"Signals saved to: {args.output_file}")
        print(f"Final balance: {last_balance}")
    except Exception as e:
        print(f"Error saving signals: {e}")

if __name__ == "__main__":
    main() 