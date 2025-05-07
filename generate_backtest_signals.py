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
    processed_indices = set()  # Keep track of processed indices
    last_processed_idx = env.envs[0].env.window_size  # Start after window_size

    print(f"Generating signals for {args.data_file}")
    print(f"Total candles: {len(df)}")
    print("Starting signal generation...")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        
        # Convert action to signal
        signal = ["HOLD", "SELL", "BUY"][action[0]]
        
        # Get current timestamp - access the underlying environment
        try:
            # Access the base environment through the wrapper chain
            current_idx = env.envs[0].env.current_idx
            if current_idx < len(df) and current_idx not in processed_indices:
                timestamp = df.index[current_idx]
                processed_indices.add(current_idx)
                last_processed_idx = current_idx
                
                # Store signal with timestamp
                signals.append({
                    'time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'signal': signal
                })

                # Print progress every 1000 candles
                if len(signals) % 1000 == 0:
                    progress = (len(signals) / len(df)) * 100
                    print(f"Progress: {len(signals)}/{len(df)} signals generated ({progress:.1f}%)")
                    print(f"Current date: {timestamp}")

        except Exception as e:
            print(f"Error accessing current_idx: {e}")
            print("Environment structure:", type(env), type(env.envs[0]), type(env.envs[0].env))
            break

        if done:
            # Check if we've reached the end of data
            if current_idx >= len(df) - 1:
                print("\nReached end of data. Finalizing...")
                break
            else:
                # Get the reason from the base environment's info
                base_env = env.envs[0].env
                reason = "unknown"
                if hasattr(base_env, 'last_info'):
                    reason = base_env.last_info.get('reason', 'unknown')
                print(f"\nResetting environment at {timestamp} (reason: {reason})")
                
                # Reset environment but maintain the current index
                base_env.current_idx = last_processed_idx + 1
                obs = env.reset()

    # Save signals to CSV
    try:
        signals_df = pd.DataFrame(signals)
        # Sort by timestamp to ensure chronological order
        signals_df['time'] = pd.to_datetime(signals_df['time'])
        signals_df = signals_df.sort_values('time')
        signals_df.to_csv(args.output_file, index=False)
        print(f"\nGenerated {len(signals)} signals")
        print(f"Signals saved to: {args.output_file}")
    except Exception as e:
        print(f"Error saving signals: {e}")

if __name__ == "__main__":
    main() 