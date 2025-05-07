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
    try:
        base_env = CSVGameEnv(csv_path=args.data_file, window_size=30)
        env = Monitor(base_env)
        env = DummyVecEnv([lambda: env])
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
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
    truncated = False
    
    # Initialize balance tracking
    initial_balance = 10000  # Starting with $10,000
    current_balance = initial_balance
    position = 0
    entry_price = 0
    last_processed_idx = 30  # Start after window_size

    print(f"Generating signals for {args.data_file}")
    print(f"Total candles: {len(df)}")
    print(f"Initial balance: ${initial_balance:,.2f}")

    try:
        while True:
            try:
                action, _ = model.predict(obs, deterministic=True)
            except Exception as e:
                print(f"Error predicting action: {e}")
                break

            try:
                step_result = env.step(action)
                # Handle both old (4 values) and new (5 values) gym API
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
            except Exception as e:
                print(f"Error stepping environment: {e}")
                break
            
            # Convert action to signal
            signal = ["HOLD", "SELL", "BUY"][action[0]]
            
            try:
                current_idx = env.envs[0].env.current_idx
                if current_idx < len(df):
                    timestamp = df.index[current_idx]
                    current_price = df['close'].iloc[current_idx]
                    
                    # Update balance based on position and price changes
                    if position != 0:
                        pnl = (current_price - entry_price) * position
                        current_balance += pnl
                    
                    # Update position if action is taken
                    if action[0] == 2 and position <= 0:  # BUY
                        position = 1
                        entry_price = current_price
                    elif action[0] == 1 and position >= 0:  # SELL
                        position = -1
                        entry_price = current_price
                    
                    # Store signal with timestamp and balance
                    signals.append({
                        'time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'signal': signal,
                        'balance': current_balance,
                        'position': position,
                        'price': current_price
                    })
                    
                    # Log if this is a new day
                    if timestamp.hour == 0 and timestamp.minute == 0:
                        print(f"Processing new day: {timestamp.date()} - Balance: ${current_balance:,.2f}")
                    
                    # If done, check reason and handle accordingly
                    if done:
                        # Get the reason from the base environment
                        base_env = env.envs[0].env
                        # Handle info being a list
                        info_dict = info[0] if isinstance(info, list) else info
                        if info_dict and info_dict.get('reason') == "End of data":
                            print(f"Reached end of data at {timestamp}")
                            print(f"Final balance: ${current_balance:,.2f}")
                            break
                        else:
                            print(f"Environment reset at {timestamp}")
                            print(f"Previous balance: ${current_balance:,.2f}")
                            # Reset environment but preserve the current index
                            obs = env.reset()
                            # Set the current index to continue from where we left off
                            base_env.current_idx = last_processed_idx
                            # Keep the current balance and position
                            done = False
                    
                    last_processed_idx = current_idx
                    
            except Exception as e:
                print(f"Error processing step: {e}")
                print("Environment structure:", type(env), type(env.envs[0]), type(env.envs[0].env))
                break

    finally:
        # Save signals to CSV
        try:
            if signals:  # Only save if we have signals
                signals_df = pd.DataFrame(signals)
                signals_df.to_csv(args.output_file, index=False)
                print(f"Generated {len(signals)} signals")
                print(f"Signals saved to: {args.output_file}")
                print(f"Final balance: ${current_balance:,.2f}")
            else:
                print("No signals were generated")
        except Exception as e:
            print(f"Error saving signals: {e}")
        
        # Close environment
        try:
            env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")

if __name__ == "__main__":
    main() 