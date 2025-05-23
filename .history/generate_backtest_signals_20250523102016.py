import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_from_csv import Game, GameGymWrapper, Monitor

def reward_function(entry_price, exit_price, position, daily_profit, daily_loss, daily_profit_target=100, daily_max_loss=-50):
    # Trade PnL: positive if profitable, negative if not
    pnl = (exit_price - entry_price) * position  # position = 1 for long, -1 for short

    # Basic reward is realized PnL
    reward = pnl

    # Daily profit bonus
    if daily_profit >= daily_profit_target:
        reward += 10  # bonus for hitting daily target

    # Daily loss penalty
    if daily_loss <= daily_max_loss:
        reward -= 10  # penalty for exceeding daily loss

    return reward


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate trading signals for backtesting RL strategy')
    
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol (default: EURUSD)')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained PPO model zip file')
    parser.add_argument('--vec-normalize-path', type=str, required=True,
                      help='Path to the vector normalization file (.pkl)')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, default='backtest_signals.csv',
                      help='Path to save the generated signals (default: backtest_signals.csv)')
    
    return parser.parse_args()

def clean_signals_dataframe(df):
    """
    Clean the signals dataframe by removing any rows where the timestamp is prior to the previous row.
    This ensures chronological order in the signals.
    
    Args:
        df (pd.DataFrame): DataFrame containing signals with 'time' column
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with only chronologically ordered signals
    """
    if df.empty:
        return df
        
    # Convert time column to datetime if it's not already
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time to ensure chronological order
    df = df.sort_values('time')
    
    # Keep only rows where current time is greater than or equal to previous time
    mask = df['time'] >= df['time'].shift(1)
    mask.iloc[0] = True  # Keep the first row
    
    return df[mask]

def load_mt5_data(symbol='EURUSD', timeframes=['M30', 'H4', 'D1']):
    """
    Load data from MT5 exported CSV files and prepare it for the Game environment.
    
    Args:
        symbol (str): Trading symbol (default: 'EURUSD')
        timeframes (list): List of timeframes to load (default: ['M30', 'H4', 'D1'])
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        count (int): Number of candles to load
        
    Returns:
        tuple: (bars30m, bars4h, bars1d) pandas DataFrames with OHLCV data
    """
    
    # Load data for each timeframe
    data = {}
    for tf in timeframes:
        filename = f"data/{symbol}_{tf.lower()}.csv"
        try:
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            
            
            data[tf] = df
        except Exception as e:
            print(f"Error loading data for {symbol} {tf}: {e}")
            return None, None, None
    
    # Return data in the format expected by Game
    return data['M30'], data['H4'], data['D1']

def main():
    # Parse command line arguments
    args = parse_arguments()

    df30m, df4h, df1d = load_mt5_data(args.symbol, ['M30', 'H4', 'D1'], )
    
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
        env = Game(
            bars30m=df30m,
            bars4h=df4h, 
            bars1d=df1d,
            reward_function=reward_function,
            lkbk=100,
            init_idx= 101
        )
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
                # Clean the signals dataframe before saving
                signals_df = clean_signals_dataframe(signals_df)
                signals_df.to_csv(args.output_file, index=False)
                print(f"Generated {len(signals)} signals")
                print(f"Cleaned signals count: {len(signals_df)}")
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