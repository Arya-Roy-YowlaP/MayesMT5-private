import MetaTrader5 as mt5
from datetime import datetime
import os
import glob
import torch
from backtest_framework import Backtest
from game_environment import Game
import numpy as np
from stable_baselines3 import PPO
from unified_train import DQN  # Import DQN class from unified_train

def find_latest_model(model_type="ppo"):
    """Find the latest model file of specified type"""
    if model_type.lower() == "ppo":
        pattern = "models/ppo_model_*"
    else:
        pattern = "models/dqn_model_*.pt"
    
    model_files = glob.glob(pattern)
    if not model_files:
        raise FileNotFoundError(f"No {model_type.upper()} model files found in models/ directory")
    
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def load_model(model_path, model_type="ppo"):
    """Load a trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if model_type.lower() == "ppo":
        return PPO.load(model_path)
    else:  # DQN
        model = DQN(input_dim=30, output_dim=3)  # Assuming window_size=30
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

def initialize_mt5():
    """Initialize MetaTrader 5 connection"""
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return False
    return True

def main():
    # Initialize MT5
    if not initialize_mt5():
        return

    # Parameters
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M1
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Get model type from user
    model_type = input("Enter model type (dqn/ppo): ").lower()
    if model_type not in ["dqn", "ppo"]:
        print("Invalid model type. Please choose 'dqn' or 'ppo'")
        return
    
    try:
        # Find and load the latest model
        print("Finding latest model...")
        model_path = find_latest_model(model_type)
        print(f"Found model: {model_path}")
        
        print("Loading model...")
        model = load_model(model_path, model_type)
        
        # Create backtest instance
        backtest = Backtest(
            model=model,
            initial_balance=10000,
            risk_per_trade=0.01,
            model_type=model_type  # Pass model type to backtest
        )
        
        # Run backtest
        print(f"Running backtest for {symbol} from {start_date} to {end_date}")
        metrics = backtest.run(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print detailed results
        print("\nDetailed Backtest Results:")
        print("-" * 50)
        print(f"Model Type: {model_type.upper()}")
        print(f"Model Path: {model_path}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        
        # Save results to file
        results_file = f"results/backtest_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("results", exist_ok=True)
        with open(results_file, "w") as f:
            f.write("Backtest Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model Type: {model_type.upper()}\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
        print(f"\nResults saved to {results_file}")
        
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
    finally:
        # Shutdown MT5
        mt5.shutdown()

if __name__ == "__main__":
    main() 