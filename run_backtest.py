import MetaTrader5 as mt5
from datetime import datetime
from backtest_framework import Backtest
from game_environment import Game
import numpy as np
from stable_baselines3 import PPO
import os

def load_model(model_path):
    """Load a trained PPO model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return PPO.load(model_path)

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
    model_path = "models/ppo_trading_model"  # Update this path to your trained model
    
    try:
        # Load the trained model
        print("Loading model...")
        model = load_model(model_path)
        
        # Create backtest instance
        backtest = Backtest(
            model=model,
            initial_balance=10000,
            risk_per_trade=0.01
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
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
    finally:
        # Shutdown MT5
        mt5.shutdown()

if __name__ == "__main__":
    main() 