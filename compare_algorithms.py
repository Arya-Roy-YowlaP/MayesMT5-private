import os
import torch
import numpy as np
from datetime import datetime
import MetaTrader5 as mt5
from stable_baselines3 import PPO
from backtest_framework import Backtest
from game_environment import Game
from utils import setup_logging, create_directories, load_historical_data
from config import TRADING_PARAMS, BACKTEST_PARAMS, PATHS
from visualization import (
    plot_algorithm_comparison,
    plot_equity_curves,
    plot_trade_distribution,
    create_performance_report
)

class DQNModel:
    def __init__(self, model_path):
        from unified_train import DQN
        self.model = DQN(input_dim=30, output_dim=3)  # Assuming window_size=30
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

def load_models(dqn_path, ppo_path):
    """Load trained models"""
    dqn_model = DQNModel(dqn_path)
    ppo_model = PPO.load(ppo_path)
    return dqn_model, ppo_model

def run_backtest(model, algorithm, env, logger):
    """Run backtest for a specific model"""
    backtest = Backtest(
        model=model,
        initial_balance=TRADING_PARAMS['initial_balance'],
        risk_per_trade=TRADING_PARAMS['risk_per_trade']
    )
    
    logger.info(f"Running {algorithm} backtest...")
    metrics = backtest.run(
        symbol=TRADING_PARAMS['symbol'],
        timeframe=TRADING_PARAMS['timeframe'],
        start_date=datetime.strptime(BACKTEST_PARAMS['start_date'], "%Y-%m-%d"),
        end_date=datetime.strptime(BACKTEST_PARAMS['end_date'], "%Y-%m-%d")
    )
    
    return metrics, backtest.results.trades, backtest.results.equity_curve

def main():
    # Setup
    logger = setup_logging('algorithm_comparison')
    create_directories()
    
    # Initialize MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return
    
    try:
        # Get model paths
        models_dir = PATHS['models_dir']
        dqn_models = [f for f in os.listdir(models_dir) if f.startswith('dqn_model_')]
        ppo_models = [f for f in os.listdir(models_dir) if f.startswith('ppo_model_')]
        
        if not dqn_models or not ppo_models:
            raise ValueError("No trained models found. Please train models first.")
        
        # Use most recent models
        dqn_path = os.path.join(models_dir, sorted(dqn_models)[-1])
        ppo_path = os.path.join(models_dir, sorted(ppo_models)[-1])
        
        # Load models
        logger.info("Loading trained models...")
        dqn_model, ppo_model = load_models(dqn_path, ppo_path)
        
        # Create environment
        env = Game(
            symbol=TRADING_PARAMS['symbol'],
            timeframe=TRADING_PARAMS['timeframe'],
            window_size=30
        )
        
        # Run backtests
        dqn_metrics, dqn_trades, dqn_equity = run_backtest(dqn_model, "DQN", env, logger)
        ppo_metrics, ppo_trades, ppo_equity = run_backtest(ppo_model, "PPO", env, logger)
        
        # Generate visualizations
        logger.info("Generating comparison visualizations...")
        plot_algorithm_comparison(dqn_metrics, ppo_metrics)
        plot_equity_curves(dqn_equity, ppo_equity)
        plot_trade_distribution(dqn_trades, ppo_trades)
        
        # Create performance report
        report = create_performance_report(dqn_metrics, ppo_metrics)
        logger.info("\nPerformance Report:")
        logger.info(report.to_string())
        
        logger.info("Comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main() 