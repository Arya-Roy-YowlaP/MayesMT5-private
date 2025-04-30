import MetaTrader5 as mt5

# Trading Parameters
TRADING_PARAMS = {
    'symbol': "EURUSD",
    'timeframe': mt5.TIMEFRAME_M1,
    'initial_balance': 10000,
    'risk_per_trade': 0.01,
    'spread': 0.0002,  # 2 pips
    'commission': 0.0001,  # 1 pip
}

# RL Model Parameters
MODEL_PARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,  # Discount factor for future rewards
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'total_timesteps': 1000000,
}

# Environment Parameters
ENV_PARAMS = {
    'window_size': 30,  # Number of candles to look back
    'reward_scaling': 100,  # Scale factor for base PnL rewards
    'max_position_size': 1.0,  # Maximum position size in lots
    'stop_loss_pips': 20,  # Stop loss in pips
    'take_profit_pips': 40,  # Take profit in pips
    'max_position_hold_time': 100,  # Maximum steps to hold a position
    'trend_weights': {
        'short_term': 0.2,  # 1m timeframe weight
        'medium_term': 0.3,  # 15m timeframe weight
        'long_term': 0.5,   # 1h timeframe weight
    },
    'reward_weights': {
        'pnl': 1.0,         # Base PnL weight
        'duration': 0.3,    # Duration penalty weight
        'volatility': 0.4,  # Volatility factor weight
        'trend': 0.3,       # Trend alignment weight
        'risk': 0.4,        # Risk management weight
    }
}

# Backtest Parameters
BACKTEST_PARAMS = {
    'start_date': "2023-01-01",
    'end_date': "2023-12-31",
    'train_test_split': 0.8,  # 80% training, 20% testing
}

# File Paths
PATHS = {
    'models_dir': "models",
    'logs_dir': "logs",
    'data_dir': "data",
    'results_dir': "results"
}

# Risk Management
RISK_PARAMS = {
    'max_daily_loss': 0.02,     # 2% max daily loss
    'max_drawdown': 0.1,        # 10% max drawdown
    'max_open_trades': 3,       # Maximum concurrent positions
    'min_win_rate': 0.4,        # Minimum acceptable win rate
    'min_profit_factor': 1.5,   # Minimum profit factor
    'risk_reward_target': 2.0,  # Target risk-reward ratio
    'position_sizing': {
        'base_size': 0.01,      # Base position size (1% of account)
        'max_size': 0.05,       # Maximum position size (5% of account)
        'scaling_factor': 0.5,   # Position size scaling based on confidence
    },
    'drawdown_scaling': {
        'threshold': 0.05,       # 5% drawdown threshold
        'reduction_factor': 0.5, # Reduce position size by half when exceeded
    }
} 