# MayesMT5 - Reinforcement Learning Trading Bot

This project implements a reinforcement learning-based trading bot for MetaTrader 5, using PyTorch and Stable-Baselines3 for the RL implementation.

## Prerequisites

- Windows 10 or later
- MetaTrader 5 (MT5) platform
- Python 3.8 or later
- Git (optional, for version control)

## Installation Guide

### 1. Install MetaTrader 5

1. Download MT5 from the official website: [MetaTrader 5 Download](https://www.metatrader5.com/en/download)
2. Run the installer and follow the setup wizard
3. Create a demo account or use your existing account
4. Launch MT5 and ensure it's running

### 2. Install Python Dependencies

1. Open Command Prompt or PowerShell
2. Navigate to the project directory:
   ```bash
   cd C:\Users\aryar\OneDrive\Documents\CursorProjects\MayesMT5-private
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
MayesMT5-private/
├── config.py           # Configuration parameters
├── utils.py           # Utility functions
├── rl_signal_writer.py # RL model implementation
├── ma_ribbon.py       # MA ribbon implementation
├── cci_strategy.py    # CCI strategy implementation
├── backtest.py        # Backtesting engine
├── requirements.txt    # Python dependencies
├── models/            # Saved model checkpoints
├── logs/              # Training logs
├── data/              # Market data storage
└── results/           # Backtest results
```

## Usage Guide

### 1. Running the RL Model

1. Ensure MT5 is running and logged in
2. Run the training script:
   ```bash
   python rl_signal_writer.py
   ```

The script will:
- Initialize MT5 connection
- Load historical data
- Train the RL model
- Save model checkpoints
- Generate trading signals

### 2. Training and Inference Flow

1. **Data Collection**:
   - Historical data is fetched from MT5
   - Data is preprocessed and normalized
   - Training/validation split is performed

2. **Model Training**:
   - RL model is trained using PPO algorithm
   - Training progress is logged to TensorBoard
   - Model checkpoints are saved periodically

3. **Inference**:
   - Trained model generates trading signals
   - Signals are validated against risk parameters
   - Trading decisions are executed through MT5

### 3. Configuration

Key parameters can be modified in `config.py`:

- Trading parameters (symbol, timeframe, risk settings)
- RL model parameters (learning rate, batch size, etc.)
- Environment parameters (window size, reward scaling)
- Risk management settings

## Monitoring and Logging

- Training progress can be monitored using TensorBoard:
  ```bash
  tensorboard --logdir=logs/
  ```
- Trading performance metrics are saved in the `results/` directory
- Detailed logs are available in the `logs/` directory

## Troubleshooting

1. **MT5 Connection Issues**:
   - Ensure MT5 is running and logged in
   - Check if the correct account is selected
   - Verify internet connection

2. **Python Package Issues**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility
   - Verify virtual environment if using one

3. **Model Training Issues**:
   - Check GPU availability if using CUDA
   - Monitor memory usage during training
   - Verify data preprocessing steps

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MetaTrader 5 for the trading platform
- Stable-Baselines3 for RL implementation
- PyTorch for deep learning framework

## Strategy Implementation

### Moving Averages (MA) Setup
- Six Simple Moving Averages (SMAs) forming a ribbon:
  - Five SMAs with period 1 and shifts 0-4
  - One SMA with period 50
- Multi-timeframe analysis:
  - 1 minute (entry)
  - 15 minutes (confirmation)
  - 1 hour (trend filter)
- Trade entry requires ribbon formation on all timeframes

### Commodity Channel Index (CCI) Strategy
- Three timeframe monitoring (1m, 15m, 30m)
- Dual CCI calculation per timeframe
- Signal generation:
  - Buy: CCI > 100 and above SMA
  - Sell: CCI < -100 and below SMA
- Higher timeframe confirmation required

### Reinforcement Learning Integration
- Dynamic risk management through RL
- Reward functions:
  - Daily profit targets
  - Maximum daily loss limits
  - Additional customizable rewards/penalties
- Adaptive position sizing
- Dynamic stop-loss and take-profit levels

### Backtesting and Optimization
- MT5 Strategy Tester integration
- Customizable parameters:
  - MA periods and shifts
  - CCI periods
  - Timeframe combinations
  - Risk management settings
- Performance metrics:
  - Profit/Loss
  - Drawdown
  - Win rate
  - Risk-adjusted returns

### Multi-Asset Trading
- Support for multiple currency pairs
- Portfolio-level risk management
- Cross-asset correlation analysis
- Dynamic position sizing across assets

### Continuous Learning
- Real-time model updates
- Periodic retraining
- Adaptive strategy parameters
- Market regime detection

## Implementation Details

### Required Files
```
MayesMT5-private/
├── config.py           # Strategy parameters
├── utils.py           # Utility functions
├── rl_signal_writer.py # RL model implementation
├── ma_ribbon.py       # MA ribbon implementation
├── cci_strategy.py    # CCI strategy implementation
├── backtest.py        # Backtesting engine
├── requirements.txt   # Python dependencies
├── models/           # Saved model checkpoints
├── logs/             # Training logs
├── data/             # Market data storage
└── results/          # Backtest results
```

### Configuration Parameters
```python
# MA Ribbon Parameters
MA_PARAMS = {
    'periods': [1, 1, 1, 1, 1, 50],
    'shifts': [0, 1, 2, 3, 4, 0],
    'timeframes': ['M1', 'M15', 'H1']
}

# CCI Parameters
CCI_PARAMS = {
    'periods': [14, 20],  # Two CCI periods
    'timeframes': ['M1', 'M15', 'M30'],
    'overbought': 100,
    'oversold': -100
}

# RL Parameters
RL_PARAMS = {
    'daily_profit_target': 0.02,  # 2% daily target
    'max_daily_loss': 0.01,      # 1% max daily loss
    'position_size': 0.1,        # 10% of account per trade
    'max_positions': 3           # Maximum concurrent positions
}
``` 