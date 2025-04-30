# MayesMT5 - Reinforcement Learning Trading Bot

This project implements a reinforcement learning-based trading bot for MetaTrader 5, using PyTorch and Stable-Baselines3 for the RL implementation.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MayesMT5-private.git
   cd MayesMT5-private
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install all necessary packages including:
   - MetaTrader5
   - PyTorch
   - Stable-Baselines3
   - Pandas
   - NumPy
   - Other required libraries

3. Ensure MetaTrader 5 is installed and running

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

## Training Parameters

### PPO Model Configuration
The project uses Proximal Policy Optimization (PPO) with the following parameters:

```python
MODEL_PARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 512,              # Steps per environment per update
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,               # Discount factor
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'total_timesteps': 50000,    # Total training steps
}
```

### Training Time Estimates
- **Quick Test Run** (Current Settings):
  - Total timesteps: 50,000
  - Steps per update: 512
  - Estimated time: 8-42 minutes
  - Suitable for testing and development

- **Full Training Run** (Recommended for Production):
  - Total timesteps: 1,000,000
  - Steps per update: 2,048
  - Estimated time: 2.8-14 hours
  - Better model performance

### Environment Settings
```python
ENV_PARAMS = {
    'window_size': 30,           # Lookback period
    'reward_scaling': 100,       # PnL reward scaling
    'max_position_size': 1.0,    # Maximum position size
    'stop_loss_pips': 20,        # Stop loss
    'take_profit_pips': 40,      # Take profit
    'max_position_hold_time': 100 # Maximum hold time
}
```

### Reward Structure
The model is trained using a multi-component reward system:
- Base PnL (weight: 1.0)
- Duration penalty (weight: 0.3)
- Volatility factor (weight: 0.4)
- Trend alignment (weight: 0.3)
- Risk management (weight: 0.4)

### Training Progress
- Monitor training progress using TensorBoard
- Model checkpoints are saved periodically
- Training logs include:
  - Episode rewards
  - Loss values
  - Policy entropy
  - Value function loss

## Project Structure

```
MayesMT5-private/
├── config.py           # Configuration parameters
├── utils.py           # Utility functions
├── rl_signal_writer_torch.py # RL model implementation
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

### 1. Training the RL Model

1. Ensure MT5 is running and logged in
2. Train the RL model:
   ```bash
   python train_rl_model.py
   ```

The training script will:
- Initialize MT5 connection
- Create and configure the training environment
- Train the PPO model
- Save model checkpoints to the `models/` directory
- Save training logs to TensorBoard

The model will be saved as `models/ppo_trading_model_TIMESTAMP` where TIMESTAMP is the current date and time.

### 2. Running the RL Model Backtest

1. After training is complete, run the backtest:
   ```bash
   python run_backtest.py
   ```

Note: Make sure to update the model path in `run_backtest.py` to match your trained model's timestamp:
```python
# In run_backtest.py
model_path = "models/ppo_trading_model_TIMESTAMP"  # Update this with your model's timestamp
```

The backtest will:
- Load the trained model
- Run it against historical data
- Generate performance metrics
- Create visualization plots

### 3. Running Traditional Strategy Backtest

The project includes a traditional strategy implementation combining Moving Average Ribbon and CCI indicators.

1. Run the traditional backtest script:
   ```bash
   python run_traditional_backtest.py
   ```

The script will:
- Run backtest using MA Ribbon + CCI strategy
- Display detailed performance metrics
- Generate equity curve visualization
- Perform parameter optimization
- Save results to the results directory

### 4. Training and Inference Flow

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

### 5. Configuration

Key parameters can be modified in `config.py`:

- Trading parameters (symbol, timeframe, risk settings)
- RL model parameters (learning rate, batch size, etc.)
- Environment parameters (window size, reward scaling)
- Risk management settings
- Traditional strategy parameters (MA periods, CCI settings)

## Backtesting Frameworks

The project includes two backtesting approaches:

### 1. Traditional Strategy Backtest (`backtest.py`)
- Implements MA Ribbon + CCI strategy combination
- Supports multi-timeframe analysis
- Includes parameter optimization
- Generates detailed trade statistics
- Visualizes equity curve
- Configurable risk management

### 2. RL Model Backtest (`backtest_framework.py`)
- Tests reinforcement learning model performance
- Supports PPO model evaluation
- Includes drawdown analysis
- Generates performance visualizations
- Provides detailed metrics:
  - Win rate
  - Profit factor
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown

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
├── config.py                    # Strategy parameters
├── utils.py                     # Utility functions
├── rl_signal_writer_torch.py    # RL model implementation
├── ma_ribbon.py                 # MA ribbon implementation
├── cci_strategy.py             # CCI strategy implementation
├── backtest.py                 # Traditional strategy backtest
├── backtest_framework.py       # RL model backtest
├── run_backtest.py            # RL backtest runner
├── run_traditional_backtest.py # Traditional backtest runner
├── requirements.txt            # Python dependencies
├── models/                    # Saved model checkpoints
├── logs/                      # Training logs
├── data/                      # Market data storage
└── results/                   # Backtest results
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

### Backtesting Results
Both backtesting frameworks generate:
- Performance metrics (returns, ratios)
- Trade statistics
- Equity curves
- Risk metrics
- Parameter optimization results

Results are saved in the `results/` directory with:
- Equity curve plots
- Performance reports
- Optimization results
- Trade logs 