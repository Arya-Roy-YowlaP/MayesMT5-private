import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
import logging
from config import PATHS

def setup_logging(name):
    """Setup logging configuration"""
    os.makedirs(PATHS['logs_dir'], exist_ok=True)
    log_file = os.path.join(PATHS['logs_dir'], f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def save_results(results, filename):
    """Save backtest or training results"""
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    filepath = os.path.join(PATHS['results_dir'], filename)
    
    if isinstance(results, dict):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
    elif isinstance(results, pd.DataFrame):
        results.to_csv(filepath, index=False)
    else:
        raise ValueError("Results must be either a dictionary or pandas DataFrame")

def load_historical_data(symbol, timeframe, start_date, end_date):
    """Load historical data from MT5"""
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        raise ValueError(f"No data available for {symbol} from {start_date} to {end_date}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips, symbol):
    """Calculate position size based on risk parameters"""
    # Get current symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise ValueError(f"Could not get symbol info for {symbol}")
    
    # Calculate position size
    risk_amount = account_balance * risk_per_trade
    pip_value = symbol_info.trade_tick_value * symbol_info.trade_tick_size
    position_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Round to nearest lot size
    lot_step = symbol_info.volume_step
    position_size = round(position_size / lot_step) * lot_step
    
    return min(position_size, symbol_info.volume_max)

def check_risk_limits(account_balance, current_drawdown, open_trades, win_rate, profit_factor):
    """Check if risk limits are being exceeded"""
    from config import RISK_PARAMS
    
    risk_checks = {
        'daily_loss': current_drawdown > RISK_PARAMS['max_daily_loss'],
        'max_drawdown': current_drawdown > RISK_PARAMS['max_drawdown'],
        'max_trades': len(open_trades) >= RISK_PARAMS['max_open_trades'],
        'min_win_rate': win_rate < RISK_PARAMS['min_win_rate'],
        'min_profit_factor': profit_factor < RISK_PARAMS['min_profit_factor']
    }
    
    return risk_checks

def normalize_data(data, window_size=30):
    """Normalize price data for model input"""
    if isinstance(data, pd.DataFrame):
        data = data[['open', 'high', 'low', 'close', 'tick_volume']].values
    
    # Calculate rolling statistics
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Normalize data
    normalized = (data - means) / (stds + 1e-8)
    
    return normalized

def create_directories():
    """Create necessary directories for the project"""
    for directory in PATHS.values():
        os.makedirs(directory, exist_ok=True) 