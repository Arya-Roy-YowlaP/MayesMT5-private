import os
import numpy as np
import pandas as pd
from datetime import datetime
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