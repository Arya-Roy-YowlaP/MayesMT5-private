import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from backtest_framework import Backtest, BacktestResults
from utils import (
    setup_logging,
    load_historical_data,
    calculate_position_size,
    check_risk_limits,
    normalize_data
)
from config import (
    TRADING_PARAMS,
    RISK_PARAMS,
    ENV_PARAMS
)

class TestTradingSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test environment"""
        cls.logger = setup_logging('test_trading_system')
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        mt5.shutdown()
        
    def test_backtest_results(self):
        """Test BacktestResults class"""
        results = BacktestResults()
        
        # Test adding trades
        results.add_trade(1.1000, 1.1050, 'BUY', datetime.now(), datetime.now())
        results.add_trade(1.2000, 1.1950, 'SELL', datetime.now(), datetime.now())
        
        self.assertEqual(len(results.trades), 2)
        self.assertEqual(results.win_count + results.loss_count, 2)
        
        # Test metrics calculation
        metrics = results.calculate_metrics()
        self.assertIn('total_return', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        
    def test_position_size_calculation(self):
        """Test position size calculation"""
        account_balance = 10000
        risk_per_trade = 0.01
        stop_loss_pips = 20
        
        position_size = calculate_position_size(
            account_balance,
            risk_per_trade,
            stop_loss_pips,
            TRADING_PARAMS['symbol']
        )
        
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 1.0)  # Assuming max lot size is 1.0
        
    def test_risk_limits(self):
        """Test risk limit checks"""
        risk_checks = check_risk_limits(
            account_balance=10000,
            current_drawdown=0.01,
            open_trades=[],
            win_rate=0.6,
            profit_factor=2.0
        )
        
        self.assertFalse(any(risk_checks.values()))
        
        # Test exceeding limits
        risk_checks = check_risk_limits(
            account_balance=10000,
            current_drawdown=0.15,  # Exceeds max_drawdown
            open_trades=[1, 2, 3, 4],  # Exceeds max_open_trades
            win_rate=0.3,  # Below min_win_rate
            profit_factor=1.2  # Below min_profit_factor
        )
        
        self.assertTrue(any(risk_checks.values()))
        
    def test_data_normalization(self):
        """Test data normalization"""
        # Create sample data
        data = pd.DataFrame({
            'open': [1.1, 1.2, 1.3],
            'high': [1.2, 1.3, 1.4],
            'low': [1.0, 1.1, 1.2],
            'close': [1.15, 1.25, 1.35],
            'tick_volume': [100, 200, 300]
        })
        
        normalized = normalize_data(data)
        
        self.assertEqual(normalized.shape, (3, 5))
        self.assertAlmostEqual(np.mean(normalized), 0, places=1)
        self.assertAlmostEqual(np.std(normalized), 1, places=1)
        
    def test_historical_data_loading(self):
        """Test loading historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        df = load_historical_data(
            TRADING_PARAMS['symbol'],
            TRADING_PARAMS['timeframe'],
            start_date,
            end_date
        )
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('time', df.columns)
        self.assertIn('close', df.columns)

if __name__ == '__main__':
    unittest.main() 