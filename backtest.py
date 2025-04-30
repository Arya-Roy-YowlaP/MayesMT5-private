import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from ma_ribbon import MARibbon
from cci_strategy import CCIStrategy

class BacktestEngine:
    def __init__(self, 
                 symbol: str,
                 start_date: datetime,
                 end_date: datetime,
                 initial_balance: float = 10000.0,
                 risk_per_trade: float = 0.01):
        """
        Initialize backtesting engine
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_balance: Initial account balance
            risk_per_trade: Risk per trade as fraction of balance
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        
        # Initialize MT5 connection
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")
            
        # Initialize strategies
        self.ma_ribbon = MARibbon(
            timeframes=['M1', 'M15', 'H1'],
            periods=[1, 1, 1, 1, 1, 50],
            shifts=[0, 1, 2, 3, 4, 0]
        )
        
        self.cci_strategy = CCIStrategy(
            timeframes=['M1', 'M15', 'M30'],
            periods=[14, 20],
            overbought=100,
            oversold=-100
        )
        
        # Initialize results storage
        self.trades = []
        self.equity_curve = []
        self.current_balance = initial_balance
        
    def fetch_data(self, timeframe: str) -> pd.DataFrame:
        """Fetch historical data from MT5"""
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1
        }
        
        rates = mt5.copy_rates_range(
            self.symbol,
            tf_map[timeframe],
            self.start_date,
            self.end_date
        )
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk parameters"""
        risk_amount = self.current_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        return position_size
    
    def run_backtest(self) -> Dict:
        """Run the backtest and return results"""
        # Fetch data for all timeframes
        data = {
            'M1': self.fetch_data('M1'),
            'M15': self.fetch_data('M15'),
            'M30': self.fetch_data('M30'),
            'H1': self.fetch_data('H1')
        }
        
        # Initialize results
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.current_balance = self.initial_balance
        
        # Iterate through each bar
        for i in range(len(data['M1'])):
            # Update indicators
            for timeframe in data:
                df = data[timeframe]
                if i < len(df):
                    self.ma_ribbon.update_ribbon(
                        timeframe,
                        df['close'].values[:i+1]
                    )
                    self.cci_strategy.update_indicators(
                        timeframe,
                        df['high'].values[:i+1],
                        df['low'].values[:i+1],
                        df['close'].values[:i+1]
                    )
            
            # Get signals
            ma_signal, ma_direction = self.ma_ribbon.get_signal()
            cci_signal, cci_direction = self.cci_strategy.get_signal()
            
            # Check for trade entry
            if ma_signal and cci_signal and ma_direction == cci_direction:
                entry_price = data['M1']['close'].iloc[i]
                stop_loss = data['M1']['low'].iloc[i] if ma_direction == 'bullish' else data['M1']['high'].iloc[i]
                
                position_size = self.calculate_position_size(entry_price, stop_loss)
                
                # Record trade
                trade = {
                    'entry_time': data['M1']['time'].iloc[i],
                    'entry_price': entry_price,
                    'direction': ma_direction,
                    'position_size': position_size,
                    'stop_loss': stop_loss
                }
                self.trades.append(trade)
            
            # Update equity curve
            if self.trades:
                last_trade = self.trades[-1]
                if last_trade['entry_time'] <= data['M1']['time'].iloc[i]:
                    pnl = (data['M1']['close'].iloc[i] - last_trade['entry_price']) * \
                          last_trade['position_size'] * (1 if last_trade['direction'] == 'bullish' else -1)
                    self.current_balance += pnl
            
            self.equity_curve.append(self.current_balance)
        
        # Calculate performance metrics
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (pd.Series(self.equity_curve).cummax() - self.equity_curve).max() / self.initial_balance,
            'win_rate': len([t for t in self.trades if t.get('pnl', 0) > 0]) / len(self.trades) if self.trades else 0,
            'total_trades': len(self.trades),
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        return results
    
    def optimize_parameters(self, param_grid: Dict) -> Dict:
        """
        Optimize strategy parameters using grid search
        
        Args:
            param_grid: Dictionary of parameters to optimize
        """
        best_result = None
        best_sharpe = -np.inf
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            # Update strategy parameters
            self._update_strategy_params(params)
            
            # Run backtest
            results = self.run_backtest()
            
            # Update best result if better
            if results['sharpe_ratio'] > best_sharpe:
                best_sharpe = results['sharpe_ratio']
                best_result = {
                    'parameters': params,
                    'results': results
                }
        
        return best_result
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all combinations of parameters for optimization"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _update_strategy_params(self, params: Dict):
        """Update strategy parameters"""
        if 'ma_periods' in params:
            self.ma_ribbon.periods = params['ma_periods']
        if 'ma_shifts' in params:
            self.ma_ribbon.shifts = params['ma_shifts']
        if 'cci_periods' in params:
            self.cci_strategy.periods = params['cci_periods']
        if 'overbought' in params:
            self.cci_strategy.overbought = params['overbought']
        if 'oversold' in params:
            self.cci_strategy.oversold = params['oversold']
    
    def __del__(self):
        """Cleanup MT5 connection"""
        mt5.shutdown() 