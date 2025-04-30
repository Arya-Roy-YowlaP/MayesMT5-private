import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from ma_ribbon import MARibbon
from cci_strategy import CCIStrategy

class RLGame:
    def __init__(self, 
                 bars1m: pd.DataFrame,
                 bars15m: pd.DataFrame,
                 bars1h: pd.DataFrame,
                 reward_function,
                 ma_params: Dict = None,
                 cci_params: Dict = None,
                 lkbk: int = 20,
                 init_idx: int = None):
        """
        Initialize RL Game with enhanced trading strategies
        
        Args:
            bars1m: 1-minute bars DataFrame
            bars15m: 15-minute bars DataFrame
            bars1h: 1-hour bars DataFrame
            reward_function: Function to calculate rewards
            ma_params: MA ribbon parameters
            cci_params: CCI strategy parameters
            lkbk: Lookback period
            init_idx: Initial index
        """
        self.bars1m = bars1m
        self.bars15m = bars15m
        self.bars1h = bars1h
        self.lkbk = lkbk
        self.trade_len = 0
        self.stop_pnl = None
        self.is_over = False
        self.reward = 0
        self.pnl_sum = 0
        self.init_idx = init_idx
        self.reward_function = reward_function
        
        # Initialize strategies
        self.ma_ribbon = MARibbon(
            timeframes=['M1', 'M15', 'H1'],
            periods=ma_params.get('periods', [1, 1, 1, 1, 1, 50]),
            shifts=ma_params.get('shifts', [0, 1, 2, 3, 4, 0])
        )
        
        self.cci_strategy = CCIStrategy(
            timeframes=['M1', 'M15', 'M30'],
            periods=cci_params.get('periods', [14, 20]),
            overbought=cci_params.get('overbought', 100),
            oversold=cci_params.get('oversold', -100)
        )
        
        self.reset()
    
    def _update_position(self, action):
        """Update position based on action and strategy signals"""
        if action == 0:  # Hold
            pass
        elif action == 2:  # Buy
            if self.position == 1:  # Already long
                pass
            elif self.position == 0:  # No position
                # Check strategy signals
                ma_signal, ma_direction = self.ma_ribbon.get_signal()
                cci_signal, cci_direction = self.cci_strategy.get_signal()
                
                if ma_signal and cci_signal and ma_direction == 'bullish' and cci_direction == 'buy':
                    self.position = 1
                    self.entry = self.curr_price
                    self.start_idx = self.curr_idx
            elif self.position == -1:  # Short position
                self.is_over = True
                
        elif action == 1:  # Sell
            if self.position == -1:  # Already short
                pass
            elif self.position == 0:  # No position
                # Check strategy signals
                ma_signal, ma_direction = self.ma_ribbon.get_signal()
                cci_signal, cci_direction = self.cci_strategy.get_signal()
                
                if ma_signal and cci_signal and ma_direction == 'bearish' and cci_direction == 'sell':
                    self.position = -1
                    self.entry = self.curr_price
                    self.start_idx = self.curr_idx
            elif self.position == 1:  # Long position
                self.is_over = True
    
    def _assemble_state(self):
        """Assemble state with enhanced features"""
        self.state = np.array([])
        
        # Get latest bars
        self._get_last_N_timebars()
        
        # Add normalized candlesticks
        def _get_normalised_bars_array(bars):
            bars = bars.iloc[-10:, :-1].values.flatten()
            bars = (bars-np.mean(bars))/np.std(bars)
            return bars
        
        self.state = np.append(self.state, _get_normalised_bars_array(self.last1m))
        self.state = np.append(self.state, _get_normalised_bars_array(self.last15m))
        self.state = np.append(self.state, _get_normalised_bars_array(self.last1h))
        
        # Add MA ribbon features
        for timeframe in ['M1', 'M15', 'H1']:
            ma_values = self.ma_ribbon.get_ma_values(timeframe)
            for ma in ma_values.values():
                self.state = np.append(self.state, ma[-1])
        
        # Add CCI features
        for timeframe in ['M1', 'M15', 'M30']:
            cci_values, sma_values = self.cci_strategy.get_indicator_values(timeframe)
            for cci in cci_values.values():
                self.state = np.append(self.state, cci[-1])
            for sma in sma_values.values():
                self.state = np.append(self.state, sma[-1])
        
        # Add time features
        self.curr_time = self.bars1m.index[self.curr_idx]
        tm_lst = list(map(float, str(self.curr_time.time()).split(':')[:2]))
        self._time_of_day = (tm_lst[0]*60 + tm_lst[1])/(24*60)
        self._day_of_week = self.curr_time.weekday()/6
        self.state = np.append(self.state, self._time_of_day)
        self.state = np.append(self.state, self._day_of_week)
        
        # Add position
        self.state = np.append(self.state, self.position)
    
    def _get_last_N_timebars(self):
        """Get latest bars for all timeframes"""
        wdw1m = 9
        wdw15m = np.ceil(self.lkbk*15/24.)
        wdw1h = np.ceil(self.lkbk*15)
        
        self.last1m = self.bars1m[self.curr_time - timedelta(minutes=wdw1m):self.curr_time].iloc[-self.lkbk:]
        self.last15m = self.bars15m[self.curr_time - timedelta(minutes=wdw15m):self.curr_time].iloc[-self.lkbk:]
        self.last1h = self.bars1h[self.curr_time - timedelta(hours=wdw1h):self.curr_time].iloc[-self.lkbk:]
        
        # Update indicators
        self.ma_ribbon.update_ribbon('M1', self.last1m['close'].values)
        self.ma_ribbon.update_ribbon('M15', self.last15m['close'].values)
        self.ma_ribbon.update_ribbon('H1', self.last1h['close'].values)
        
        self.cci_strategy.update_indicators(
            'M1',
            self.last1m['high'].values,
            self.last1m['low'].values,
            self.last1m['close'].values
        )
        self.cci_strategy.update_indicators(
            'M15',
            self.last15m['high'].values,
            self.last15m['low'].values,
            self.last15m['close'].values
        )
        self.cci_strategy.update_indicators(
            'H1',
            self.last1h['high'].values,
            self.last1h['low'].values,
            self.last1h['close'].values
        )
    
    def _get_reward(self):
        """Calculate reward with enhanced metrics"""
        if self.is_over:
            # Base PnL reward with scaling
            base_pnl = self.reward_function(self.entry, self.curr_price, self.position)
            scaled_pnl = base_pnl * self.reward_scaling
            
            # Trade duration penalty (exponential decay)
            trade_duration = self.curr_idx - self.start_idx
            duration_penalty = -0.001 * np.exp(trade_duration / 100)
            
            # Volatility adjustment (non-linear)
            volatility = np.std(self.last1m['close'].values) / np.mean(self.last1m['close'].values)
            volatility_factor = -0.1 * (volatility ** 2)  # Quadratic penalty for high volatility
            
            # Trend alignment bonus
            trend_direction = self._get_trend_direction()
            trend_bonus = 0.1 if (trend_direction > 0 and self.position > 0) or \
                                (trend_direction < 0 and self.position < 0) else -0.05
            
            # Risk management bonus/penalty
            risk_reward_ratio = abs(base_pnl) / (self.stop_loss if self.stop_loss else 0.01)
            risk_bonus = 0.1 if risk_reward_ratio >= 2.0 else -0.05  # Reward 2:1 or better RR ratio
            
            # Combine all components
            self.reward = (
                scaled_pnl +  # Base PnL (scaled)
                duration_penalty +  # Time decay
                volatility_factor +  # Volatility impact
                trend_bonus +  # Trend alignment
                risk_bonus  # Risk management
            )
            
            # Apply position sizing multiplier
            position_size_mult = min(1.0, self.position_size / self.max_position_size)
            self.reward *= position_size_mult
            
            # Apply drawdown penalty if applicable
            if self.curr_drawdown > self.max_allowed_drawdown:
                self.reward *= 0.5  # Reduce reward by half if exceeding drawdown limit
    
    def _get_trend_direction(self):
        """Calculate trend direction using multiple timeframes"""
        # Short-term trend (1m)
        short_sma = self.last1m['close'].rolling(5).mean().iloc[-1]
        short_trend = 1 if self.last1m['close'].iloc[-1] > short_sma else -1
        
        # Medium-term trend (15m)
        med_sma = self.last15m['close'].rolling(10).mean().iloc[-1]
        med_trend = 1 if self.last15m['close'].iloc[-1] > med_sma else -1
        
        # Long-term trend (1h)
        long_sma = self.last1h['close'].rolling(20).mean().iloc[-1]
        long_trend = 1 if self.last1h['close'].iloc[-1] > long_sma else -1
        
        # Weighted trend score
        return (0.2 * short_trend + 0.3 * med_trend + 0.5 * long_trend)
    
    def get_state(self):
        """Get current state"""
        self._assemble_state()
        return np.array([self.state])
    
    def act(self, action):
        """Execute action and update state"""
        self.curr_time = self.bars1m.index[self.curr_idx]
        self.curr_price = self.bars1m['close'][self.curr_idx]
        
        self._update_position(action)
        
        # Calculate unrealized PnL
        self.pnl = (-self.entry + self.curr_price)*self.position/self.entry
        
        self._get_reward()
        if self.is_over:
            self.trade_len = self.curr_idx - self.start_idx
        
        return self.reward, self.is_over
    
    def reset(self):
        """Reset game state"""
        self.pnl = 0
        self.entry = 0
        self._time_of_day = 0
        self._day_of_week = 0
        self.curr_idx = self.init_idx
        self.t_in_secs = (self.bars1m.index[-1]-self.bars1m.index[0]).total_seconds()
        self.start_idx = self.curr_idx
        self.curr_time = self.bars1m.index[self.curr_idx]
        self._get_last_N_timebars()
        self.position = 0
        self.act(0)
        self.state = []
        self._assemble_state() 