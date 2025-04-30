import numpy as np
import MetaTrader5 as mt5
from typing import List, Dict, Tuple

class CCIStrategy:
    def __init__(self, timeframes: List[str], periods: List[int], overbought: float = 100, oversold: float = -100):
        """
        Initialize CCI Strategy with specified parameters
        
        Args:
            timeframes: List of timeframes to analyze
            periods: List of CCI periods
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        self.timeframes = timeframes
        self.periods = periods
        self.overbought = overbought
        self.oversold = oversold
        self.cci_values = {}
        self.sma_values = {}
        
    def calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Commodity Channel Index"""
        tp = (high + low + close) / 3
        tp_sma = np.zeros_like(tp)
        tp_md = np.zeros_like(tp)
        
        for i in range(len(tp)):
            if i >= period:
                tp_sma[i] = np.mean(tp[i-period:i])
                tp_md[i] = np.mean(np.abs(tp[i-period:i] - tp_sma[i]))
                
        cci = np.zeros_like(tp)
        for i in range(len(tp)):
            if tp_md[i] != 0:
                cci[i] = (tp[i] - tp_sma[i]) / (0.015 * tp_md[i])
                
        return cci
    
    def calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        sma = np.zeros_like(data)
        for i in range(len(data)):
            if i >= period:
                sma[i] = np.mean(data[i-period:i])
        return sma
    
    def update_indicators(self, timeframe: str, high: np.ndarray, low: np.ndarray, close: np.ndarray):
        """Update CCI and SMA values for a specific timeframe"""
        self.cci_values[timeframe] = {}
        self.sma_values[timeframe] = {}
        
        for period in self.periods:
            # Calculate CCI
            cci = self.calculate_cci(high, low, close, period)
            self.cci_values[timeframe][f'CCI_{period}'] = cci
            
            # Calculate SMA of CCI
            sma = self.calculate_sma(cci, period)
            self.sma_values[timeframe][f'SMA_{period}'] = sma
    
    def check_signal(self, timeframe: str) -> Tuple[bool, str]:
        """
        Check for trading signals on a specific timeframe
        
        Returns:
            Tuple of (has_signal, direction)
            direction: 'buy', 'sell', or None
        """
        if timeframe not in self.cci_values:
            return False, None
            
        for period in self.periods:
            cci = self.cci_values[timeframe][f'CCI_{period}']
            sma = self.sma_values[timeframe][f'SMA_{period}']
            
            # Buy signal: CCI above SMA and above overbought level
            if cci[-1] > sma[-1] and cci[-1] > self.overbought:
                return True, 'buy'
                
            # Sell signal: CCI below SMA and below oversold level
            if cci[-1] < sma[-1] and cci[-1] < self.oversold:
                return True, 'sell'
                
        return False, None
    
    def get_signal(self) -> Tuple[bool, str]:
        """
        Get trading signal based on CCI across all timeframes
        
        Returns:
            Tuple of (should_trade, direction)
        """
        # Check higher timeframes first
        higher_timeframes = [tf for tf in self.timeframes if tf != 'M1']
        higher_signals = []
        
        for tf in higher_timeframes:
            has_signal, direction = self.check_signal(tf)
            if not has_signal:
                return False, None
            higher_signals.append(direction)
            
        # All higher timeframes must agree on direction
        if not all(s == higher_signals[0] for s in higher_signals):
            return False, None
            
        # Check entry timeframe (M1)
        has_signal, direction = self.check_signal('M1')
        if not has_signal or direction != higher_signals[0]:
            return False, None
            
        return True, direction
    
    def get_indicator_values(self, timeframe: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Get current CCI and SMA values for a specific timeframe"""
        return (self.cci_values.get(timeframe, {}), 
                self.sma_values.get(timeframe, {})) 