import numpy as np
import MetaTrader5 as mt5
from typing import List, Dict, Tuple

class MARibbon:
    def __init__(self, timeframes: List[str], periods: List[int], shifts: List[int]):
        """
        Initialize MA Ribbon with specified parameters
        
        Args:
            timeframes: List of timeframes to analyze (e.g., ['M1', 'M15', 'H1'])
            periods: List of MA periods
            shifts: List of MA shifts
        """
        self.timeframes = timeframes
        self.periods = periods
        self.shifts = shifts
        self.ma_values = {}
        
    def calculate_ma(self, data: np.ndarray, period: int, shift: int) -> np.ndarray:
        """Calculate moving average with specified period and shift"""
        ma = np.zeros_like(data)
        for i in range(len(data)):
            if i >= period + shift:
                ma[i] = np.mean(data[i-period-shift:i-shift])
        return ma
    
    def update_ribbon(self, timeframe: str, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Update MA ribbon values for a specific timeframe"""
        ribbon = {}
        for period, shift in zip(self.periods, self.shifts):
            key = f'MA_{period}_{shift}'
            ribbon[key] = self.calculate_ma(data, period, shift)
        self.ma_values[timeframe] = ribbon
        return ribbon
    
    def check_ribbon_formation(self, timeframe: str) -> Tuple[bool, str]:
        """
        Check if a valid ribbon formation exists
        
        Returns:
            Tuple of (is_formed, direction)
            direction: 'bullish', 'bearish', or None
        """
        if timeframe not in self.ma_values:
            return False, None
            
        ribbon = self.ma_values[timeframe]
        ma_values = [ribbon[f'MA_{p}_{s}'] for p, s in zip(self.periods, self.shifts)]
        
        # Check if all MAs are properly aligned
        current_values = [ma[-1] for ma in ma_values]
        
        # Bullish formation: shorter MAs above longer MAs
        if all(current_values[i] >= current_values[i+1] for i in range(len(current_values)-1)):
            return True, 'bullish'
            
        # Bearish formation: shorter MAs below longer MAs
        if all(current_values[i] <= current_values[i+1] for i in range(len(current_values)-1)):
            return True, 'bearish'
            
        return False, None
    
    def get_signal(self) -> Tuple[bool, str]:
        """
        Get trading signal based on ribbon formation across all timeframes
        
        Returns:
            Tuple of (should_trade, direction)
        """
        # Check higher timeframes first (trend filter)
        higher_timeframes = [tf for tf in self.timeframes if tf != 'M1']
        higher_signals = []
        
        for tf in higher_timeframes:
            is_formed, direction = self.check_ribbon_formation(tf)
            if not is_formed:
                return False, None
            higher_signals.append(direction)
            
        # All higher timeframes must agree on direction
        if not all(s == higher_signals[0] for s in higher_signals):
            return False, None
            
        # Check entry timeframe (M1)
        is_formed, direction = self.check_ribbon_formation('M1')
        if not is_formed or direction != higher_signals[0]:
            return False, None
            
        return True, direction
    
    def get_ma_values(self, timeframe: str) -> Dict[str, np.ndarray]:
        """Get current MA values for a specific timeframe"""
        return self.ma_values.get(timeframe, {}) 