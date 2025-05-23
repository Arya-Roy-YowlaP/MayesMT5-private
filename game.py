import numpy as np
import pandas as pd
from datetime import timedelta
import talib  # For technical indicators


def reward_function(entry_price, exit_price, position, daily_profit, daily_loss, daily_profit_target=100, daily_max_loss=-50):
    # Trade PnL: positive if profitable, negative if not
    pnl = (exit_price - entry_price) * position  # position = 1 for long, -1 for short

    # Basic reward is realized PnL
    reward = pnl

    # Daily profit bonus
    if daily_profit >= daily_profit_target:
        reward += 10  # bonus for hitting daily target

    # Daily loss penalty
    if daily_loss <= daily_max_loss:
        reward -= 10  # penalty for exceeding daily loss

    return reward



class Game(object):

    def __init__(self, bars30m, bars1d, bars4h, reward_function, lkbk=20, init_idx=None):
        self.bars30m = bars30m
        self.lkbk = lkbk
        self.trade_len = 0
        self.stop_pnl = None
        self.bars1d = bars1d
        self.bars4h = bars4h
        self.is_over = False
        self.reward = 0
        self.pnl_sum = 0
        self.init_idx = init_idx
        self.reward_function = reward_function
        # Add daily tracking variables
        self.daily_profit = 0
        self.daily_loss = 0
        self.daily_profit_target = 100
        self.daily_max_loss = -50
        self.reset()

    def _update_position(self, action):
        '''This is where we update our position'''
        if action == 0:
            pass

        elif action == 2:
            """---Enter a long or exit a short position---"""

            # If the current position (buy) is the same as the action (buy), do nothing
            if self.position == 1:
                pass

            # If there is no current position, we update the position to indicate buy
            elif self.position == 0:
                self.position = 1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx

            # If action is different than current position, we end the game and get rewards & trade duration
            elif self.position == -1:
                self.is_over = True
                # Calculate trade PnL
                trade_pnl = (self.curr_price - self.entry) * self.position
                # Update daily profit/loss
                if trade_pnl > 0:
                    self.daily_profit += trade_pnl
                else:
                    self.daily_loss += trade_pnl
                # Calculate reward using the reward function
                self.reward += self.reward_function(
                    self.entry, self.curr_price, self.position,
                    self.daily_profit, self.daily_loss,
                    self.daily_profit_target, self.daily_max_loss
                )

        elif action == 1:
            """---Enter a short or exit a long position---"""
            if self.position == -1:
                pass

            elif self.position == 0:
                self.position = -1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx

            elif self.position == 1:
                self.is_over = True
                # Calculate trade PnL
                trade_pnl = (self.curr_price - self.entry) * self.position
                # Update daily profit/loss
                if trade_pnl > 0:
                    self.daily_profit += trade_pnl
                else:
                    self.daily_loss += trade_pnl
                # Calculate reward using the reward function
                self.reward += self.reward_function(
                    self.entry, self.curr_price, self.position,
                    self.daily_profit, self.daily_loss,
                    self.daily_profit_target, self.daily_max_loss
                )

    def _check_ribbon_formation(self, sma_values):
        """Check if a ribbon formation exists (all SMAs aligned in order)"""
        # Check if SMAs are aligned in ascending order (bullish) or descending order (bearish)
        is_bullish = all(sma_values[i] <= sma_values[i+1] for i in range(len(sma_values)-1))
        is_bearish = all(sma_values[i] >= sma_values[i+1] for i in range(len(sma_values)-1))
        return 1 if is_bullish else -1 if is_bearish else 0

    def _check_cci_conditions(self, cci_values, cci_sma_values):
        """Check if CCI conditions are met for entry"""
        # Check if both CCIs are above their SMAs and above 100 (bullish)
        # or below their SMAs and below -100 (bearish)
        is_bullish = all(cci > sma and cci > 100 for cci, sma in zip(cci_values, cci_sma_values))
        is_bearish = all(cci < sma and cci < -100 for cci, sma in zip(cci_values, cci_sma_values))
        return 1 if is_bullish else -1 if is_bearish else 0

    def _assemble_state(self):
        '''Here we can add secondary features such as indicators and times to our current state.
        First, we create candlesticks for different bar sizes of 30mins, 4hr and 1d.
        We then add some state variables such as time of day, day of week and position.
        Next, several indicators are added and subsequently z-scored.
        '''

        """---Initializing State Variables---"""
        self.state = np.array([])

        self._get_last_N_timebars()

        """"---Adding Normalised Candlesticks---"""

        def _get_normalised_bars_array(bars):
            bars = bars.iloc[-10:, :-1].values.flatten()
            """Normalizing candlesticks"""
            bars = (bars-np.mean(bars))/np.std(bars)
            return bars

        self.state = np.append(
            self.state, _get_normalised_bars_array(self.last30m))
        self.state = np.append(
            self.state, _get_normalised_bars_array(self.last4h))
        self.state = np.append(
            self.state, _get_normalised_bars_array(self.last1d))

        """---Adding Technical Indicators---"""

        def _get_technical_indicators(bars):
            # Create an array to store the value of indicators
            tech_ind = np.array([])
            
            # Add SMA Ribbon indicators
            """SMA Ribbon - 5 SMAs with period 1 and shifts 0-4"""
            sma_values = []
            for shift in range(5):  # shifts 0 to 4
                sma = talib.SMA(bars['close'], timeperiod=1)[-1-shift]  # Using negative indexing for shift
                sma_values.append(sma)
                tech_ind = np.append(tech_ind, sma)
            
            """SMA with period 50"""
            sma50 = talib.SMA(bars['close'], timeperiod=50)[-1]
            tech_ind = np.append(tech_ind, sma50)
            
            # Add two CCIs with shifted SMAs
            """First CCI with 20-period SMA"""
            cci1 = talib.CCI(bars['high'], bars['low'], bars['close'], timeperiod=20)
            cci1_sma = talib.SMA(cci1, timeperiod=20)[-1]  # 20-period SMA of CCI
            tech_ind = np.append(tech_ind, cci1[-1])  # Current CCI value
            tech_ind = np.append(tech_ind, cci1_sma)  # SMA of CCI
            
            """Second CCI with 50-period SMA"""
            cci2 = talib.CCI(bars['high'], bars['low'], bars['close'], timeperiod=50)
            cci2_sma = talib.SMA(cci2, timeperiod=50)[-1]  # 50-period SMA of CCI
            tech_ind = np.append(tech_ind, cci2[-1])  # Current CCI value
            tech_ind = np.append(tech_ind, cci2_sma)  # SMA of CCI
            
            # Add entry/exit signals
            ribbon_signal = self._check_ribbon_formation(sma_values)
            cci_signal = self._check_cci_conditions([cci1[-1], cci2[-1]], [cci1_sma, cci2_sma])
            
            tech_ind = np.append(tech_ind, ribbon_signal)
            tech_ind = np.append(tech_ind, cci_signal)
            
            return tech_ind

        # Get indicators for each timeframe
        indicators_30m = _get_technical_indicators(self.last30m)
        indicators_4h = _get_technical_indicators(self.last4h)
        indicators_1d = _get_technical_indicators(self.last1d)

        # Add indicators to state
        self.state = np.append(self.state, indicators_30m)
        self.state = np.append(self.state, indicators_4h)
        self.state = np.append(self.state, indicators_1d)

        # Add entry/exit signals based on all timeframes
        entry_signal = 0
        if (indicators_4h[-2] == 1 and indicators_1d[-2] == 1 and  # Ribbon formation on higher timeframes
            indicators_30m[-2] == 1 and  # Ribbon formation on entry timeframe
            indicators_4h[-1] == 1 and indicators_1d[-1] == 1 and  # CCI conditions on higher timeframes
            indicators_30m[-1] == 1):  # CCI conditions on entry timeframe
            entry_signal = 1  # Buy signal
        elif (indicators_4h[-2] == -1 and indicators_1d[-2] == -1 and  # Ribbon formation on higher timeframes
              indicators_30m[-2] == -1 and  # Ribbon formation on entry timeframe
              indicators_4h[-1] == -1 and indicators_1d[-1] == -1 and  # CCI conditions on higher timeframes
              indicators_30m[-1] == -1):  # CCI conditions on entry timeframe
            entry_signal = -1  # Sell signal

        # Exit signal based on ribbon direction change on lowest timeframe
        exit_signal = 0
        if self.position != 0:  # Only check exit if we have a position
            if (self.position == 1 and indicators_30m[-2] == -1) or  # Long position and ribbon turns bearish
               (self.position == -1 and indicators_30m[-2] == 1):  # Short position and ribbon turns bullish
                exit_signal = 1

        # Add entry/exit signals to state
        self.state = np.append(self.state, entry_signal)
        self.state = np.append(self.state, exit_signal)

        # """---Adding Time Signature---"""
        # self.curr_time = self.bars30m.index[self.curr_idx]
        # tm_lst = list(map(float, str(self.curr_time.time()).split(':')[:2]))
        # self._time_of_day = (tm_lst[0]*60 + tm_lst[1])/(24*60)
        # self._day_of_week = self.curr_time.weekday()/6
        # self.state = np.append(self.state, self._time_of_day)
        # self.state = np.append(self.state, self._day_of_week)

        """---Adding Position---"""
        self.state = np.append(self.state, self.position)

    def _get_last_N_timebars(self):
        '''This function gets the timebars for the 30m, 4hr and 1d resolution based
        on the lookback we've specified.
        '''

        """---Getting candlesticks before current time---"""
        self.last30m = self.bars30m.iloc[-self.lkbk:]
        self.last4h = self.bars4h.iloc[-self.lkbk:]
        self.last1d = self.bars1d.iloc[-self.lkbk:]

    def _get_reward(self):
        """Here we calculate the reward when the game is finished.
        Reward function design is very difficult and can significantly
        impact the performance of our algo.
        In this case, we use a simple pnl reward but it is conceivable to use
        other metrics such as Sharpe ratio, average return, etc.
        """
        if self.is_over:
            self.reward = self.reward_function(
                self.entry, self.curr_price, self.position)

    def get_state(self):
        """This function returns the state of the system.
        Returns:
            self.state: the state including indicators, position and times.
        """
        # Assemble new state
        self._assemble_state()
        return np.array([self.state])

    def act(self, action):
        """This function updates the state based on an action
        that was calculated by the NN.
        This is the point where the game interacts with the trading
        algo.
        """

        self.curr_time = self.bars30m.index[self.curr_idx]
        self.curr_price = self.bars30m['close'][self.curr_idx]

        self._update_position(action)

        # Unrealized or realized pnl. This is different from pnl in reward method which is only realized pnl.
        self.pnl = (-self.entry + self.curr_price)*self.position/self.entry

        self._get_reward()
        if self.is_over:
            self.trade_len = self.curr_idx - self.start_idx

        return self.reward, self.is_over

    def reset(self):
        """Resetting the system for each new trading game.
        Here, we also resample the bars for 1h and 1d.
        Ideally, we should do this on every update but this will take very long.
        """
        self.pnl = 0
        self.entry = 0
        self._time_of_day = 0
        self._day_of_week = 0
        self.curr_idx = self.init_idx
        self.t_in_secs = (
            self.bars30m.index[-1]-self.bars30m.index[0]).total_seconds()
        self.start_idx = self.curr_idx
        self.curr_time = self.bars30m.index[self.curr_idx]
        self._get_last_N_timebars()
        self.position = 0
        # Reset daily tracking variables if it's a new day
        if self.curr_time.date() != self.last_reset_date:
            self.daily_profit = 0
            self.daily_loss = 0
            self.last_reset_date = self.curr_time.date()
        self.act(0)
        self.state = []
        self._assemble_state()

def load_mt5_data(symbol='EURUSD', timeframes=['M30', 'H4', 'D1'], start_date=None, end_date=None, count=None):
    """
    Load data from MT5 exported CSV files and prepare it for the Game environment.
    
    Args:
        symbol (str): Trading symbol (default: 'EURUSD')
        timeframes (list): List of timeframes to load (default: ['M30', 'H4', 'D1'])
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        count (int): Number of candles to load
        
    Returns:
        tuple: (bars30m, bars4h, bars1d) pandas DataFrames with OHLCV data
    """
    from datetime import datetime
    
    # Convert dates if provided
    start_dt = None
    end_dt = None
    if start_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Load data for each timeframe
    data = {}
    for tf in timeframes:
        filename = f"data/{symbol}_{tf.lower()}.csv"
        try:
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Filter by date range if provided
            if start_dt:
                df = df[df.index >= start_dt]
            if end_dt:
                df = df[df.index <= end_dt]
            
            # Filter by count if provided
            if count:
                df = df.iloc[-count:]
            
            data[tf] = df
        except Exception as e:
            print(f"Error loading data for {symbol} {tf}: {e}")
            return None, None, None
    
    # Return data in the format expected by Game
    return data['M30'], data['H4'], data['D1']

# Example usage in main.py or where you create the environment:
"""
# Load data
bars30m, bars4h, bars1d = load_mt5_data(
    symbol='EURUSD',
    timeframes=['M30', 'H4', 'D1'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Create environment
env = Game(
    bars30m=bars30m,
    bars4h=bars4h,
    bars1d=bars1d,
    reward_function=reward_function,
    lkbk=20
)
"""