import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import matplotlib.pyplot as plt
from game_environment import Game
import torch

class BacktestResults:
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.returns = []
        self.current_equity = 10000  # Starting equity
        self.peak_equity = 10000
        self.max_drawdown = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        
    def add_trade(self, entry_price, exit_price, position_type, entry_time, exit_time):
        pnl = (exit_price - entry_price) if position_type == 'BUY' else (entry_price - exit_price)
        pnl_pct = pnl / entry_price
        self.current_equity *= (1 + pnl_pct)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_type': position_type,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'equity': self.current_equity,
            'drawdown': drawdown
        }
        
        self.trades.append(trade)
        self.equity_curve.append(self.current_equity)
        self.drawdown_curve.append(drawdown)
        self.returns.append(pnl_pct)
        
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.total_trades += 1
        
    def calculate_metrics(self):
        if not self.trades:
            return {}
            
        returns = np.array(self.returns)
        equity = np.array(self.equity_curve)
        
        # Basic metrics
        total_return = (self.current_equity - 10000) / 10000
        win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
        profit_factor = abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if any(r < 0 for r in returns) else float('inf')
        
        # Risk metrics
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if any(r < 0 for r in returns) else float('inf')
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades
        }
        
    def plot_results(self):
        plt.figure(figsize=(15, 10))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trades')
        plt.ylabel('Equity')
        
        # Drawdown curve
        plt.subplot(2, 1, 2)
        plt.plot(self.drawdown_curve)
        plt.title('Drawdown')
        plt.xlabel('Trades')
        plt.ylabel('Drawdown %')
        
        plt.tight_layout()
        plt.show()

class Backtest:
    def __init__(self, model, initial_balance=10000, risk_per_trade=0.01, model_type="ppo"):
        self.model = model
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.model_type = model_type.lower()
        self.results = BacktestResults()
        
    def run(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, start_date=None, end_date=None):
        # Initialize environment
        env = Game(symbol=symbol, timeframe=timeframe)
        
        # Get historical data
        if start_date is None:
            start_date = datetime(2023, 1, 1)
        if end_date is None:
            end_date = datetime.now()
            
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None:
            print("No historical data available")
            return
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Run backtest
        position = None
        entry_price = 0
        entry_time = None
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = df['time'].iloc[i]
            
            # Get state and action
            state = env.get_state()
            
            # Handle different model types
            if self.model_type == "ppo":
                action, _ = self.model.predict(state, deterministic=True)
            else:  # DQN
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    q_values = self.model(state_tensor)
                    action = torch.argmax(q_values).item()
            
            # Handle position management
            if position is None:
                if action == 2:  # BUY
                    position = 'BUY'
                    entry_price = current_price
                    entry_time = current_time
                elif action == 1:  # SELL
                    position = 'SELL'
                    entry_price = current_price
                    entry_time = current_time
            else:
                # Check for exit conditions
                if (position == 'BUY' and action == 1) or (position == 'SELL' and action == 2):
                    self.results.add_trade(entry_price, current_price, position, entry_time, current_time)
                    position = None
                    
        # Close any open position at the end
        if position is not None:
            self.results.add_trade(entry_price, df['close'].iloc[-1], position, entry_time, df['time'].iloc[-1])
            
        # Calculate and display results
        metrics = self.results.calculate_metrics()
        print("\nBacktest Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        self.results.plot_results()
        
        env.close()
        return metrics 