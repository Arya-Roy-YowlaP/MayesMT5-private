from datetime import datetime
from backtest import BacktestEngine
import matplotlib.pyplot as plt

def main():
    # Initialize backtester
    backtest = BacktestEngine(
        symbol="EURUSD",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_balance=10000.0,
        risk_per_trade=0.01
    )
    
    # Run backtest
    print("Running traditional strategy backtest (MA Ribbon + CCI)...")
    results = backtest.run_backtest()
    
    # Print results
    print("\nBacktest Results:")
    print("-" * 50)
    print(f"Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'])
    plt.title('Equity Curve - Traditional Strategy')
    plt.xlabel('Bars')
    plt.ylabel('Account Balance ($)')
    plt.grid(True)
    plt.savefig('results/traditional_strategy_equity.png')
    plt.close()
    
    # Optional: Run parameter optimization
    print("\nOptimizing strategy parameters...")
    param_grid = {
        'ma_periods': [[1, 1, 1, 1, 1, 50], [1, 2, 3, 4, 5, 100]],
        'ma_shifts': [[0, 1, 2, 3, 4, 0], [0, 2, 4, 6, 8, 0]],
        'cci_periods': [[14, 20], [10, 30]],
        'overbought': [100, 80],
        'oversold': [-100, -80]
    }
    
    optimization_results = backtest.optimize_parameters(param_grid)
    
    print("\nOptimal Parameters:")
    print("-" * 50)
    for param, value in optimization_results['parameters'].items():
        print(f"{param}: {value}")
    
    print("\nOptimal Results:")
    print("-" * 50)
    optimal_results = optimization_results['results']
    print(f"Total Return: {optimal_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {optimal_results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {optimal_results['win_rate']:.2%}")

if __name__ == "__main__":
    main() 