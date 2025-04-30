import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
from config import PATHS

def plot_training_progress(rewards, losses=None, algorithm="dqn", save=True):
    """Plot training progress (rewards and losses)"""
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 2 if losses is not None else 1, 1)
    plt.plot(rewards, label='Episode Reward')
    plt.title(f'{algorithm.upper()} Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot losses if available
    if losses is not None:
        plt.subplot(1, 2, 2)
        plt.plot(losses, label='Training Loss')
        plt.title(f'{algorithm.upper()} Training Losses')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        plt.savefig(os.path.join(PATHS['results_dir'], f'training_progress_{algorithm}_{timestamp}.png'))
    
    plt.show()

def plot_algorithm_comparison(dqn_metrics, ppo_metrics, save=True):
    """Compare performance metrics between DQN and PPO"""
    metrics = ['total_return', 'win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown']
    dqn_values = [dqn_metrics[m] for m in metrics]
    ppo_values = [ppo_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, dqn_values, width, label='DQN')
    rects2 = ax.bar(x + width/2, ppo_values, width, label='PPO')
    
    ax.set_ylabel('Value')
    ax.set_title('Algorithm Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        plt.savefig(os.path.join(PATHS['results_dir'], f'algorithm_comparison_{timestamp}.png'))
    
    plt.show()

def plot_equity_curves(dqn_equity, ppo_equity, save=True):
    """Plot equity curves for both algorithms"""
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_equity, label='DQN')
    plt.plot(ppo_equity, label='PPO')
    plt.title('Equity Curve Comparison')
    plt.xlabel('Trade')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        plt.savefig(os.path.join(PATHS['results_dir'], f'equity_curves_{timestamp}.png'))
    
    plt.show()

def plot_trade_distribution(dqn_trades, ppo_trades, save=True):
    """Plot distribution of trade outcomes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # DQN trades
    dqn_returns = [t['pnl_pct'] for t in dqn_trades]
    ax1.hist(dqn_returns, bins=30, alpha=0.7, label='DQN')
    ax1.set_title('DQN Trade Distribution')
    ax1.set_xlabel('Return %')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # PPO trades
    ppo_returns = [t['pnl_pct'] for t in ppo_trades]
    ax2.hist(ppo_returns, bins=30, alpha=0.7, label='PPO')
    ax2.set_title('PPO Trade Distribution')
    ax2.set_xlabel('Return %')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        plt.savefig(os.path.join(PATHS['results_dir'], f'trade_distribution_{timestamp}.png'))
    
    plt.show()

def create_performance_report(dqn_metrics, ppo_metrics, save=True):
    """Create a comprehensive performance report"""
    metrics = ['total_return', 'win_rate', 'profit_factor', 'sharpe_ratio', 
              'sortino_ratio', 'max_drawdown', 'total_trades']
    
    report = pd.DataFrame({
        'Metric': metrics,
        'DQN': [dqn_metrics[m] for m in metrics],
        'PPO': [ppo_metrics[m] for m in metrics],
        'Difference': [ppo_metrics[m] - dqn_metrics[m] for m in metrics]
    })
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        report.to_csv(os.path.join(PATHS['results_dir'], f'performance_report_{timestamp}.csv'), index=False)
    
    return report 