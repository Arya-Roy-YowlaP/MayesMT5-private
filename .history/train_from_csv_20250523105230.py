import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces
import gymnasium as gym
from utils import setup_logging, create_directories
from config import MODEL_PARAMS, TRADING_PARAMS, ENV_PARAMS, PATHS
from visualization import plot_training_progress
from ta.trend import sma_indicator, cci
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


import numpy as np
import talib

class Game(object):
    def __init__(self, bars30m, bars1d, bars4h, reward_function, lkbk=20, init_idx=None):
        self.bars30m = bars30m
        self.bars1d = bars1d
        self.bars4h = bars4h
        self.lkbk = lkbk
        self.reward_function = reward_function
        self.init_idx = init_idx if init_idx is not None else 0
        self.daily_profit = 0
        self.daily_loss = 0
        self.daily_profit_target = 100
        self.daily_max_loss = -50
        self.last_reset_date = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(219,), dtype=np.float32)
        self.reset()

    def _update_position(self, action):
        if action == 0: pass
        elif action == 2:
            if self.position == 1: pass
            elif self.position == 0: self.position = 1; self.entry = self.curr_price; self.start_idx = self.curr_idx
            elif self.position == -1:
                self.is_over = True
                trade_pnl = (self.curr_price - self.entry) * self.position
                self.daily_profit += trade_pnl if trade_pnl > 0 else 0
                self.daily_loss += trade_pnl if trade_pnl <= 0 else 0
                self.reward += self.reward_function(self.entry, self.curr_price, self.position, self.daily_profit, self.daily_loss, self.daily_profit_target, self.daily_max_loss)
        elif action == 1:
            if self.position == -1: pass
            elif self.position == 0: self.position = -1; self.entry = self.curr_price; self.start_idx = self.curr_idx
            elif self.position == 1:
                self.is_over = True
                trade_pnl = (self.curr_price - self.entry) * self.position
                self.daily_profit += trade_pnl if trade_pnl > 0 else 0
                self.daily_loss += trade_pnl if trade_pnl <= 0 else 0
                self.reward += self.reward_function(self.entry, self.curr_price, self.position, self.daily_profit, self.daily_loss, self.daily_profit_target, self.daily_max_loss)

    def _check_ribbon_formation(self, sma_values):
        is_bullish = all(sma_values[i] <= sma_values[i+1] for i in range(len(sma_values)-1))
        is_bearish = all(sma_values[i] >= sma_values[i+1] for i in range(len(sma_values)-1))
        return 1 if is_bullish else -1 if is_bearish else 0

    def _check_cci_conditions(self, cci_values, cci_sma_values):
        is_bullish = all(cci > sma and cci > 100 for cci, sma in zip(cci_values, cci_sma_values))
        is_bearish = all(cci < sma and cci < -100 for cci, sma in zip(cci_values, cci_sma_values))
        return 1 if is_bullish else -1 if is_bearish else 0

    def _assemble_state(self):
        self.state = np.array([])
        self._get_last_N_timebars()
        def _get_normalised_bars_array(bars): return (bars.iloc[-10:, :-1].values.flatten() - np.mean(bars.iloc[-10:, :-1].values.flatten())) / np.std(bars.iloc[-10:, :-1].values.flatten())
        self.state = np.append(self.state, _get_normalised_bars_array(self.last30m))
        self.state = np.append(self.state, _get_normalised_bars_array(self.last4h))
        self.state = np.append(self.state, _get_normalised_bars_array(self.last1d))
        def _get_technical_indicators(bars):
            tech_ind = np.array([])

            # Ensure inputs are numeric pandas Series
            # bars['close'] = pd.to_numeric(bars['close'], errors='coerce')
            # bars['high'] = pd.to_numeric(bars['high'], errors='coerce')
            # bars['low'] = pd.to_numeric(bars['low'], errors='coerce')

            # Must have at least 99 rows for longest indicator to work
            if len(bars) < 99 or bars[['close', 'high', 'low']].isnull().any().any():
                print("Insufficient or invalid data for indicators")
                return np.full(16, np.nan)

            try:
                # SMA Ribbon (5 values using SMA(1) with shift 0–4)
                sma_values = []
                sma1_series = sma_indicator(bars['close'], window=1)
                for shift in range(5):
                    sma_values.append(sma1_series.iloc[-1 - shift])
                tech_ind = np.append(tech_ind, sma_values)

                # SMA(50)
                sma50 = sma_indicator(bars['close'], window=50).iloc[-1]
                tech_ind = np.append(tech_ind, sma50)

                # CCI(20) + SMA(CCI(20), 20)
                cci1 = cci(bars['high'], bars['low'], bars['close'], window=20, constant=0.015)
                cci1_sma = cci1.rolling(window=20).mean().iloc[-1]
                tech_ind = np.append(tech_ind, [cci1.iloc[-1], cci1_sma])

                # CCI(50) + SMA(CCI(50), 50)
                cci2 = cci(bars['high'], bars['low'], bars['close'], window=50, constant=0.015)
                cci2_sma = cci2.rolling(window=50).mean().iloc[-1]
                tech_ind = np.append(tech_ind, [cci2.iloc[-1], cci2_sma])

                # Entry/exit signal logic
                ribbon_signal = self._check_ribbon_formation(sma_values)
                cci_signal = self._check_cci_conditions([cci1.iloc[-1], cci2.iloc[-1]], [cci1_sma, cci2_sma])
                tech_ind = np.append(tech_ind, [ribbon_signal, cci_signal])

            except Exception as e:
                print("Error in technical indicator calculation:", e)
                return np.full(16, np.nan)

            return tech_ind

        indicators_30m = _get_technical_indicators(self.last30m)
        indicators_4h = _get_technical_indicators(self.last4h)
        indicators_1d = _get_technical_indicators(self.last1d)
        self.state = np.append(self.state, indicators_30m)
        self.state = np.append(self.state, indicators_4h)
        self.state = np.append(self.state, indicators_1d)
        entry_signal = 1 if (indicators_4h[-2] == 1 and indicators_1d[-2] == 1 and indicators_30m[-2] == 1 and indicators_4h[-1] == 1 and indicators_1d[-1] == 1 and indicators_30m[-1] == 1) else -1 if (indicators_4h[-2] == -1 and indicators_1d[-2] == -1 and indicators_30m[-2] == -1 and indicators_4h[-1] == -1 and indicators_1d[-1] == -1 and indicators_30m[-1] == -1) else 0
        exit_signal = 1 if self.position != 0 and ((self.position == 1 and indicators_30m[-2] == -1) or (self.position == -1 and indicators_30m[-2] == 1)) else 0
        self.state = np.append(self.state, entry_signal)
        self.state = np.append(self.state, exit_signal)
        self.state = np.append(self.state, self.position)

    def _get_last_N_timebars(self):
        self.last30m = self.bars30m.iloc[int(self.curr_idx)-self.lkbk+1:int(self.curr_idx)+1] if int(self.curr_idx) >= self.lkbk-1 else self.bars30m.iloc[:int(self.curr_idx)+1]
        self.last4h = self.bars4h.iloc[int(self.curr_idx)-self.lkbk+1:int(self.curr_idx)+1] if int(self.curr_idx) >= self.lkbk-1 else self.bars4h.iloc[:int(self.curr_idx)+1]
        self.last1d = self.bars1d.iloc[int(self.curr_idx)-self.lkbk+1:int(self.curr_idx)+1] if int(self.curr_idx) >= self.lkbk-1 else self.bars1d.iloc[:int(self.curr_idx)+1]

    def _get_reward(self):
        if self.is_over: self.reward = self.reward_function(self.entry, self.curr_price, self.position)

    def get_state(self):
        self._assemble_state()
        return np.array(self.state, dtype=np.float32)

    def act(self, action):
        self.curr_time = self.bars30m.index[int(self.curr_idx)]
        self.curr_price = self.bars30m['close'].iloc[int(self.curr_idx)]
        self._update_position(action)
        self.pnl = (-self.entry + self.curr_price) * self.position / self.entry if self.entry != 0 else 0
        if self.is_over: self.trade_len = self.curr_idx - self.start_idx
        return self.reward, self.is_over

    def step(self, action):
        reward, _ = self.act(action)  # ignore is_over here
        self.is_over = False  # override act’s termination unless at dataset end

        if self.curr_idx < len(self.bars30m) - 1:
            self.curr_idx += 1
        else:
            self.is_over = True  # only terminate at end of data
            print("State shape:", self.state.shape)

        self._assemble_state()
        obs = np.array(self.state, dtype=np.float32)
        terminated = self.is_over
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info


    def reset(self):
        self.pnl = 0
        self.entry = 0
        self._time_of_day = 0
        self._day_of_week = 0
        self.curr_idx = self.init_idx if self.init_idx is not None else 0
        self.t_in_secs = (self.bars30m.index[-1] - self.bars30m.index[0]).total_seconds()
        self.start_idx = self.curr_idx
        self.curr_time = self.bars30m.index[int(self.curr_idx)]
        self._get_last_N_timebars()
        self.position = 0
        self.is_over = False
        self.reward = 0
        self.state = []
        if self.curr_time.date() != self.last_reset_date:
            self.daily_profit = 0
            self.daily_loss = 0
            self.last_reset_date = self.curr_time.date()
        self._assemble_state()
        obs = np.array(self.state, dtype=np.float32)
        info = {}
        return obs, info

class GameGymWrapper(gym.Env):
    def __init__(self, game_instance):
        super().__init__()
        self.game = game_instance
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(219,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, _ = self.game.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.game.step(action)
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        pass


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_dqn(env, logger, save_path="models"):
    input_dim = env.window_size
    model = DQN(input_dim=input_dim, output_dim=3)
    target_model = DQN(input_dim=input_dim, output_dim=3)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'])
    buffer = ReplayBuffer()
    
    gamma = MODEL_PARAMS['gamma']
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    batch_size = MODEL_PARAMS['batch_size']
    episodes = MODEL_PARAMS.get('episodes', 200)
    max_steps_per_episode = MODEL_PARAMS.get('max_steps_per_episode', 1000)
    
    episode_rewards = []
    training_losses = []
    
    logger.info(f"Starting DQN training with {episodes} episodes")
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, input_dim])
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            step_count += 1
            
            if np.random.rand() < epsilon:
                action = np.random.choice(3)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, input_dim])
            
            buffer.push((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state
            
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                s, a, r, ns, d = zip(*batch)
                
                s = torch.tensor(np.concatenate(s), dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64)
                r = torch.tensor(r, dtype=torch.float32)
                ns = torch.tensor(np.concatenate(ns), dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)
                
                q_values = model(s).gather(1, a.unsqueeze(1)).squeeze()
                max_q = target_model(ns).max(1)[0]
                target = r + gamma * max_q * (1 - d)
                
                loss = nn.MSELoss()(q_values, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                training_losses.append(loss.item())
        
        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            logger.info(f"Episode {episode}: Reward={total_reward:.2f}, Epsilon={epsilon:.4f}")
            
            # Plot training curves every 10 episodes
            plot_training_curves(
                episode_rewards,
                training_losses,
                save_path=os.path.join(save_path, "training_curves")
            )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"dqn_model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"DQN model saved to {model_path}")
    
    return model

def train_ppo(env, logger, save_path="models"):
    # Store the original csv_path and window_size before wrapping
    bars30m,bars4h,bars1d,reward_function,lkbk,init_idx = env.bars30m,env.bars4h,env.bars1d,env.reward_function,env.lkbk,env.init_idx
    # Create a function that returns a new environment instance
    def make_env():
        env = Game(
            bars30m,
            bars4h, 
            bars1d,
            reward_function,
            lkbk,
            init_idx
        )
        wrapped = GameGymWrapper(env)
        env = Monitor(wrapped)
        return env
    
    # Create vectorized environment with multiple instances
    n_envs = MODEL_PARAMS['n_envs']
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = Game(
            bars30m,
            bars4h, 
            bars1d,
            reward_function,
            lkbk,
            init_idx
        )
    eval_env = GameGymWrapper(eval_env)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    # Setup tensorboard logging
    tensorboard_log = os.path.join(save_path, "tensorboard")
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Create custom callback to track rewards and losses
    class TrainingCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rewards = []
            self.losses = []
            
        def _on_step(self):
            result = super()._on_step()

            # Log training rewards from rollout
            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                if isinstance(rewards, (np.ndarray, list)) and len(rewards) > 0:
                    self.rewards.append(np.mean(rewards))

            if len(self.rewards) % 100 == 0:
                plot_training_curves(
                    self.rewards,
                    self.losses,
                    save_path=os.path.join(self.log_path, "training_curves")
                )

            return result

    
    eval_callback = TrainingCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create model with GPU configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=MODEL_PARAMS['learning_rate'],
        n_steps=MODEL_PARAMS['n_steps'],
        batch_size=MODEL_PARAMS['batch_size'],
        n_epochs=MODEL_PARAMS['n_epochs'],
        gamma=MODEL_PARAMS['gamma'],
        gae_lambda=MODEL_PARAMS['gae_lambda'],
        clip_range=MODEL_PARAMS['clip_range'],
        ent_coef=MODEL_PARAMS['ent_coef'],
        tensorboard_log=tensorboard_log,
        device=MODEL_PARAMS['device'],
        policy_kwargs=MODEL_PARAMS['policy_kwargs']
    )
    
    logger.info("Starting PPO training...")
    model.learn(
        total_timesteps=MODEL_PARAMS['total_timesteps'],
        callback=eval_callback
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"ppo_model_{timestamp}")
    model.save(model_path)
    env.save(f"{save_path}/vec_normalize_{timestamp}.pkl")
    logger.info(f"PPO model saved to {model_path}")
    
    return model

def load_mt5_data(symbol='EURUSD', timeframes=['M30', 'H4', 'D1']):
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
    
    # Load data for each timeframe
    data = {}
    for tf in timeframes:
        filename = f"data/{symbol}_{tf.lower()}.csv"
        try:
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            
            
            data[tf] = df
        except Exception as e:
            print(f"Error loading data for {symbol} {tf}: {e}")
            return None, None, None
    
    # Return data in the format expected by Game
    return data['M30'], data['H4'], data['D1']

def plot_training_curves(rewards, losses, save_path="training_curves"):
    """
    Plot and save training curves showing rewards and losses over epochs.
    
    Args:
        rewards (list): List of rewards per episode
        losses (list): List of training losses
        save_path (str): Directory to save the plots
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.title('Training Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'rewards_{timestamp}.png'))
    plt.close()
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'losses_{timestamp}.png'))
    plt.close()

def main():
        logger = setup_logging('train_from_csv')
        create_directories()
    
    # try:
        parser = argparse.ArgumentParser(description='Train a DQN or PPO model from CSV data')
        parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'], help='Algorithm to use (dqn or ppo)')
        parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol (default: EURUSD)')
        args = parser.parse_args()

        df30m, df4h, df1d = load_mt5_data(args.symbol, ['M30', 'H4', 'D1'], )

        
        env = Game(
            bars30m=df30m,
            bars4h=df4h, 
            bars1d=df1d,
            reward_function=reward_function,
            lkbk=100,
            init_idx= 101
        )
        
        # Train model
        if args.algorithm == "dqn":
            model = train_dqn(env, logger, PATHS['models_dir'])
        else:
            model = train_ppo(env, logger, PATHS['models_dir'])
        
        logger.info("Training completed successfully!")
        
    # except Exception as e:
    #     logger.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 