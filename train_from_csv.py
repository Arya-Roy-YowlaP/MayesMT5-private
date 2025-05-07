import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces
import gymnasium as gym
from utils import setup_logging, create_directories
from config import MODEL_PARAMS, TRADING_PARAMS, ENV_PARAMS, PATHS
from visualization import plot_training_progress

class CSVGameEnv(gym.Env):
    def __init__(self, csv_path, window_size=10):
        super().__init__()
        self.csv_path = csv_path  # Store csv_path as instance variable
        self.window_size = window_size
        self.current_idx = window_size
        self.position = 0
        self.entry_price = 0
        self.position_hold_time = 0
        self.max_position_hold_time = 100
        self.profit_threshold = 0.02
        self.loss_threshold = -0.01
        
        # Load and prepare data
        self.df = pd.read_csv(csv_path)
        if 'time' in self.df.columns:
            self.df['time'] = pd.to_datetime(self.df['time'])
            self.df.set_index('time', inplace=True)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: SELL, 2: BUY
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size,), 
            dtype=np.float32
        )
        
    def get_state(self):
        window = self.df['close'].iloc[self.current_idx-self.window_size:self.current_idx].values
        state = (window - window.mean()) / (window.std() + 1e-6)
        return state.astype(np.float32)
    
    def step(self, action):
        current_price = self.df['close'].iloc[self.current_idx]
        
        reward = 0
        if action == 2 and self.position <= 0:  # BUY
            self.position = 1
            self.entry_price = current_price
            self.position_hold_time = 0
            reward = 0
        elif action == 1 and self.position >= 0:  # SELL
            self.position = -1
            self.entry_price = current_price
            self.position_hold_time = 0
            reward = 0
        elif action == 0:  # HOLD
            if self.position > 0:  # Long position
                reward = (current_price - self.entry_price) / self.entry_price
            elif self.position < 0:  # Short position
                reward = (self.entry_price - current_price) / self.entry_price
            self.position_hold_time += 1
        
        self.current_idx += 1
        next_state = self.get_state()
        
        terminated = False
        truncated = False
        info = {}
        
        # End conditions
        if self.position != 0 and self.position_hold_time >= self.max_position_hold_time:
            terminated = True
            info['reason'] = f"Position held too long ({self.position_hold_time} steps)"
        if reward >= self.profit_threshold:
            terminated = True
            info['reason'] = f"Profit target reached ({reward*100:.2f}%)"
        if reward <= self.loss_threshold:
            terminated = True
            info['reason'] = f"Stop loss hit ({reward*100:.2f}%)"
        if self.current_idx >= len(self.df) - 1:
            terminated = True
            info['reason'] = "End of data"
        
        return next_state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.window_size
        self.position = 0
        self.entry_price = 0
        self.position_hold_time = 0
        return self.get_state(), {}

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
            
            plot_training_progress(
                episode_rewards,
                training_losses,
                algorithm="dqn",
                save=True
            )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"dqn_model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"DQN model saved to {model_path}")
    
    return model

def train_ppo(env, logger, save_path="models"):
    # Store the original csv_path before wrapping
    original_csv_path = env.csv_path
    
    # Wrap the environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment using the original csv_path
    eval_env = CSVGameEnv(csv_path=original_csv_path, window_size=env.envs[0].env.window_size)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
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
        tensorboard_log=save_path
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

def main():
    logger = setup_logging('train_from_csv')
    create_directories()
    
    try:
        # List available CSV files in data directory
        data_dir = 'data'
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory '{data_dir}' not found. Please run export_mt5_data.py first.")
            
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in '{data_dir}' directory. Please run export_mt5_data.py first.")
            
        # Display available files
        print("\nAvailable data files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
            
        # Get user selection
        while True:
            try:
                choice = int(input("\nSelect a file number: "))
                if 1 <= choice <= len(csv_files):
                    csv_path = os.path.join(data_dir, csv_files[choice-1])
                    break
                else:
                    print(f"Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print("Please enter a valid number")
        
        # Get algorithm choice
        while True:
            algorithm = input("\nChoose algorithm (dqn/ppo): ").lower()
            if algorithm in ["dqn", "ppo"]:
                break
            print("Invalid choice. Please enter 'dqn' or 'ppo'")
        
        # Create environment
        logger.info(f"Loading data from {csv_path}")
        env = CSVGameEnv(
            csv_path=csv_path,
            window_size=ENV_PARAMS['window_size']
        )
        
        # Train model
        if algorithm == "dqn":
            model = train_dqn(env, logger, PATHS['models_dir'])
        else:
            model = train_ppo(env, logger, PATHS['models_dir'])
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 