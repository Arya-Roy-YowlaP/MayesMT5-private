import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from datetime import datetime
import MetaTrader5 as mt5
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from game_environment import Game
from utils import setup_logging, create_directories
from config import MODEL_PARAMS, TRADING_PARAMS, ENV_PARAMS, PATHS
from visualization import plot_training_progress

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

def create_training_env(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, algorithm="ppo"):
    """Create and wrap the training environment"""
    env = Game(
        symbol=symbol,
        timeframe=timeframe,
        window_size=ENV_PARAMS['window_size']
    )
    
    if algorithm == "ppo":
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    return env

def train_dqn(env, logger, save_path="models"):
    """Train DQN model"""
    # Setup
    input_dim = env.window_size
    model = DQN(input_dim=input_dim, output_dim=3)
    target_model = DQN(input_dim=input_dim, output_dim=3)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'])
    buffer = ReplayBuffer()
    
    # Training parameters
    gamma = MODEL_PARAMS['gamma']
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    batch_size = MODEL_PARAMS['batch_size']
    episodes = MODEL_PARAMS.get('episodes', 200)
    max_steps_per_episode = MODEL_PARAMS.get('max_steps_per_episode', 1000)
    
    # Track progress
    episode_rewards = []
    training_losses = []
    
    logger.info(f"Starting DQN training with {episodes} episodes")
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, input_dim])
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            step_count += 1
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(3)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()
            
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, input_dim])
            
            buffer.push((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state
            
            # Train on batch
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
            
            # Plot training progress
            plot_training_progress(
                episode_rewards,
                training_losses,
                algorithm="dqn",
                save=True
            )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"dqn_model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"DQN model saved to {model_path}")
    
    return model

def train_ppo(env, logger, save_path="models"):
    """Train PPO model"""
    # Create evaluation environment
    eval_env = create_training_env(algorithm="ppo")
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Initialize model
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
    
    # Train model
    logger.info("Starting PPO training...")
    model.learn(
        total_timesteps=MODEL_PARAMS['total_timesteps'],
        callback=eval_callback
    )
    
    # Save model and normalization stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"ppo_model_{timestamp}")
    model.save(model_path)
    env.save(f"{save_path}/vec_normalize_{timestamp}.pkl")
    logger.info(f"PPO model saved to {model_path}")
    
    return model

def main():
    # Setup
    logger = setup_logging('unified_train')
    create_directories()
    
    # Initialize MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return
    
    try:
        # Get algorithm choice
        algorithm = input("Choose algorithm (dqn/ppo): ").lower()
        if algorithm not in ["dqn", "ppo"]:
            raise ValueError("Invalid algorithm choice. Choose 'dqn' or 'ppo'")
        
        # Create environment
        env = create_training_env(
            symbol=TRADING_PARAMS['symbol'],
            timeframe=TRADING_PARAMS['timeframe'],
            algorithm=algorithm
        )
        
        # Train model
        if algorithm == "dqn":
            model = train_dqn(env, logger, PATHS['models_dir'])
        else:
            model = train_ppo(env, logger, PATHS['models_dir'])
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main() 