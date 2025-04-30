import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from game_environment import Game
import os
from datetime import datetime

def create_training_env(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1):
    """Create and wrap the training environment"""
    env = Game(symbol=symbol, timeframe=timeframe)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env

def train_model(env, total_timesteps=1000000, save_path="models"):
    """Train the PPO model with evaluation callback"""
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Create evaluation environment
    eval_env = create_training_env()
    
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
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=save_path
    )
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Save final model and normalization stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"{save_path}/ppo_trading_model_{timestamp}")
    env.save(f"{save_path}/vec_normalize_{timestamp}.pkl")
    
    return model

def main():
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    try:
        # Create training environment
        env = create_training_env()
        
        # Train model
        print("Starting training...")
        model = train_model(env)
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main() 