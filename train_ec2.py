import os
import sys
import logging
from datetime import datetime
from stable_baselines3 import PPO
from game_environment import Game
from config import MODEL_PARAMS, ENV_PARAMS, TRADING_PARAMS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # Create timestamp for model saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize environment
        logger.info("Initializing trading environment...")
        env = Game(
            symbol=TRADING_PARAMS['symbol'],
            timeframe=TRADING_PARAMS['timeframe'],
            window_size=ENV_PARAMS['window_size']
        )
        
        # Initialize PPO model
        logger.info("Initializing PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=MODEL_PARAMS['learning_rate'],
            n_steps=MODEL_PARAMS['n_steps'],
            batch_size=MODEL_PARAMS['batch_size'],
            n_epochs=MODEL_PARAMS['n_epochs'],
            gamma=MODEL_PARAMS['gamma'],
            gae_lambda=MODEL_PARAMS['gae_lambda'],
            clip_range=MODEL_PARAMS['clip_range'],
            ent_coef=MODEL_PARAMS['ent_coef'],
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
        
        # Train the model
        logger.info(f"Starting training for {MODEL_PARAMS['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=MODEL_PARAMS['total_timesteps'],
            progress_bar=True
        )
        
        # Save the model
        model_path = os.path.join(model_dir, f"ppo_trading_model_{timestamp}")
        logger.info(f"Saving model to {model_path}")
        model.save(model_path)
        
        # Save training configuration
        config_path = os.path.join(model_dir, f"training_config_{timestamp}.txt")
        with open(config_path, 'w') as f:
            f.write("Training Configuration:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Timesteps: {MODEL_PARAMS['total_timesteps']}\n")
            f.write(f"Learning Rate: {MODEL_PARAMS['learning_rate']}\n")
            f.write(f"Batch Size: {MODEL_PARAMS['batch_size']}\n")
            f.write(f"Window Size: {ENV_PARAMS['window_size']}\n")
            f.write(f"Symbol: {TRADING_PARAMS['symbol']}\n")
            f.write(f"Timeframe: {TRADING_PARAMS['timeframe']}\n")
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False
    finally:
        env.close()

if __name__ == "__main__":
    train_model() 