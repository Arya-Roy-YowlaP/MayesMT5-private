import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from game_environment import Game
import MetaTrader5 as mt5

print("Starting DQN training...")

# --- Define DQN ---
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

# --- Define Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- Setup Environment ---
print("Initializing environment...")
env = Game(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, window_size=20)
print("Environment initialized successfully")

# --- Model + Training Setup ---
input_dim = env.window_size
print(f"Input dimension: {input_dim}")
model = DQN(input_dim=input_dim, output_dim=3)
target_model = DQN(input_dim=input_dim, output_dim=3)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=0.001)
buffer = ReplayBuffer()

# --- Hyperparameters ---
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
episodes = 200
max_steps_per_episode = 1000  # Limit steps per episode

print("Starting training loop...")
print(f"Total episodes: {episodes}")
print(f"Batch size: {batch_size}")
print(f"Initial epsilon: {epsilon}")
print(f"Max steps per episode: {max_steps_per_episode}")

# --- Training Loop ---
for episode in range(episodes):
    print(f"\nEpisode {episode + 1}/{episodes}")
    state = env.reset()
    state = np.reshape(state, [1, input_dim])
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < max_steps_per_episode:
        step_count += 1
        if np.random.rand() < epsilon:
            action = np.random.choice(3)
            print(f"Step {step_count}: Random action: {['HOLD', 'SELL', 'BUY'][action]}")
        else:
            with torch.no_grad():
                action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()
            print(f"Step {step_count}: Model action: {['HOLD', 'SELL', 'BUY'][action]}")

        next_state, reward, done = env.step(action)
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
            
            print(f"Step {step_count}: Loss: {loss.item():.4f}")

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"\nEpisode {episode} Summary:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Epsilon: {epsilon:.4f}")
        print(f"Steps taken: {step_count}")

# --- Save Trained Model ---
print("\nTraining completed!")
print("Saving model...")
torch.save(model.state_dict(), "dqn_model.pt")
print("Model saved to dqn_model.pt")
env.close()
print("Environment closed")
