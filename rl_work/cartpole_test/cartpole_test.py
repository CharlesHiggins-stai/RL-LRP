import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Step 1: Create the environment
env = gym.make("CartPole-v1")

# Step 2: Define a custom callback to extract and train the value network
class CustomCallback(BaseCallback):
    def __init__(self, update_freq, custom_training_fn, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.update_freq = update_freq
        self.custom_training_fn = custom_training_fn
        self.update_count = 0

    def _on_step(self) -> bool:
        self.update_count += 1
        if self.update_count % self.update_freq == 0:
            # Extract the value network
            value_net = self.model.policy.value_net
            
            # Run the custom training loop
            self.custom_training_fn(value_net)

        return True

# Step 3: Define a custom training function for the value network
def custom_training_fn(value_net):
    # Example custom training loop
    optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Dummy data for illustration purposes
    dummy_states = torch.randn(10, env.observation_space.shape[0])
    dummy_targets = torch.randn(10, 1)

    value_net.train()
    for _ in range(10):  # Run for 10 epochs
        optimizer.zero_grad()
        predictions = value_net(dummy_states)
        loss = loss_fn(predictions, dummy_targets)
        loss.backward()
        optimizer.step()

# Step 4: Instantiate the PPO agent with the custom callback
model = PPO("MlpPolicy", env, verbose=1)

# Define the callback
callback = CustomCallback(update_freq=1000, custom_training_fn=custom_training_fn)

# Step 5: Train the agent
model.learn(total_timesteps=10000, callback=callback)

# Save the model
model.save("ppo_cartpole")

# Done
env.close()
