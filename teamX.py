import numpy as np
from collections import deque
import cv2
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # typically 4 for stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the output size of CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Final linear layer to get to desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# Build a Python class for your solution, do preprocessing (image processing, frame stacking, etc) here.
# During the competition, only the policy function is called at each time step, providing the observation and reward for that time step only.
# Your agent is expected to return actions to be executed.
class TeamX:

    def __init__(self, frame_stack=4):
        self.frames = deque(maxlen=frame_stack)

        custom_objects = {
             # Standard mappings for SB3 custom feature extractors
             "policy": {"features_extractor_class": CustomCNN},
             "features_extractor_class": CustomCNN,

             # --- Explicit mapping based on how the model was RESAVED ---
             # If resave_model.py was the main script, the reference is likely "__main__.CustomCNN"
             "__main__.CustomCNN": CustomCNN
             # If you encountered "No module named 'mylib'" BEFORE resaving, use that:
             # "mylib.CustomCNN": CustomCNN
        }

        # --- Load the trained policy network ---
        model_path = r"C:\Users\user\Desktop\Pickleball submission\a2c_Servingbot_left_servedown_RESAVED.zip"
        print(f"Attempting to load model from: {model_path}")

        self.model = A2C.load(model_path, custom_objects = custom_objects)


    # Your policy takes only visual representation as input,
    # and reward is 1 when you score, -1 when your opponent scores
    # Your policy function returns actions
    def policy(self, observation, reward):
        # Implement your solution here

        # image processing
        obs = observation.transpose(1, 2, 0) # Transpose from (C, H, W) → (H, W, C)
        obs = cv2.resize(obs, (168, 84), interpolation=cv2.INTER_AREA) # Resize
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        obs = obs.astype(np.float32)/255.0  # Normalize to [0, 1]

        self.frames.append(obs)

        # Ensure the deque is full before predicting
        if len(self.frames) < self.frames.maxlen:
            # On the first few steps, fill the deque with the current frame
            while len(self.frames) < self.frames.maxlen:
                self.frames.append(obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        # Use your policy network here
        # Task 3: Use the model to predict the action
        action, _ = self.model.predict(stacked_obs, deterministic=True)

        return action


