from stable_baselines3 import PPO, SAC, TD3
import numpy as np

class ModelManager:
    def __init__(self, algorithm, policy_type=None, env=None, tensorboard_log=None, config=None):
        self.algorithm = algorithm
        self.policy_type = policy_type
        self.env = env
        self.tensorboard_log = tensorboard_log
        self.config = config

    def create_model(self):
        if self.config:
            initial_lr = self.config["learning_rate"]

            # lambda functions use progress, which represents the progress remaining, as required by the Stable Baselines 3 API.
            # When training starts, progress is 1 (100% of training remaining), and it should decrease linearly to 0 by the end of training

            # Linear schedule
            if self.config["schedule"] == "linear":
                final_lr = self.config.get("final_lr", 1e-5)
                learning_rate = lambda progress: initial_lr + (final_lr - initial_lr) * (1 - progress)

            # Exponential decay schedule
            elif self.config["schedule"] == "exponential":
                decay_rate = self.config.get("decay_rate", 0.5)
                learning_rate = lambda progress: initial_lr * (decay_rate ** (1 - progress))

            # Cosine annealing schedule
            elif self.config["schedule"] == "cosine":
                min_lr = self.config.get("min_lr", 1.5e-4)
                learning_rate = lambda progress: min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * (1 - progress)))
                                                 
            # Constant schedule (default)
            else:
                learning_rate = initial_lr

        else:
            # Default learning rate if config is not provided
            learning_rate = lambda progress: 0.0003

        # Instantiate the model with the specified learning rate
        if self.algorithm.lower() == 'ppo':
            # Define the initial and final values for the clip range
            initial_clip_range = 0.5
            final_clip_range = 0.1

            # Use a lambda function to define an exponentially decreasing clip range
           # clip_range_lambda = lambda progress_remaining: final_clip_range + (initial_clip_range - final_clip_range) * np.exp(-10 * (1 - progress_remaining))
            clip_range_lambda = 0.2

            return PPO(self.policy_type, self.env, verbose=1, tensorboard_log=self.tensorboard_log, learning_rate=learning_rate, clip_range=clip_range_lambda, ent_coef=0.01)#, clip_range=0.1) #clip_range=lambda progress: 0.2 * (0.4**(1 - progress)))# clip_range will decrease exp from 0.2 to 0 by the end of training
        elif self.algorithm.lower() == 'sac':
            return SAC(self.policy_type, self.env, verbose=1, tensorboard_log=self.tensorboard_log, learning_rate=learning_rate)
        elif self.algorithm.lower() == 'td3':
            return TD3(self.policy_type, self.env, verbose=1, tensorboard_log=self.tensorboard_log, learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def load_model(self, path):
        if self.algorithm.lower() == 'ppo':
            return PPO.load(path)
        elif self.algorithm.lower() == 'sac':
            return SAC.load(path)
        elif self.algorithm.lower() == 'td3':
            return TD3.load(path)
        else:
            raise ValueError(f"Unsupported algorithm for loading: {self.algorithm}")
