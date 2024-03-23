import gym
import numpy as np

class NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizeActionWrapper, self).__init__(env)
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def action(self, action):
        # Normalize action from [-1, 1] to [action_space_lower_bound, action_space_upper_bound]
        norm_action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        norm_action = np.clip(norm_action, self.action_low, self.action_high)
        return norm_action

    def reverse_action(self, action):
        # Reverse normalization if needed
        rev_action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        return rev_action
