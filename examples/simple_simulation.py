# run the environment "TaxationEnv" with no learning agent
# python simple_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import the environment
from src.TaxationEnv import TaxationEnv


# initialise the environment
env = TaxationEnv(use_wandb=False, reward_type="R2", alpha=0)


media = []
rewards = []
ginis = []

initial_wealth = env.wealth


for i in range(1000):
    # choose an action

    #action = env.action_space.sample()
    action = np.array([0.0, 1.0, 0.0])

    # take the action
    observation, reward, done, info = env.step(action)
    mean_wealth = np.mean(env.wealth)
    # render the environment
    print(f"Indice di Gini: {env.gini_index * mean_wealth:.4f}")
    print(f"Media della Ricchezza: {np.mean(env.wealth):.2f}")

    media.append(np.mean(env.wealth)) 
    rewards.append(reward)
    ginis.append(env.gini_index * mean_wealth)

    if done:
        print(f"Episode finished after {i} timesteps")
        break


plt.plot(media)
plt.title("Wealth mean")
plt.show()

plt.plot(rewards)
plt.title("Reward")
plt.show()

plt.plot(ginis)
plt.title("Gini index")
plt.show()

# histogram of wealth  
plt.hist(initial_wealth, bins=50, alpha=0.5, label='Initial')
plt.hist(env.wealth, bins=50, alpha=0.5, label='Final')
plt.legend(loc='upper right')
plt.show()
