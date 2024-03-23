import numpy as np
import plotly.express as px
import pandas as pd
import pickle
#import src.TaxationEnv as TaxationEnv
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import wandb
from gym.utils.seeding import np_random
import os
import sys
from enum import Enum
import time
from stable_baselines3 import PPO, SAC  
import csv


class RewardType(Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    
class TaxationEnv(gym.Env):
    """
    Envirnoment for taxation policy optimization.
    """

    def __init__(self,
                 pop_size=1000,
                 num_states=9,
                 inital_taxes_params=(0.01, 0.5, 0.5), # T_max, T_esp, T_scale
                 consumptions_params=(0.8, 5), # eta, K
                 returns_params=(0.04, 0.2), #mean, std
                 episode_steps=500,
                 percentiles=[1, 10, 25, 50, 75, 90, 95, 99, 99.9],
                 use_wandb=False,
                 alpha=1,  # param for convex combination btw Gini and consumption
                 reward_type=RewardType.R2,
                 exp_salaries_params=(10, 1), # mean, std
                 wealth_init_params=(12, 2), # mean, std
                 action_space_lower_bound = np.array([0, 0, 0.01]),# T_max, T_esp, T_scale
                 action_space_upper_bound = np.array([0.10, 3, 1]),# T_max, T_esp, T_scale
                 render_mode=None
                 ):
        super(TaxationEnv, self).__init__()

        # Definizione della grandezza della popolazione e dei parametri di default
        self.pop_size = pop_size
        self.inital_taxes_params = inital_taxes_params
        self.consumptions_params = consumptions_params
        self.returns_params = returns_params
        self.exp_salaries_params = exp_salaries_params
        self.expected_salaries = np.random.lognormal(self.exp_salaries_params[0], self.exp_salaries_params[1], size=self.pop_size)
        self.gini_salari = self._calculate_gini(self.expected_salaries)
        self.episode_steps = episode_steps # Define the maximum number of steps for each episode
        self.percentiles = percentiles
        self.steps = 0  # Initialize step counter for each episode.
        self.num_states = num_states
        self.episode_counter = 1  # Initialize episode counter
        self.current_episode_reward = 0  # Initialize cumulative reward for the new episode
        self.gini_index = None  # Initialize Gini index
        self.use_wandb = use_wandb
        self.alpha = alpha
        self.avg_rel_consumptions = None
        self.reward_type = reward_type
        self.initial_wealth = None
        self.wealth_init_params = wealth_init_params
        self.action_space_lower_bound = action_space_lower_bound
        self.action_space_upper_bound = action_space_upper_bound
        self.render_mode = render_mode
        self.average_tax_rate = None
        self.tax_rates_perc = None
        self.W_scale = None
        self.cv = None
        #self.salaries = None
        self.average_savings_after_taxes = None
        self.average_consumption_normalized = None

        # Definizione dello spazio delle azioni e dello spazio degli stati
        self.action_space = spaces.Box(low=self.action_space_lower_bound, high=self.action_space_upper_bound, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_states,), dtype=np.float32)

        # Inizializzazione della popolazione
        self.reset()

    def reset(self):
        # Inizializza o reimposta lo stato dell'ambiente
        self.wealth = np.random.lognormal(self.wealth_init_params[0], self.wealth_init_params[1], size=self.pop_size)
        
        if(self.episode_counter == 1 and self.steps == 0):
            self.initial_wealth = self.wealth

        obs = self._get_observations(self.wealth)
        self.steps = 0 # Reset step counter for each episode.
        self.steps_accumulator = []  # Initialize accumulator for steps for logging
        #self.episode_counter += 1  # Increment episode counter at the start of a new episode
        self.current_episode_reward = 0  # Reset cumulative reward for the new episode
        return obs

    def step(self, action):
        # Increment the step counter for each episode.
        self.steps += 1
        # Apply the taxation policy: get new_taxes_params
        self._apply_taxation(action)
        # Update also the W_scale parameter:
        self.W_scale = np.percentile(self.wealth, self.new_taxes_params[2]*100)
        # Update the wealth based on the new policy, after evalutaing consumptions and taxes on the current wealth:
        self.wealth = self._update_wealth()

        # Get the new observation of the environment.
        obs = self._get_observations(self.wealth)
        self.gini_index = self._calculate_gini(self.wealth)
        # Calculate the reward based on the outcome of the new policy.
        reward = self._calculate_reward()

        # Accumulate reward for the current episode
        self.current_episode_reward += reward  

        # Terminal conditions
        if (self.steps >= self.episode_steps):
            done = True
        # elif (np.sum(self.wealth) > np.exp(50)):
        #     done = False
        elif (np.sum(self.wealth) >= np.finfo(np.float32).max):
            done = False
            print("Warning: Wealth has reached the maximum value of float32: 3.4028235e+38")
        else:
            done = False

        # Log metrics at the end of each episode
        if done:
            # if self.use_wandb:
            #     wandb.log({
            #         "Episode": self.episode_counter,
            #         "Average Reward per Episode": self.current_episode_reward / self.steps,
            #         "Episode Length": self.steps,
            #         "Final Gini Index": self.gini_index,
            #         "Final T_max": action[0],
            #         "Final T_esp": action[1],
            #         "Final T_scale": action[2],
            #         "Final Wealth Mean": np.mean(self.wealth),
            #         "Final Wealth Median": np.median(self.wealth),
            #     })
            self.episode_counter += 1  # Increment episode counter at the end of an episode

        # Create the info dictionary to include additional data
        info = {
            'gini_index': self.gini_index,
            'reward': reward,
            'T_max': action[0],
            'T_esp': action[1],
            'T_scale': action[2],
            'total_wealth': np.sum(self.wealth),
            'avg_rel_consumptions': self.avg_rel_consumptions,
            'wealth_mean': np.mean(self.wealth),
            'wealth_median': np.median(self.wealth),
            'new_state': obs,  # Include percentiles data,
            'perc_consumptions': self.perc_consumptions,
            'tax_rates_perc': self.tax_rates_perc,
            'average_tax_rate': self.average_tax_rate,
            'W_scale': self.W_scale,
            'cv': self.cv,
            'R1': self.average_savings_after_taxes,
            'R2': self.average_consumption_normalized
        }

        return obs, reward, done, info

    def _get_observations(self, wealth):
        # Fai un check che il numero di percentiles sia uguale al numero di stati altrimenti raise ValueError
        if (len(self.percentiles) != self.num_states):
            raise ValueError("The number of percentiles must be equal to the number of states")

        if (self.percentiles):
            stato_t = [np.log(np.percentile(wealth,x)) for x in self.percentiles]
        else:
            self.percentiles = list(np.logspace(0, 1.99, num=self.num_states, endpoint=True))
            stato_t = [np.log(np.percentile(wealth,x)) for x in self.percentiles]
        # [[ 1.  1.77316847  3.14412642  5.57506583  9.88553095 17.52871178, 31.08135903 55.11248581 97.7237221 ]
        return stato_t

    def _taxes(self, wealth, taxes_params):
        max, esp, _ = taxes_params
        # old version:
        # ts = max * (1 - 1 / (1 + (wealth / np.exp(scale*25))**esp))
        # Pick the true scale parameter as the wealth at the scale_percentile
        # scale = np.percentile(wealth, scale_percentile*100)# returns a variable of dimension 1
        # new version:
        ts = max * (1 - 1 / (1 + (wealth / self.W_scale)**esp))
        return ts

    def _consumptions(self, wealth, expected_salaries, taxes_params):
        # Calcola i consumi relativi alla ricchezza
        η, K = self.consumptions_params
        # cons = 1 / (1 + ((wealth / (K * expected_salaries)) *
        #                  (1 - self._taxes(wealth, taxes_params)))**η)
        
        # New form:
        cons = 1 / (1 + ((wealth * (1/(K * expected_salaries) + 10**-5)) *
                         (1 - self._taxes(wealth, taxes_params)))**η)
        return cons

    def _apply_taxation(self, action):
        # Applica la logica della tassazione qui
        self.new_taxes_params = action

    def _calculate_reward(self):

        gini_index = self._calculate_gini(self.wealth)

        # reward_type is an enum of "R1" or "R2"
        if(self.reward_type == RewardType.R2):

            average_consumption_normalized = np.mean(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params) * (self.wealth / (self.consumptions_params[1] * self.expected_salaries)))

            #average_consumption_not_normalized = np.mean(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)*self.wealth)

            #reward = -self.alpha * gini_index + (1 - self.alpha) * average_consumption_normalized

            self.average_consumption_normalized = average_consumption_normalized

            #new version:
            reward = -self.alpha * (gini_index - self.gini_salari)/(1-self.gini_salari) + (1 - self.alpha) * average_consumption_normalized

            # use the cofficient of variation as a measure of inequality instead of gini index:
            self.cv = np.std(self.wealth)/np.mean(self.wealth)
            #reward = -self.alpha * self.cv + (1 - self.alpha) * average_consumption_normalized

        elif(self.reward_type == RewardType.R3):

            average_consumption_normalized = np.mean(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params) * (self.wealth / (self.consumptions_params[1] * self.expected_salaries)))
            self.average_consumption_normalized = average_consumption_normalized

            # use the cofficient of variation as a measure of inequality instead of gini index:
            self.cv = np.std(self.wealth)/np.mean(self.wealth)
            reward = -self.alpha * self.cv + (1 - self.alpha) * average_consumption_normalized

        elif(self.reward_type == RewardType.R1): 

            #taxes = self._taxes(self.wealth, self.new_taxes_params)
            self.cv = np.std(self.wealth)/np.mean(self.wealth)

            consumptions = self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)

            # average_savings_after_taxes = np.mean((1 - consumptions) * (1 - taxes))

            average_savings_after_taxes = np.mean((1 - consumptions))
            self.average_savings_after_taxes = average_savings_after_taxes

            #new version:
            reward = -self.alpha * (gini_index - self.gini_salari)/(1-self.gini_salari) + (1 - self.alpha) * average_savings_after_taxes



        else:
            raise ValueError("Reward type must be a enum of 'R1' or 'R2'")
        return reward

    def _salaries(self, mean_sal, osc=0.2):

        mean_sal = np.asarray(mean_sal)
        mean_sal[mean_sal<0] = 0

        sal = mean_sal*np.random.lognormal(0, osc, size=mean_sal.shape)

        return sal

    def _update_wealth(self):
        # Generate returns:
        μ_r, σ_r = self.returns_params
        returns = np.random.lognormal(μ_r, σ_r, size=self.pop_size)

        # Calculate taxes and consumptions with current wealth and expected salaries
        taxes = self._taxes(self.wealth, self.new_taxes_params)
        consumptions = self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)
        
        #FOR LOGGING:
        self.avg_rel_consumptions = np.sum(consumptions)/self.pop_size #it is the average relative consumption
        self.average_tax_rate = np.mean(taxes)
        # Calculate the tax rates for each percentile:
        self.tax_rates_perc = [self._taxes(np.percentile(self.wealth, x), self.new_taxes_params) for x in self.percentiles]
        # Calculate wealth percentiles
        wealth_percentiles = np.percentile(self.wealth, self.percentiles)
        # Find indices of the closest wealth values in self.wealth for each percentile
        closest_indices = np.array([np.abs(self.wealth - wp).argmin() for wp in wealth_percentiles])
        # Use these indices to extract the corresponding expected salaries
        salaries_percentiles = self.expected_salaries[closest_indices]
        # Call _consumptions with the matched wealth and salary percentile values
        self.perc_consumptions = self._consumptions(wealth_percentiles, salaries_percentiles, self.new_taxes_params)
        

        # Update wealth
        salaries = self._salaries(self.expected_salaries)

        new_wealth = ( salaries + self.wealth * returns * (1 - consumptions) *(1 - taxes))

        # Aggiungi a new wealth un termine lotteria del tipo:
        # new_wealth += 1e12 * (np.random.uniform(size=self.pop_size) < 1e-5)

        return new_wealth

    def _calculate_gini(self, wealth):
        # Ordina la wealth in ordine crescente
        sorted_wealth = np.sort(wealth)
        n = len(wealth)
        cum_wealth = np.cumsum(sorted_wealth, dtype=float)
        # Calcola l'indice di Gini come rapporto delle aree sul grafico di Lorenz
        gini_index = (n + 1 - 2 * np.sum(cum_wealth) / cum_wealth[-1]) / n
        return gini_index

    def calculate_gini(self):
        return self._calculate_gini(self.wealth)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    #Parse command line arg and take alpha or defualt 0.5, and reward type or default R1 and the path where to pick the model
    if len(sys.argv) == 5:
        alpha = float(sys.argv[1])
        reward_type = sys.argv[2]
        algorithm = sys.argv[3]
        file_path = sys.argv[4]

    else:
        raise RuntimeError("You must provide alpha, reward type, algorithm, and file_path as command line arguments")
    
    if not os.path.exists(file_path):
        RuntimeError("The file does not exist")
    # Load the trained model
    if algorithm.lower() == 'ppo':
        model = PPO.load(file_path)
    elif algorithm.lower() == 'sac':
        model = SAC.load(file_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Assuming TaxationEnv is already defined and imported
    env = TaxationEnv(reward_type=RewardType[reward_type], use_wandb=False)

    # Define a wrapper function for the environment's step method
    def simulate():
        obs = env.reset()
        inequality_history = []
        consumption_history = []
        gini_sal = env.gini_salari
        action_history = []
        # Perform 500 steps in the environment
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # First time register the gini of salarie
            if(env.reward_type == RewardType.R1):
                consumption_history.append(info['R1'])
                inequality_history.append((info['gini_index']-gini_sal)/(1-gini_sal))
            elif(env.reward_type == RewardType.R2):
                consumption_history.append(info['R2'])
                inequality_history.append((info['gini_index']-gini_sal)/(1-gini_sal))
            else:
                consumption_history.append(info['R2'])
                inequality_history.append(info['cv'])
            action_history.append(action)
            if done:
                env.reset()  # Reset the environment if it reaches a terminal state
        
        # Calculate the mean of the last 100 values for gini and consumptions:
        inequality = np.mean(inequality_history[-100:])
        consumption = np.mean(consumption_history[-100:])
        action = np.mean(action_history[-100:], axis=0)# mean along the columns

        # Assuming the environment returns two objectives you want to optimize
        objective1 = -inequality  # Minimize inequality
        objective2 = consumption  # Maximize consumption

        return [objective1, objective2, action[0], action[1], action[2]]

    objective1, objective2, T_max, T_exp, T_scale = simulate()


    # Define your directory path and filename
    directory_path = 'simulation_PF/' + algorithm + '/' + reward_type
    filename = 'results.csv'
    file_path = os.path.join(directory_path, filename)

    # Check if the directory exists, if not, create it
    os.makedirs(directory_path, exist_ok=True)

    # Define the header and data rows
    header = ['alpha', 'T_max', 'T_exp', 'T_scale', 'objective1', 'objective2']
    data = [alpha, T_max, T_exp, T_scale, objective1, objective2]

    # Check if the CSV file exists and if it's empty to decide on writing the header
    file_exists = os.path.exists(file_path)
    write_header = not file_exists or os.stat(file_path).st_size == 0

    # Open the file in append mode and write data
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)

    print(f"Results appended to {file_path}")
    
# run it like:
# python sim.py 0.5 R1 PPO file_path
