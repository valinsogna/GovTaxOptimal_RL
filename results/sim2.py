import numpy as np
import plotly.express as px
import pandas as pd
import pickle
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
import json

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
                 episode_steps=1000,
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
        self.std_avg_rel_consumptions = None
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
            'R2': self.average_consumption_normalized,
            'std_avg_rel_consumptions':self.std_avg_rel_consumptions,
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
        self.std_avg_rel_consumptions = np.std(self.avg_rel_consumptions)
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)  # Convert float32/64 to Python float
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)  # Convert int32/64 to Python int
        return json.JSONEncoder.default(self, obj)

def append_data_to_json(file_path, new_data):
    try:
        with open(file_path, 'r+') as file:
            try:
                file_data = json.load(file)
            except json.JSONDecodeError:
                file_data = []  # Initialize as an empty list if file is empty or not valid JSON
            file_data.append(new_data)
            file.seek(0)  # Go back to the start of the file
            file.truncate()  # Clear the file content
            # Use NumpyEncoder when dumping data to handle NumPy types
            json.dump(file_data, file, indent=4, cls=NumpyEncoder)
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            # Use NumpyEncoder here as well
            json.dump([new_data], file, indent=4, cls=NumpyEncoder)

# def append_data_to_json(file_path, new_data):
#     # Attempt to open and read the file. If the file does not exist or is empty,
#     # initialize file_data as an empty list.
#     try:
#         with open(file_path, 'r+') as file:
#             try:
#                 file_data = json.load(file)
#             except json.JSONDecodeError:
#                 file_data = []  # Initialize as an empty list if file is empty or not valid JSON
#             file_data.append(new_data)
#             file.seek(0)  # Go back to the start of the file
#             file.truncate()  # Clear the file content
#             json.dump(file_data, file, indent=4)
#     except FileNotFoundError:
#         # If the file does not exist, create it and write the new data in a list
#         with open(file_path, 'w') as file:
#             json.dump([new_data], file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise RuntimeError("You must provide alpha, reward type, algorithm, and file_path as command line arguments")

    alpha = float(sys.argv[1])
    reward_type = sys.argv[2]
    algorithm = sys.argv[3]
    file_path = sys.argv[4]

    if not os.path.exists(file_path):
        raise RuntimeError("The file does not exist")

    if algorithm.lower() == 'ppo':
        model = PPO.load(file_path)
    elif algorithm.lower() == 'sac':
        model = SAC.load(file_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    env = TaxationEnv(reward_type=RewardType[reward_type], use_wandb=False, alpha=alpha)

    obs = env.reset()
    done = False
    cumulative_reward = 0  # Initialize cumulative reward
    avg_rel_consumptions = []

    tracked_variables = {
        'gini_index': [],
        'reward': [],
        'wealth_mean': [],
        'wealth_std': [],
    }
    tax_function_values = []
    tax_rates_per_percentile = {p: [] for p in env.percentiles}
    all_wealth_values = []  # To store wealth values at each step
    consumption_history = []
    cv_history = []
    inequality_history = []
    gini_sal = env.gini_salari
    T_max_history, T_esp_history, T_scale_history = [], [], []
    W_scale_history = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        cumulative_reward += reward  # Update cumulative reward
        
        tracked_variables['gini_index'].append(info['gini_index'])
        tracked_variables['reward'].append(reward)
        tracked_variables['wealth_mean'].append(info['wealth_mean'])
        tracked_variables['wealth_std'].append(np.std(env.wealth))
        all_wealth_values.append(sorted(env.wealth))  # Store sorted wealth values at each step

        T_max_history.append(info['T_max'])
        T_esp_history.append(info['T_esp'])
        T_scale_history.append(info['T_scale'])
        W_scale_history.append(info['W_scale'])

        # New logic for tax function values and tax rates per percentile
        current_tax_rates = [env._taxes(w, action) for w in env.wealth]
        tax_function_values.append(current_tax_rates)
        for p in env.percentiles:
            percentile_wealth = np.percentile(env.wealth, p)
            tax_rate = env._taxes(percentile_wealth, action)
            tax_rates_per_percentile[p].append(tax_rate)
        
        avg_rel_consumptions.append(info['avg_rel_consumptions'])
        if(env.reward_type == RewardType.R1):
            consumption_history.append(info['R1'])
            inequality_history.append((info['gini_index']-gini_sal)/(1-gini_sal))
            cv_history.append(info['cv'])
        elif(env.reward_type == RewardType.R2):
            consumption_history.append(info['R2'])
            inequality_history.append((info['gini_index']-gini_sal)/(1-gini_sal))
            cv_history.append(info['cv'])
        else:
            consumption_history.append(info['R2'])
            inequality_history.append((info['gini_index']-gini_sal)/(1-gini_sal))
            cv_history.append(info['cv'])

    # Final wealth and expected salaries
    final_wealth_sorted = sorted(env.wealth.copy())  # Ensure this is sorted for plotting
    expected_salaries = env.expected_salaries.copy()  # Copy the expected salaries

    # Extract the final tax parameters from their history
    final_T_max = T_max_history[-1]
    final_T_esp = T_esp_history[-1]
    final_T_scale = T_scale_history[-1]

    # Combine these into a list as expected by your env._taxes function
    taxes_params = [final_T_max, final_T_esp, final_T_scale]

    # Calculate the tax rates for the final sorted wealth using these parameters
    tax_rates_final_wealth = [env._taxes(w, taxes_params) for w in final_wealth_sorted]


    # After exiting the loop, calculate metrics for the last 100 steps
    avg_rel_consumptions_last_100 = np.mean(avg_rel_consumptions[-100:])
    std_avg_rel_consumptions_last_100 = np.std(avg_rel_consumptions[-100:])

    gini_index_last_100_avg = np.mean(tracked_variables['gini_index'][-100:])
    gini_index_last_100_std = np.std(tracked_variables['gini_index'][-100:])

    gini_normalized_last_100_avg = np.mean(inequality_history[-100:])
    gini_normalized_last_100_std = np.std(inequality_history[-100:])
    cv_last_100_avg = np.mean(cv_history[-100:])
    cv_last_100_std = np.std(cv_history[-100:])

    reward_consumption_part_last_100_avg = np.mean(consumption_history[-100:])
    reward_consumption_part_last_100_std = np.std(consumption_history[-100:])

    T_max_last_100_avg = np.mean(T_max_history[-100:])
    T_max_last_100_std = np.std(T_max_history[-100:])
    T_esp_last_100_avg = np.mean(T_esp_history[-100:])
    T_esp_last_100_std = np.std(T_esp_history[-100:])
    T_scale_last_100_avg = np.mean(T_scale_history[-100:])
    T_scale_last_100_std = np.std(T_scale_history[-100:])
    W_scale_last_100_avg = np.mean(W_scale_history[-100:])
    W_scale_last_100_std = np.std(W_scale_history[-100:])


    # Final wealth statistics
    final_wealth_avg = np.mean(all_wealth_values[-1])  # Assuming the last entry is the final state
    final_wealth_std = np.std(all_wealth_values[-1])

    # Calculation for average sorted wealth over the last 100 steps
    avg_sorted_wealth_last_100 = np.mean(all_wealth_values[-100:], axis=0)

    # Additional data to include after simulation
    tax_function_avg_std_last_100 = {
        'average': np.mean(tax_function_values[-100:], axis=0).tolist(),
        'std': np.std(tax_function_values[-100:], axis=0).tolist(),
    }
    tax_rates_per_percentile_avg_std_last_100 = {
        p: {
            'average': np.mean(tax_rates_per_percentile[p][-100:]),
            'std': np.std(tax_rates_per_percentile[p][-100:])
        }
        for p in env.percentiles
    }

    simulation_data = {
        'alpha': alpha,
        **tracked_variables,  # Ensuring all tracked variables are included
        'avg_sorted_wealth_last_100': avg_sorted_wealth_last_100.tolist(),  # Converting numpy array to list for JSON serialization
        'tax_function_vs_wealth_last_100_avg_std': tax_function_avg_std_last_100,
        'tax_rate_vs_percentile_last_100_avg_std': tax_rates_per_percentile_avg_std_last_100,
    }

    # After the loop, include the final cumulative reward in your simulation data
    simulation_data['final_cumulative_reward'] = cumulative_reward

    simulation_data.update({
        'avg_rel_consumptions_last_100': avg_rel_consumptions_last_100,
        'std_avg_rel_consumptions_last_100': std_avg_rel_consumptions_last_100,
        'gini_index_last_100_avg': gini_index_last_100_avg,
        'gini_index_last_100_std': gini_index_last_100_std,
        'final_wealth_avg': final_wealth_avg,
        'final_wealth_std': final_wealth_std,
        'gini_normalized_last_100_avg': gini_normalized_last_100_avg,
        'gini_normalized_last_100_std': gini_normalized_last_100_std,
        'reward_consumption_part_last_100_avg': reward_consumption_part_last_100_avg,
        'reward_consumption_part_last_100_std': reward_consumption_part_last_100_std,
        'T_max_last_100_avg': T_max_last_100_avg,
        'T_max_last_100_std': T_max_last_100_std,
        'T_esp_last_100_avg': T_esp_last_100_avg,
        'T_esp_last_100_std': T_esp_last_100_std,
        'T_scale_last_100_avg': T_scale_last_100_avg,
        'T_scale_last_100_std': T_scale_last_100_std,
        'W_scale_last_100_avg': W_scale_last_100_avg,
        'W_scale_last_100_std': W_scale_last_100_std,
        # Add the new variables for plotting
        'final_wealth_sorted': final_wealth_sorted,
        'expected_salaries': expected_salaries,
        'tax_rates_final_wealth': tax_rates_final_wealth,
        'cv_last_100_avg':cv_last_100_avg,
        'cv_last_100_std': cv_last_100_std,
    })

    output_dir = 'simulation_results_json'
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"{algorithm}_{reward_type}.json"
    output_file_path = os.path.join(output_dir, output_file_name)

    append_data_to_json(output_file_path, simulation_data)

    print(f"Simulation data for alpha={alpha} appended to {output_file_path}.")