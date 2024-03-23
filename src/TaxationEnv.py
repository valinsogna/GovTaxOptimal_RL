import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import wandb
from gym.utils.seeding import np_random
import os
import sys
from src.utils import RewardType, set_log_scale_xaxis, ecdf

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
                 use_wandb=True,
                 alpha=1,  # param for convex combination btw Gini and consumption
                 reward_type=RewardType.R1,
                 exp_salaries_params=(10, 1), # mean, std
                 wealth_init_params=(12, 2), # mean, std
                 action_space_lower_bound = np.array([0, 0, 0.01]),# T_max, T_esp, T_scale
                 action_space_upper_bound = np.array([0.10, 3, 1]),# T_max, T_esp, T_scale
                #  action_space_lower_bound = np.array([0, 0, 0.01]),
                #  action_space_upper_bound = np.array([1.0, 1.0, 1.0]),
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
        self.new_taxes_params = action #* np.array([10, 1.0/3.0, 1.0])

    def _calculate_reward(self):

        gini_index = self._calculate_gini(self.wealth)

        # reward_type is an enum of "R1" or "R2"
        if(self.reward_type == RewardType.R2):

            average_consumption_normalized = np.mean(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params) * (self.wealth / (self.consumptions_params[1] * self.expected_salaries)))

            #average_consumption_not_normalized = np.mean(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)*self.wealth)

            #reward = -self.alpha * gini_index + (1 - self.alpha) * average_consumption_normalized

            #new version:
            reward = -self.alpha * (gini_index - self.gini_salari)/(1-self.gini_salari) + (1 - self.alpha) * average_consumption_normalized

            # use the cofficient of variation as a measure of inequality instead of gini index:
            self.cv = np.std(self.wealth)/np.mean(self.wealth)
            # reward = -self.alpha * self.cv + (1 - self.alpha) * average_consumption_normalized


        elif(self.reward_type == RewardType.R1): 

            #taxes = self._taxes(self.wealth, self.new_taxes_params)

            consumptions = self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)

            # average_savings_after_taxes = np.mean((1 - consumptions) * (1 - taxes))

            average_savings_after_taxes = np.mean((1 - consumptions))

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

    def show_statistics(self, mode='single', file_path=None):
        # Check if the directory exist otherwise make it:
        if file_path != None and not os.path.exists(file_path):
            os.makedirs(file_path)

        if mode == 'single':
            # Basic information
            print(f"Gini index: {self.gini_index:.4f}")
            print(f"Wealth mean: {np.mean(self.wealth):.2f}")
            print(f"Wealth median: {np.median(self.wealth):.2f}")
            print(f"T_max: {self.new_taxes_params[0]:.8f}")
            print(f"T_esp: {self.new_taxes_params[1]:.8f}")
            print(f"T_scale: {self.new_taxes_params[2]:.8f}")

            #Save on file_path also the outputs of the previous prints:
            with open(file_path + 'statistics.txt', 'w') as f:
                print("Statistics for the Taxation Environment", file=f)
                print(f"Alpha: {self.alpha}", file=f)
                print(f"Reward: {self.reward_type}", file=f)
                print(f"Gini index: {self.gini_index:.4f}", file=f)
                print(f"Wealth mean: {np.mean(self.wealth):.2f}", file=f)
                print(f"Wealth median: {np.median(self.wealth):.2f}", file=f)
                print(f"T_max: {self.new_taxes_params[0]:.8f}", file=f)
                print(f"T_esp: {self.new_taxes_params[1]:.8f}", file=f)
                print(f"T_scale: {self.new_taxes_params[2]:.8f}", file=f)
                #Stampa i percentili a che ricchezza corrispondono:
                print(f"Percentiles: {self.percentiles}", file=f)
                print(f"Wealth at percentiles: {[np.percentile(self.wealth, x) for x in self.percentiles]}", file=f)
                print('\n', file=f)
                # Print mean and median, std min max of consumptions
                print(f"Mean of consumptions: {np.mean(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)):.4f}", file=f)
                print(f"Median of consumptions: {np.median(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)):.4f}", file=f)
                print(f"Standard deviation of consumptions: {np.std(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)):.4f}", file=f)
                print(f"Max of consumptions: {np.max(self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params)):.4f}", file=f)
                print('\n', file=f)
                # Print mean, median std, max e min of wealth
                print(f"Mean of wealth: {np.mean(self.wealth):.4f}", file=f)
                print(f"Median of wealth: {np.median(self.wealth):.4f}", file=f)
                print(f"Standard deviation of wealth: {np.std(self.wealth):.4f}", file=f)
                print(f"Max of wealth: {np.max(self.wealth):.4f}", file=f)
                print(f"Min of wealth: {np.min(self.wealth):.4f}", file=f)
                print('\n', file=f)
                # Print mean, median std, max e min of salaries
                print(f"Mean of exp salaries: {np.mean(self.expected_salaries):.4f}", file=f)
                print(f"Median of exp salaries: {np.median(self.expected_salaries):.4f}", file=f)
                print(f"Standard deviation of exp salaries: {np.std(self.expected_salaries):.4f}", file=f)
                print(f"Max of exp salaries: {np.max(self.expected_salaries):.4f}", file=f)
                print(f"Min of exp salaries: {np.min(self.expected_salaries):.4f}", file=f)
                print('\n', file=f)
                # Print mean, median std, max e min of taxes
                print(f"Mean of taxes: {np.mean([self._taxes(w, self.new_taxes_params) for w in self.wealth]):.4f}", file=f)
                print(f"Median of taxes: {np.median([self._taxes(w, self.new_taxes_params) for w in self.wealth]):.4f}", file=f)
                print(f"Standard deviation of taxes: {np.std([self._taxes(w, self.new_taxes_params) for w in self.wealth]):.4f}", file=f)
                print(f"Max of taxes: {np.max([self._taxes(w, self.new_taxes_params) for w in self.wealth]):.4f}", file=f)
                print(f"Min of taxes: {np.min([self._taxes(w, self.new_taxes_params) for w in self.wealth]):.4f}", file=f)
            

            # Wealth Distribution Plot
            plt.figure(figsize=(10, 7))
            plt.hist(self.initial_wealth, bins=1000, alpha=0.5, label='Initial', log=True)
            plt.hist(self.wealth, bins=1000, alpha=0.5, label='Final', log=True)
            plt.legend(loc='upper right')
            plt.title("Initial vs Final Wealth Distribution")
            plt.xlabel("Wealth")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(file_path + "Wealth_Distribution.png")
            plt.show()

            # Plot for Consumptions vs Wealth
            plt.figure(figsize=(10, 7))
            # Sort the wealth and corresponding consumptions for line plot
            indices = np.argsort(self.wealth)
            sorted_wealth = self.wealth[indices]
            sorted_consumptions = self._consumptions(sorted_wealth, self.expected_salaries, self.new_taxes_params)
            plt.plot(sorted_wealth, sorted_consumptions, label='Consumptions', linewidth=2)
            plt.xscale('log')
            # Assuming set_log_scale_xaxis is a function that adjusts the x-axis
            set_log_scale_xaxis(sorted_wealth)
            plt.xlabel("Wealth (log scale)")
            plt.ylabel("Consumptions")
            plt.title("Consumptions vs Wealth")
            plt.grid(True)
            plt.legend()
            plt.savefig(file_path + "Consumptions_vs_Wealth.png")
            plt.show()

            # Plot for Absolute Consumptions vs Wealth
            plt.figure(figsize=(10, 7))
            # Use sorted wealth and consumptions from above
            absolute_consumptions = sorted_consumptions * sorted_wealth
            plt.plot(sorted_wealth, absolute_consumptions, label='Absolute Consumptions', linewidth=2)
            plt.xscale('log')
            set_log_scale_xaxis(sorted_wealth)
            plt.xlabel("Wealth (log scale)")
            plt.ylabel("Absolute Consumptions")
            plt.title("Absolute Consumptions vs Wealth")
            plt.grid(True)
            plt.legend()
            plt.savefig(file_path + "Absolute_Consumptions_vs_Wealth.png")
            plt.show()


            # Plot for Tax Function vs Wealth
            plt.figure(figsize=(10, 7))
            # Sort the wealth and corresponding tax rates for line plot
            sorted_tax_rates = [self._taxes(w, self.new_taxes_params) for w in sorted_wealth]
            plt.plot(sorted_wealth, sorted_tax_rates, label='Tax Rates', linewidth=2)
            plt.xscale('log')
            set_log_scale_xaxis(sorted_wealth)
            plt.xlabel("Wealth (log scale)")
            plt.ylabel("Tax Rates")
            plt.title("Tax Function vs Wealth")
            plt.grid(True)
            plt.legend()
            plt.savefig(file_path + "Tax_Function_vs_Wealth.png")
            plt.show()

            # Plotting tax function with wealth and expected salaries distribution
            fig, ax1 = plt.subplots(figsize=(12, 7))

            # Plot Tax Function
            sorted_indices = np.argsort(self.wealth)
            sorted_wealth = self.wealth[sorted_indices]
            sorted_tax_rates = [self._taxes(w, self.new_taxes_params) for w in sorted_wealth]
            ax1.plot(sorted_wealth, sorted_tax_rates, label='Tax Rates', color='blue', linewidth=2)
            ax1.set_xlabel("Wealth (log scale)")
            ax1.set_ylabel("Tax Rates", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xscale('log')

            # Instantiate a second y-axis to plot wealth and salary distributions
            ax2 = ax1.twinx()
            # Plotting Wealth Distribution
            density_wealth, bins_wealth, _ = ax2.hist(self.wealth, bins=1000, alpha=0.5, label='Wealth Distribution', color='green', density=True)
            # Plotting Expected Salaries Distribution
            density_salaries, bins_salaries, _ = ax2.hist(self.expected_salaries, bins=1000, alpha=0.5, label='Expected Salaries Distribution', color='red', density=True)
            ax2.set_ylabel('Distributions', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            fig.tight_layout()
            fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
            plt.title("Tax Function vs Wealth with Distributions")
            plt.savefig(file_path + "Tax_Function_vs_Wealth_with_Distributions.png")
            plt.show()

            # Plot for Taxation Function wrt Percentiles
            plt.figure(figsize=(10, 7))
            tax_rates_percentiles = [self._taxes(np.percentile(self.wealth, p), self.new_taxes_params) for p in self.percentiles]
            plt.plot(self.percentiles, tax_rates_percentiles, marker='o', linestyle='-', color='purple')
            plt.xlabel('Percentiles')
            plt.ylabel('Tax Rates')
            plt.title('Taxation Function with respect to Percentiles')
            plt.grid(True)
            plt.savefig(file_path + "Tax_Function_vs_Percentiles.png")
            plt.show()


        else:
        # Calculate the statistics you're interested in
            stats = {
                'gini_index': self.gini_index,
                'wealth': [ w for w in self.wealth],
                'initial_wealth': [w for w in self.initial_wealth],
                'mean_wealth': np.mean(self.wealth),
                'median_wealth': np.median(self.wealth),
                'perc_consumptions': self.perc_consumptions,
                'absolute_consumptions': self._consumptions(self.wealth, self.expected_salaries, self.new_taxes_params) * self.wealth,
                'tax_rates_perc': self.tax_rates_perc,
                'T_max': self.new_taxes_params[0],
                'T_esp': self.new_taxes_params[1],
                'T_scale': self.new_taxes_params[2],
                'average_tax_rate': self.average_tax_rate,
                'W_scale': self.W_scale,
            }
            return stats

    def calculate_gini(self):
        return self._calculate_gini(self.wealth)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]