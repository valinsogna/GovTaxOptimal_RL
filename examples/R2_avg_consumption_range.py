import numpy as np
from tqdm import tqdm

# Parameters
pop_size = 1000
exp_salaries_params = (10, 1)  # mean, std for lognormal distribution of expected salaries
wealth_init_params = (12, 2)  # mean, std for lognormal distribution of wealth
consumptions_params = (0.8, 5)  # eta, K for the consumption function
step_size = 0.01  # Step size for iterating over tax parameters

# Generate distributions
np.random.seed(42)
expected_salaries = np.random.lognormal(exp_salaries_params[0], exp_salaries_params[1], size=pop_size)
wealth = np.random.lognormal(wealth_init_params[0], wealth_init_params[1], size=pop_size)

def _taxes(wealth, taxes_params):
    max_tax, esp, scale = taxes_params
    ts = max_tax * (1 - 1 / (1 + (wealth / np.exp(scale * 25))**esp))
    return ts

def _consumptions(wealth, expected_salaries, consumptions_params, taxes_params):
    η, K = consumptions_params
    cons = 1 / (1 + ((wealth * (1 / (K * expected_salaries) + 10**-5)) * (1 - _taxes(wealth, taxes_params)))**η)
    return cons

# Total iterations for tqdm
total_iterations = int((1/step_size) * (1/step_size) * (1/step_size))

with tqdm(total=total_iterations) as pbar:
    min_value = np.inf
    max_value = -np.inf
    min_params = ()
    max_params = ()
# Iterate over all possible combinations of tax parameters
    for T_max in np.arange(0, 1, step_size):
        for T_esp in np.arange(0, 1.01, step_size):  # Goes slightly above 1 to include 1 in the range
            for T_scale in np.arange(0, 1.01, step_size):
                taxes_params = (T_max, T_esp, T_scale)
                consumptions = _consumptions(wealth, expected_salaries, consumptions_params, taxes_params)
                average_consumption_normalized = np.mean(consumptions * (wealth / (consumptions_params[1] * expected_salaries)))

                if average_consumption_normalized < min_value:
                    min_value = average_consumption_normalized
                    min_params = taxes_params

                if average_consumption_normalized > max_value:
                    max_value = average_consumption_normalized
                    max_params = taxes_params

                pbar.update(1)


print(f"Minimum Average Consumption Normalized: {min_value} for T_max={min_params[0]}, T_esp={min_params[1]}, T_scale={min_params[2]}")
print(f"Maximum Average Consumption Normalized: {max_value} for T_max={max_params[0]}, T_esp={max_params[1]}, T_scale={max_params[2]}")

# For step_size=0.01, the output is:
# 1020100it [00:34, 29855.58it/s]                                                                                                                  
# Minimum Average Consumption Normalized: 0.5708442856601331 for T_max=0.0, T_esp=0.0, T_scale=0.0
# Maximum Average Consumption Normalized: 6.674197164418275 for T_max=0.99, T_esp=1.0, T_scale=0.0