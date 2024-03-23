import numpy as np

# Parameters
pop_size = 1000
exp_salaries_params = (10, 1)  # mean, std for lognormal distribution of expected salaries
wealth_init_params = (12, 2)  # mean, std for lognormal distribution of wealth
consumptions_params = (0.8, 5)  # eta, K for the consumption function

# Generate distributions
np.random.seed(42)  # For reproducibility
expected_salaries = np.random.lognormal(exp_salaries_params[0], exp_salaries_params[1], size=pop_size)
wealth = np.random.lognormal(wealth_init_params[0], wealth_init_params[1], size=pop_size)

# Define the consumption function and tax from TaxationEnv
def _taxes(wealth, taxes_params):
    max, esp, scale = taxes_params
    ts = max * (1 - 1 / (1 + (wealth / np.exp(scale*25))**esp))
    return ts

def _consumptions(wealth, expected_salaries, consumptions_params, taxes_params):
    η, K = consumptions_params
    cons = 1 / (1 + ((wealth * (1/(K * expected_salaries) + 10**-5)) * (1 - _taxes(wealth, taxes_params)))**η)
    return cons

# Mock taxes params for demonstration
taxes_params = (0.99, 1, 0) 

# Calculate consumptions using the defined function
consumptions = _consumptions(wealth, expected_salaries, consumptions_params, taxes_params)

# Calculate average consumption normalized
average_consumption_normalized = np.mean(consumptions * (wealth / (consumptions_params[1] * expected_salaries)))

print(f"Average Consumption Normalized: {average_consumption_normalized}")

# 0.75 for T_max=0.5, T_esp=0.5, T_scale=0.5
# 0.6 for T_max=0, T_esp=0.5, T_scale=0.5
# 1.32 for T_max=1, T_esp=0.5, T_scale=0.5
# 0.57 for T_max=0, T_esp=0, T_scale=0
# 0.57 for T_max=0.99, T_esp=1, T_scale=1
# 6.67 for T_max=0.99, T_esp=1, T_scale=0

