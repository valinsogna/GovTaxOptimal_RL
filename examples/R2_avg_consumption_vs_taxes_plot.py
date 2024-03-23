import numpy as np
import matplotlib.pyplot as plt

# Your provided variables and functions, adjusted for correct parameter passing
pop_size = 1000
exp_salaries_params = (10, 1)  # mean, std
wealth_init_params = (12, 2)  # mean, std
consumptions_params = (0.8, 5)  # eta, K

np.random.seed(42)
expected_salaries = np.random.lognormal(exp_salaries_params[0], exp_salaries_params[1], size=pop_size)
wealth = np.random.lognormal(wealth_init_params[0], wealth_init_params[1], size=pop_size)

def _taxes(wealth, max_tax, esp, scale):
    # Ensure inputs to np.exp are floats to avoid TypeError
    ts = max_tax * (1 - 1 / (1 + (wealth / np.exp(float(scale) * 25))**esp))
    return ts

def _consumptions(wealth, expected_salaries, consumptions_params, max_tax, esp, scale):
    eta, K = consumptions_params
    # Ensure wealth and expected_salaries are compatible for division
    wealth = np.array(wealth, dtype=np.float64)
    expected_salaries = np.array(expected_salaries, dtype=np.float64)
    cons = 1 / (1 + ((wealth * (1 / (K * expected_salaries + 10**-5))) * (1 - _taxes(wealth, max_tax, esp, scale)))**eta)
    return cons

# Fixed parameters for demonstration
fixed_params = {'T_max': 0.5, 'T_esp': 0.5, 'T_scale': 0.5}

# Plotting function adjusted for correct dictionary unpacking
def plot_avg_consumption_vs_tax_param(param_range, param_name, fixed_params):
    avg_consumptions = []
    for param in param_range:
        # Unpack fixed_params and update the current parameter
        max_tax, esp, scale = fixed_params['T_max'], fixed_params['T_esp'], fixed_params['T_scale']
        if param_name == 'T_max':
            max_tax = param
        elif param_name == 'T_esp':
            esp = param
        elif param_name == 'T_scale':
            scale = param
        consumptions = _consumptions(wealth, expected_salaries, consumptions_params, max_tax, esp, scale)
        avg_consumption_normalized = np.mean(consumptions * (wealth / (consumptions_params[1] * expected_salaries)))
        avg_consumptions.append(avg_consumption_normalized)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, avg_consumptions, marker='o')
    plt.title(f'Average Consumption Normalized vs. {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Average Consumption Normalized')
    plt.grid(True)
    plt.savefig(f'avg_consumption_vs_{param_name}.png')
    plt.show()

# Parameter ranges for plotting
T_max_range = np.linspace(0, 0.99, 20)
T_esp_range = np.linspace(0, 1, 20)
T_scale_range = np.linspace(0, 1, 20)

# Generate plots
plot_avg_consumption_vs_tax_param(T_max_range, 'T_max', fixed_params)
plot_avg_consumption_vs_tax_param(T_esp_range, 'T_esp', fixed_params)
plot_avg_consumption_vs_tax_param(T_scale_range, 'T_scale', fixed_params)

import numpy as np
import matplotlib.pyplot as plt

# Initialization parameters and functions (unchanged from previous)
pop_size = 1000
exp_salaries_params = (10, 1)  # mean, std
wealth_init_params = (12, 2)  # mean, std
consumptions_params = (0.8, 5)  # eta, K

np.random.seed(42)
expected_salaries = np.random.lognormal(exp_salaries_params[0], exp_salaries_params[1], size=pop_size)
wealth = np.random.lognormal(wealth_init_params[0], wealth_init_params[1], size=pop_size)

def _taxes(wealth, max_tax, esp, scale):
    ts = max_tax * (1 - 1 / (1 + (wealth / np.exp(float(scale) * 25))**esp))
    return ts

def _consumptions(wealth, expected_salaries, consumptions_params, max_tax, esp, scale):
    eta, K = consumptions_params
    cons = 1 / (1 + ((wealth * (1 / (K * expected_salaries + 10**-5))) * (1 - _taxes(wealth, max_tax, esp, scale)))**eta)
    return cons

# Plot average_consumption_normalized vs T_max (overall tax rate)
def plot_avg_consumption_vs_T_max():
    T_max_range = np.linspace(0, 0.99, 100)  # More granular range for T_max
    avg_consumptions = []

    # Fix T_esp and T_scale at representative values
    T_esp_fixed = 0.5
    T_scale_fixed = 0.5

    for T_max in T_max_range:
        consumptions = _consumptions(wealth, expected_salaries, consumptions_params, T_max, T_esp_fixed, T_scale_fixed)
        avg_consumption_normalized = np.mean(consumptions * (wealth / (consumptions_params[1] * expected_salaries)))
        avg_consumptions.append(avg_consumption_normalized)
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_max_range, avg_consumptions, marker='o', linestyle='-', markersize=4)
    plt.title('Average Consumption Normalized vs. Tax Rate (T_max)')
    plt.xlabel('Tax Rate (T_max)')
    plt.ylabel('Average Consumption Normalized')
    plt.grid(True)
    plt.show()
    plt.savefig('avg_consumption_vs_T_max.png')

# Generate the plot
plot_avg_consumption_vs_T_max()

print(f"average tax_function with Tmax=0.99, T_esp=1, T_scale=0 : {np.mean(_taxes(wealth, 0.99, 1, 0))}")
print(f"median tax_function with Tmax=0.99, T_esp=0, T_scale=1: {np.median(_taxes(wealth, 0.99, 1, 0))}")
print(f"average tax_function with Tmax=0.99, T_esp=0.5, T_scale=0.5: {np.mean(_taxes(wealth, 0.99, 0.5, 0.5))}")
print(f"median tax_function with Tmax=0.99, T_esp=1, T_scale=1: {np.median(_taxes(wealth, 0.99, 0.5, 0.5))}")


# average tax_function with Tmax=0.99, T_esp=1, T_scale=0 : 0.989959909743579
# median tax_function with Tmax=0.99, T_esp=0, T_scale=1: 0.9899946381747642
# average tax_function with Tmax=0.99, T_esp=0.5, T_scale=0.5: 0.4587172772106696
# median tax_function with Tmax=0.99, T_esp=1, T_scale=1: 0.4488708478099318