import os
from src.model import ModelManager
from stable_baselines3.common.vec_env import VecNormalize
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pickle
import pandas as pd

SUPPORTED_ALGORITHMS = ["PPO", "SAC", "TD3"]

def mov_average(data, window_size=1000):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().to_numpy()

# Function to save data as a pickle file
def save_as_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Define the reward type enum:
from enum import Enum
class RewardType(Enum):
    R1 = "R1"
    R2 = "R2"

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

def parse_number_list(input_str):
    """
    Parses a string of numbers separated by commas and returns a tuple of numbers.
    Supports basic mathematical expressions (e.g., "10**7").
    """
    number_list = []
    for item in input_str.split(","):
        try:
            # Evaluate the expression (e.g., "10**7") and convert to float
            number_list.append(float(eval(item.strip())))
        except Exception as e:
            raise ValueError(f"Invalid input for number list: {item}") from e
    return tuple(number_list)


def save_config(config, save_path):
    with open(save_path, 'w') as file:
        json.dump(config, file, indent=4)

def load_model_for_resume(run_id):
    """
    Loads a model based on the run ID from the saved models directory.
    
    Args:
        run_id (str): The run ID of the model to be loaded.

    Returns:
        model: The loaded model.
    """
    model_file_name = "model.zip"
    env_file_name = "vecnorm.pkl"

    for algorithm in SUPPORTED_ALGORITHMS:
        model_load_path = f"results/saved_models/{algorithm}/{run_id}/{model_file_name}"
        env_load_path = f"results/saved_models/{algorithm}/{run_id}/{env_file_name}"
        if os.path.exists(model_load_path) and os.path.exists(env_load_path):
            #Load the environment
            env = VecNormalize.load(env_load_path)
            #pass the environment to the model manager
            model_manager = ModelManager(algorithm, env=env)
            return model_manager.load_model(model_load_path), env

    raise FileNotFoundError(f"No model found for run ID {run_id} in any algorithm folder {model_load_path}.")

def load_model_for_evaluation(run_id, algorithm):
    """
    Loads a model based on the run ID from the saved models directory.
    
    Args:
        run_id (str): The run ID of the model to be loaded.

    Returns:
        model: The loaded model.
    """
    model_file_name = "model.zip"


    model_load_path = f"results/saved_models/{algorithm}/{run_id}/{model_file_name}"
    if os.path.exists(model_load_path):
        model_manager = ModelManager(algorithm)
        return model_manager.load_model(model_load_path)
    else:
        raise FileNotFoundError(f"No model found for run ID {run_id} in any algorithm folder {model_load_path}")

# Define the function to adjust x-axis for all powers of 10
def set_log_scale_xaxis(wealth_data):
    min_power = np.floor(np.log10(np.min(wealth_data)))
    max_power = np.ceil(np.log10(np.max(wealth_data)))
    plt.xlim([10**min_power, 10**max_power])

def calculate_mean_across_envs(all_stats, key):
    # Calcola la media di ciascun elemento corrispondente nelle liste attraverso gli ambienti
    return np.mean(np.array(all_stats[key]), axis=0)

# Function to plot Consumptions vs Wealth
def plot_consumptions_vs_wealth(wealth, plot_path, T_max, T_esp, T_scale, env, expected_salaries):
    sorted_indices = np.argsort(wealth)
    sorted_wealth = np.array(wealth)[sorted_indices]
    sorted_consumptions = env._consumptions(sorted_wealth, expected_salaries, [T_max, T_esp, T_scale])

    plt.figure(figsize=(10, 7))
    plt.plot(sorted_wealth, sorted_consumptions, label='Consumptions', linewidth=2)
    plt.xscale('log')
    set_log_scale_xaxis(sorted_wealth)
    plt.xlabel("Wealth (log scale)")
    plt.ylabel("Consumptions")
    plt.title("Final Consumptions vs Final Wealth")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path + "Final Consumptions_vs_Wealth.png")
    plt.close()

# Function to plot Absolute Consumptions vs Wealth
def plot_absolute_consumptions_vs_wealth(wealth, plot_path, T_max, T_esp, T_scale, env, expected_salaries):
    sorted_indices = np.argsort(wealth)
    sorted_wealth = np.array(wealth)[sorted_indices]
    sorted_absolute_consumptions = env._consumptions(sorted_wealth, expected_salaries, [T_max, T_esp, T_scale])*sorted_wealth

    plt.figure(figsize=(10, 7))
    plt.plot(sorted_wealth, sorted_absolute_consumptions, label='Absolute Consumptions', linewidth=2)
    plt.xscale('log')
    set_log_scale_xaxis(sorted_wealth)
    plt.xlabel("Wealth (log scale)")
    plt.ylabel("Absolute Consumptions")
    plt.title("Final Absolute Consumptions vs Final Wealth")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path + "Final_Absolute_Consumptions_vs_Wealth.png")
    plt.close()

# Function to plot Tax Function vs Wealth
def plot_tax_function_vs_wealth(final_wealth, T_max, T_esp, T_scale, plot_path, env):
    # Sort the final wealth to get an ordered wealth distribution
    sorted_indices = np.argsort(final_wealth)
    sorted_wealth = np.array(final_wealth)[sorted_indices]
    
    # Calculate the tax rates for the sorted wealth using the final tax parameters
    final_tax_rates = [env._taxes(w, [T_max, T_esp, T_scale]) for w in sorted_wealth]
    
    plt.figure(figsize=(10, 7))
    plt.plot(sorted_wealth, final_tax_rates, label='Tax Rates', linewidth=2)
    plt.xscale('log')
    set_log_scale_xaxis(sorted_wealth)
    plt.xlabel("Wealth (log scale)")
    plt.ylabel("Tax Rates")
    plt.title("Final Tax Function vs Final Wealth")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plot_path, "Final_Tax_Function_vs_Wealth.png"))
    plt.close()

# Function to plot Tax Function with Wealth and Expected Salaries Distributions
def plot_tax_function_with_distributions(wealth, expected_salaries, plot_path, T_max, T_esp, T_scale, env):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Tax Function
    sorted_indices = np.argsort(wealth)
    sorted_wealth = np.array(wealth)[sorted_indices]
    # Calculate the tax rates for the sorted wealth using the final tax parameters
    sorted_tax_rates = [env._taxes(w, [T_max, T_esp, T_scale]) for w in sorted_wealth]
    ax1.plot(sorted_wealth, sorted_tax_rates, label='Tax Rates', color='blue', linewidth=2)
    ax1.set_xlabel("Wealth (log scale)")
    ax1.set_ylabel("Tax Rates", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xscale('log')
    set_log_scale_xaxis(sorted_wealth)

    # Instantiate a second y-axis to plot wealth and salary distributions
    ax2 = ax1.twinx()
    # Plotting Distributions
    hist, bins = np.histogram(wealth, bins=100)
    hist2, bins2 = np.histogram(expected_salaries, bins=100)
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #logbins2 = np.logspace(np.log10(bins2[0]),np.log10(bins2[-1]),len(bins2))

    density_wealth, bins_wealth, _ = ax2.hist(wealth, bins=logbins, alpha=0.5, label='Final Wealth Distribution', color='green', density=True)
    # Plotting Expected Salaries Distribution
    density_salaries, bins_salaries, _ = ax2.hist(expected_salaries, bins=logbins, alpha=0.5, label='Expected Salaries Distribution', color='red', density=True)
    ax2.set_ylabel('Pdfs', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title("Final Tax Function vs Final Wealth and Expected Salary Distributions")
    plt.savefig(plot_path + "Final_Tax_Function_vs_Wealth_Salary_Distributions.png")
    plt.close()


def plot_tax_function_wrt_percentiles(percentiles, plot_path, final_tax_rates):
    plt.figure(figsize=(10, 7))
    plt.plot(percentiles, final_tax_rates, marker='o', linestyle='-', color='purple', label='Tax Rates', linewidth=2)
    plt.xlabel('Percentiles')
    plt.ylabel('Tax Rates')
    plt.title('Final Taxation Function with respect to Percentiles')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plot_path, "Final_Tax_Function_vs_Percentiles.png"))
    plt.close()
