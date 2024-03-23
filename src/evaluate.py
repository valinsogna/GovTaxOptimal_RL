import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from tqdm import tqdm
from src.utils import plot_absolute_consumptions_vs_wealth, plot_consumptions_vs_wealth, plot_tax_function_vs_wealth, plot_tax_function_with_distributions, set_log_scale_xaxis, save_as_pickle, mov_average, plot_tax_function_wrt_percentiles


# Function to plot and save a histogram
def plot_histogram(data, title, xlabel, ylabel, bins, save_path, x_range=None, y_range=None, density=False):
    hist, bins = np.histogram(data, bins=bins)
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.figure(figsize=(10, 7))
    plt.hist(data, bins=logbins, alpha=0.5, density=density, color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Function to plot and save a time series graph with moving average
def plot_time_series(data, title, xlabel, ylabel, save_path, window_size=1000, x_range=None, y_range=None):
    plt.figure(figsize=(10, 7))
    if window_size > 1:
        data_ma = mov_average(data, window_size)        
        plt.plot(data_ma, label=f"moving average (window={window_size})", color='r')
    # Also plot the raw data
    plt.plot(data, label="Raw Data", color='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def evaluate_and_save_plots(model, env, n_eval_episodes, plot_path, algorithm):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    with tqdm(total=n_eval_episodes) as pbar:
        # Assuming n_eval_episodes are episodes
        for episode in range(n_eval_episodes):
            obs = env.reset()
            done = False
            initial_wealth = env.wealth.copy()  # Save initial wealth at the beginning of the episode

            # Initialize lists or variables to collect episode data
            gini_index_history, reward_history, average_tax_rate_history = [], [], []
            total_wealth_history, avg_rel_consumptions_history = [], []
            wealth_mean_history, wealth_median_history = [], []
            T_max_history, T_esp_history, T_scale_history = [], [], []
            tax_rates_perc_history, consumptions_history = [], []
            W_scale_history = []
            # Initialize a list to collect the wealth percentiles at each step as a list of lists: [[percentile1_step1, percentile2_step1, ...], [percentile1_step2, percentile2_step2, ...], ...]
            wealth_percentiles_history = []

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                # Collect required statistics from the environment's returned info
                gini_index_history.append(info['gini_index'])
                reward_history.append(reward)
                average_tax_rate_history.append(info['average_tax_rate'])
                total_wealth_history.append(info['total_wealth'])
                avg_rel_consumptions_history.append(info['avg_rel_consumptions'])
                wealth_mean_history.append(info['wealth_mean'])
                wealth_median_history.append(info['wealth_median'])
                T_max_history.append(info['T_max'])
                T_esp_history.append(info['T_esp'])
                T_scale_history.append(info['T_scale'])
                W_scale_history.append(info['W_scale'])
                tax_rates_perc_history.append(info['tax_rates_perc'])  # Assuming this collects tax rates per percentile
                consumptions_history.append(info['perc_consumptions'])  # Assuming this is the consumption for each households
                # Now collect the wealth percentiles at each step through obs
                wealth_percentiles_history.append([obs[p_idx] for p_idx in range(len(env.percentiles))])

            # After the episode ends
            final_wealth = env.wealth.copy()  # Save final wealth at the end of the episode
            expected_salaries = env.expected_salaries.copy()  # Assuming you want to save this at the end of the episode
            pbar.update(1)
    
    # After collecting all your statistics, save them as a .pkl file
    stats_pickle_path = os.path.join(plot_path, 'simulation.pkl')
    save_as_pickle({
        'initial_wealth': initial_wealth,
        'final_wealth': final_wealth,
        'gini_index_history': gini_index_history,
        'reward_history': reward_history,
        'average_tax_rate_history': average_tax_rate_history,
        'total_wealth_history': total_wealth_history,
        'avg_rel_consumptions_history': avg_rel_consumptions_history,
        'wealth_mean_history': wealth_mean_history,
        'wealth_median_history': wealth_median_history,
        'T_max_history': T_max_history,
        'T_esp_history': T_esp_history,
        'T_scale_history': T_scale_history,
        'expected_salaries': expected_salaries,
        'tax_rates_perc_history': tax_rates_perc_history,
        'consumptions_history': consumptions_history,
        'wealth_percentiles_history': wealth_percentiles_history,
        'W_scale_history': W_scale_history
    }, stats_pickle_path)

    # Plot and save the histograms for initial and final wealth distribution
    plot_histogram(initial_wealth, "Initial Wealth Distribution", "Wealth", "Frequency", bins=100, save_path=os.path.join(plot_path, "initial_wealth_distribution.png"))
    plot_histogram(final_wealth, "Final Wealth Distribution", "Wealth", "Frequency", bins=100, save_path=os.path.join(plot_path, "final_wealth_distribution.png"))
    plot_histogram(expected_salaries, "Expected Salaries Distribution", "Salaries", "Frequency", bins=100, save_path=os.path.join(plot_path, "expected_salaries_distribution.png"))
    plot_histogram(initial_wealth, "Initial Wealth Distribution pdf", "Wealth", "Probability density", bins=100, save_path=os.path.join(plot_path, "initial_wealth_pdf.png"), density=True)
    plot_histogram(final_wealth, "Final Wealth Distribution pdf", "Wealth", "Probability density", bins=100, save_path=os.path.join(plot_path, "final_wealth_pdf.png"), density=True)
    plot_histogram(expected_salaries, "Expected Salaries Distribution pdf", "Salaries", "Probability density", bins=100, save_path=os.path.join(plot_path, "expected_salaries_pdf.png"), density=True)

    # Plot both initial and final wealth distribution pdf on the same graph
    plt.figure(figsize=(10, 7))
    hist, bins = np.histogram(initial_wealth, bins=100)
    hist2, bins2 = np.histogram(final_wealth, bins=100)
    hist3, bins3 = np.histogram(expected_salaries, bins=100)
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    logbins2 = np.logspace(np.log10(bins2[0]),np.log10(bins2[-1]),len(bins2))
    logbins3 = np.logspace(np.log10(bins3[0]),np.log10(bins3[-1]),len(bins3))
    plt.hist(initial_wealth, bins=logbins, alpha=0.5, density=True, color='b', label='Initial Wealth')
    plt.hist(final_wealth, bins=logbins2, alpha=0.5, density=True, color='r', label='Final Wealth')
    plt.hist(expected_salaries, bins=logbins3, alpha=0.5, density=True, color='g', label='Expected Salaries')
    plt.title("Initial, Final Wealth and Expected Salaries pdfs")
    plt.xlabel("Value")
    plt.ylabel("Probability density")
    #plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_path, "initial_final_wealth_salaries_pdfs.png"))
    plt.close()


    # Plot and save the time series graphs for other statistics
    plot_time_series(gini_index_history, "Gini Index Over Steps", "Steps", "Gini Index", os.path.join(plot_path, "gini_index_over_time.png"), window_size=env.pop_size)
    plot_time_series(reward_history, "Reward Over Steps", "Steps", "Reward", os.path.join(plot_path, "reward_over_time.png"), window_size=env.pop_size)
    plot_time_series(average_tax_rate_history, "Average Tax Rate Over Steps", "Steps", "Average Tax Rate", os.path.join(plot_path, "average_tax_rate_over_time.png"), window_size=env.pop_size)
    plot_time_series(total_wealth_history, "Total Wealth Over Steps", "Steps", "Total Wealth", os.path.join(plot_path, "total_wealth_over_time.png"), window_size=env.pop_size)
    plot_time_series(avg_rel_consumptions_history, "Average Relative Consumptions Over Steps", "Steps", "Average Relative Consumptions", os.path.join(plot_path, "avg_rel_consumptions_over_time.png"), window_size=env.pop_size)
    plot_time_series(wealth_mean_history, "Mean Wealth Over Steps", "Steps", "Mean Wealth", os.path.join(plot_path, "mean_wealth_over_time.png"), window_size=env.pop_size)
    plot_time_series(wealth_median_history, "Median Wealth Over Steps", "Steps", "Median Wealth", os.path.join(plot_path, "median_wealth_over_time.png"), window_size=env.pop_size)

    # For Tmax, T_esp, and T_scale, on the same graph for comparison
    plt.figure(figsize=(10, 7))
    plt.plot(mov_average(T_max_history, window_size=env.pop_size), label='T_max moving average', color='b')
    plt.plot(T_max_history, label='T_max', color='c', alpha=0.5)
    plt.plot(mov_average(T_esp_history, window_size=env.pop_size), label='T_esp moving average', color='g')
    plt.plot(T_esp_history, label='T_esp', color='y', alpha=0.5)
    plt.plot(mov_average(T_scale_history, window_size=env.pop_size), label='T_scale moving average', color='r')
    plt.plot(T_scale_history, label='T_scale', color='m', alpha=0.5)
    plt.title("Tax Parameters Over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_path, "tax_parameters_over_time.png"))
    plt.close()

    # Plot T_max, T_esp, T_scale on separate graphs 
    plot_time_series(T_max_history, "T_max Over Steps", "Steps", "T_max", os.path.join(plot_path, "T_max_over_time.png"), window_size=env.pop_size)
    plot_time_series(T_esp_history, "T_esp Over Steps", "Steps", "T_esp", os.path.join(plot_path, "T_esp_over_time.png"), window_size=env.pop_size)
    plot_time_series(T_scale_history, "T_scale Over Steps", "Steps", "T_scale", os.path.join(plot_path, "T_scale_over_time.png"), window_size=env.pop_size)

    
    # Save statistics to a file
    stats_file = os.path.join(plot_path, 'simulation_print.txt')
    with open(stats_file, 'w') as f:
        print(f"Statistics averages of simulation for the Taxation Environment alpha={env.alpha}, reward_type={env.reward_type}", file=f)
        # Add actual prints for each statistic collected:
        print(f"Average Gini index: {np.mean(gini_index_history):.6f}", file=f)
        print(f"Average Reward: {np.mean(reward_history):.6f}", file=f)
        print(f"Average Avg Tax Rate: {np.mean(average_tax_rate_history):.6f}", file=f)
        print(f"Average Total Wealth: {np.mean(total_wealth_history):.6f}", file=f)
        print(f"Average Relative Consumptions: {np.mean(avg_rel_consumptions_history):.6f}", file=f)
        print(f"Average Mean Wealth: {np.mean(wealth_mean_history):.6f}", file=f)
        print(f"Average Median Wealth: {np.mean(wealth_median_history):.6f}", file=f)
        print('\n', file=f)

        # Print mean, median std, max e min of salaries
        print(f"Mean of exp salaries: {np.mean(expected_salaries):.6f}", file=f)
        print(f"Median of exp salaries: {np.median(expected_salaries):.6f}", file=f)
        print(f"Standard deviation of exp salaries: {np.std(expected_salaries):.6f}", file=f)
        print(f"Max of exp salaries: {np.max(expected_salaries):.6f}", file=f)
        print(f"Min of exp salaries: {np.min(expected_salaries):.6f}", file=f)
        print('\n', file=f)

        #Stampa i percentili a che ricchezza corrispondono:
        print(f"Percentiles: {env.percentiles}", file=f)
        print(f"Final wealth for each percentile: {[np.percentile(final_wealth, p) for p in env.percentiles]}", file=f)
        print('\n', file=f)

        # Print final Tmax, T_esp, T_scale
        print(f"Final T_max: {T_max_history[-1]:.6f}", file=f)
        print(f"Final T_esp: {T_esp_history[-1]:.6f}", file=f)
        print(f"Final T_scale: {T_scale_history[-1]:.6f}", file=f)
        print('\n', file=f)

        # Print mean and median, std, max e min of T_max, T_esp, T_scale
        print(f"Mean of T_max: {np.mean(T_max_history):.6f}", file=f)
        print(f"Median of T_max: {np.median(T_max_history):.6f}", file=f)
        print(f"Standard deviation of T_max: {np.std(T_max_history):.6f}", file=f)
        print(f"Max of T_max: {np.max(T_max_history):.6f}", file=f)
        print(f"Min of T_max: {np.min(T_max_history):.6f}", file=f)
        print('\n', file=f)

        print(f"Mean of T_esp: {np.mean(T_esp_history):.6f}", file=f)
        print(f"Median of T_esp: {np.median(T_esp_history):.6f}", file=f)
        print(f"Standard deviation of T_esp: {np.std(T_esp_history):.6f}", file=f)
        print(f"Max of T_esp: {np.max(T_esp_history):.6f}", file=f)
        print(f"Min of T_esp: {np.min(T_esp_history):.6f}", file=f)
        print('\n', file=f)

        print(f"Mean of T_scale: {np.mean(T_scale_history):.6f}", file=f)
        print(f"Median of T_scale: {np.median(T_scale_history):.6f}", file=f)
        print(f"Standard deviation of T_scale: {np.std(T_scale_history):.6f}", file=f)
        print(f"Max of T_scale: {np.max(T_scale_history):.6f}", file=f)
        print(f"Min of T_scale: {np.min(T_scale_history):.6f}", file=f)
        print('\n', file=f)

        print(f"Mean of W_scale: {np.mean(W_scale_history):.6f}", file=f)
        print(f"Median of W_scale: {np.median(W_scale_history):.6f}", file=f)
        print(f"Standard deviation of W_scale: {np.std(W_scale_history):.6f}", file=f)
        print(f"Max of W_scale: {np.max(W_scale_history):.6f}", file=f)
        print(f"Min of W_scale: {np.min(W_scale_history):.6f}", file=f)
        print('\n', file=f)

        # Print mean and median, std min max of final relative perc_consumptions
        print(f"Mean of final average relative perc_consumptions: {np.mean(avg_rel_consumptions_history):.6f}", file=f)
        print(f"Median of final average relative perc_consumptions: {np.median(avg_rel_consumptions_history):.6f}", file=f)
        print(f"Standard deviation of final average relative perc_consumptions: {np.std(avg_rel_consumptions_history):.6f}", file=f)
        print(f"Max of final average relative perc_consumptions: {np.max(avg_rel_consumptions_history):.6f}", file=f)
        print(f"Min of final average relative perc_consumptions: {np.min(avg_rel_consumptions_history):.6f}", file=f)
        print('\n', file=f)

        # Print mean, median std, max e min of wealth
        print(f"Mean of final wealth: {np.mean(final_wealth):.6f}", file=f)
        print(f"Median of final wealth: {np.median(final_wealth):.6f}", file=f)
        print(f"Standard deviation of final wealth: {np.std(final_wealth):.6f}", file=f)
        print(f"Max of final wealth: {np.max(final_wealth):.6f}", file=f)
        print(f"Min of final wealth: {np.min(final_wealth):.6f}", file=f)
        print('\n', file=f)

        # Print mean, median std, max e min of final tax_rate_perc (it's a list of lists so pick latest list)
        print(f"Mean of final tax_rate_perc: {np.mean(tax_rates_perc_history[-1]):.6f}", file=f)
        print(f"Median of final tax_rate_perc: {np.median(tax_rates_perc_history[-1]):.6f}", file=f)
        print(f"Standard deviation of final tax_rate_perc: {np.std(tax_rates_perc_history[-1]):.6f}", file=f)
        print(f"Max of final tax_rate_perc: {np.max(tax_rates_perc_history[-1]):.6f}", file=f)
        print(f"Min of final tax_rate_perc: {np.min(tax_rates_perc_history[-1]):.6f}", file=f)
        print('\n', file=f)

        #Pirnt mean, median std, max e min of final perc_consumptions (it's a list of lists)
        print(f"Mean of final perc_consumptions: {np.mean(consumptions_history[-1]):.6f}", file=f)
        print(f"Median of final perc_consumptions: {np.median(consumptions_history[-1]):.6f}", file=f)
        print(f"Standard deviation of final perc_consumptions: {np.std(consumptions_history[-1]):.6f}", file=f)
        print(f"Max of final perc_consumptions: {np.max(consumptions_history[-1]):.6f}", file=f)
        print(f"Min of final perc_consumptions: {np.min(consumptions_history[-1]):.6f}", file=f)
        print('\n', file=f)

    # Plots form utils:
    plot_consumptions_vs_wealth(final_wealth, plot_path, T_max_history[-1], T_esp_history[-1], T_scale_history[-1], env, expected_salaries)
    plot_absolute_consumptions_vs_wealth(final_wealth, plot_path, T_max_history[-1], T_esp_history[-1], T_scale_history[-1], env, expected_salaries)
    plot_tax_function_vs_wealth(final_wealth, T_max_history[-1], T_esp_history[-1], T_scale_history[-1], plot_path, env)
    plot_tax_function_with_distributions(final_wealth, expected_salaries, plot_path, T_max_history[-1], T_esp_history[-1], T_scale_history[-1], env)

    #final_tax_rates = [env._taxes(np.percentile(final_wealth, p), env.new_taxes_params) for p in env.percentiles]
    plot_tax_function_wrt_percentiles(env.percentiles, plot_path, tax_rates_perc_history[-1])

    # Plot the tax rates for each percentile over time
    for p_idx, p in enumerate(env.percentiles):
        plot_time_series([tax_rates_perc[p_idx] for tax_rates_perc in tax_rates_perc_history], f"Tax Rates for {p}th Percentile Over Time", "Steps", "Tax Rate", os.path.join(plot_path, f"tax_rates_{p}th_percentile_over_time.png"), window_size=env.pop_size)
    
    # Plot the wealth percentiles over time
    for p_idx, p in enumerate(env.percentiles):
        plot_time_series([percentile[p_idx] for percentile in wealth_percentiles_history], f"{p}th Percentile Over Time", "Steps", "Wealth", os.path.join(plot_path, f"{p}th_percentile_over_time.png"), window_size=env.pop_size)

    # Plot perc_consumptions for each percentile over time
    for p_idx, p in enumerate(env.percentiles):
        plot_time_series([consumption[p_idx] for consumption in consumptions_history], f"Consumptions for {p}th Percentile Over Time", "Steps", "Consumptions", os.path.join(plot_path, f"consumptions_{p}th_percentile_over_time.png"), window_size=env.pop_size)

    # Create a folder in results/ if it does not exist called tax_param_vs_alpha
    tax_param_vs_alpha_path = os.path.join("results", "tax_param_vs_alpha")
    os.makedirs(tax_param_vs_alpha_path, exist_ok=True)

    # Check if folder called as the algorithm exists in /results/tax_param_vs_alpha, if not create it
    tax_param_vs_alpha_path = os.path.join(tax_param_vs_alpha_path, algorithm)
    os.makedirs(tax_param_vs_alpha_path, exist_ok=True)

    # Create the file path for the CSV
    csv_file_path = os.path.join(tax_param_vs_alpha_path, f"{env.reward_type.value}_mean_tax_vs_alpha.csv")
    csv_file_path2 = os.path.join(tax_param_vs_alpha_path, f"{env.reward_type.value}_median_tax_vs_alpha.csv")
    csv_file_path3 = os.path.join(tax_param_vs_alpha_path, f"{env.reward_type.value}_std_tax_vs_alpha.csv")
    csv_file_path4 = os.path.join(tax_param_vs_alpha_path, f"{env.reward_type.value}_max_tax_vs_alpha.csv")
    csv_file_path5 = os.path.join(tax_param_vs_alpha_path, f"{env.reward_type.value}_min_tax_vs_alpha.csv")
    csv_file_path6 = os.path.join(tax_param_vs_alpha_path, f"{env.reward_type.value}_final_tax_vs_alpha.csv")

    # Check if the file exists and if it's empty to decide on writing the header
    write_header = not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0
    write_header2 = not os.path.exists(csv_file_path2) or os.stat(csv_file_path2).st_size == 0
    write_header3 = not os.path.exists(csv_file_path3) or os.stat(csv_file_path3).st_size == 0
    write_header4 = not os.path.exists(csv_file_path4) or os.stat(csv_file_path4).st_size == 0
    write_header5 = not os.path.exists(csv_file_path5) or os.stat(csv_file_path5).st_size == 0
    write_header6 = not os.path.exists(csv_file_path6) or os.stat(csv_file_path6).st_size == 0

    # Open the file in append mode, write the header if necessary and the data row
    with open(csv_file_path, 'a') as file:
        if write_header:
            file.write("alpha,T_max,T_esp,T_scale\n")
        file.write(f"{env.alpha:.6f},{np.mean(T_max_history):.6f},{np.mean(T_esp_history):.6f},{np.mean(T_scale_history):.6f}\n")
    
    with open(csv_file_path2, 'a') as file:
        if write_header2:
            file.write("alpha,T_max,T_esp,T_scale\n")
        file.write(f"{env.alpha:.6f},{np.median(T_max_history):.6f},{np.median(T_esp_history):.6f},{np.median(T_scale_history):.6f}\n")

    with open(csv_file_path3, 'a') as file:
        if write_header3:
            file.write("alpha,T_max,T_esp,T_scale\n")
        file.write(f"{env.alpha:.6f},{np.std(T_max_history):.6f},{np.std(T_esp_history):.6f},{np.std(T_scale_history):.6f}\n")

    with open(csv_file_path4, 'a') as file:
        if write_header4:
            file.write("alpha,T_max,T_esp,T_scale\n")
        file.write(f"{env.alpha:.6f},{np.max(T_max_history):.6f},{np.max(T_esp_history):.6f},{np.max(T_scale_history):.6f}\n")
    
    with open(csv_file_path5, 'a') as file:
        if write_header5:
            file.write("alpha,T_max,T_esp,T_scale\n")
        file.write(f"{env.alpha:.6f},{np.min(T_max_history):.6f},{np.min(T_esp_history):.6f},{np.min(T_scale_history):.6f}\n")
    
    with open(csv_file_path6, 'a') as file:
        if write_header6:
            file.write("alpha,T_max,T_esp,T_scale\n")
        file.write(f"{env.alpha:.6f},{T_max_history[-1]:.6f},{T_esp_history[-1]:.6f},{T_scale_history[-1]:.6f}\n")
    
    # For each percentile save in a csv file its final value, mean, median, max and min of the tax rate along with the value of alpha in the folder tax_param_vs_alpha.
    # Open the file in append mode and name it including the type of reward and the percentile
    for p_idx, p in enumerate(env.percentiles):
        csv_file_path = os.path.join(tax_param_vs_alpha_path, f"{env.reward_type.value}_final_tax_{p}th_percentile_vs_alpha.csv")
        write_header = not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0
        with open(csv_file_path, 'a') as file:
            if write_header:
                file.write("alpha,final_tax_rate, mean_tax_rate, median_tax_rate, std_tax_rate, max_tax_rate, min_tax_rate\n")
            file.write(f"{env.alpha:.6f},{tax_rates_perc_history[-1][p_idx]:.6f},{np.mean([tax_rates_perc[p_idx] for tax_rates_perc in tax_rates_perc_history]):.6f},{np.median([tax_rates_perc[p_idx] for tax_rates_perc in tax_rates_perc_history]):.6f},{np.std([tax_rates_perc[p_idx] for tax_rates_perc in tax_rates_perc_history]):.6f},{np.max([tax_rates_perc[p_idx] for tax_rates_perc in tax_rates_perc_history]):.6f},{np.min([tax_rates_perc[p_idx] for tax_rates_perc in tax_rates_perc_history]):.6f}\n")
    