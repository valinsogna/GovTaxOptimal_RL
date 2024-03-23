import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation_results(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    cmap = plt.get_cmap('turbo')  # Define colormap
    
    alphas = [data['alpha'] for data in simulation_data]
    colors = cmap(np.linspace(0, 1, len(alphas)))  # Map each alpha to a color
    
    # Define a list of markers to cycle through
    markers = ['o', 's', '^', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
    
    for plot_index, plot_name in enumerate(['gini_vs_steps', 'reward_vs_steps', 'wealth_mean_vs_steps', 'tax_rates_per_percentile']):
        fig, ax = plt.subplots(figsize=(5, 4))
        
        for data, color, marker in zip(simulation_data, colors, markers * (len(simulation_data) // len(markers) + 1)):  # Ensure enough markers
            alpha = data['alpha']
            steps = list(range(len(data['gini_index'])))
            
            # Use LaTeX notation for alpha in the label
            label = r'$\alpha=' + f"{alpha}$"
            
            if plot_index == 0:  # Gini vs. Steps
                ax.plot(steps, data['gini_index'], label=label, color=color)
            elif plot_index == 1:  # Reward vs. Steps
                ax.plot(steps, data['reward'], label=label, color=color)
            elif plot_index == 2:  # Wealth Mean with Error Bars (std)
                ax.plot(steps, data['wealth_mean'], label=label, color=color)
                ax.set_yscale('log')  # Apply log scale to y-axis
            elif plot_index == 3:  # Tax Function vs. Percentile
                percentiles = list(data['tax_rate_vs_percentile_last_100_avg_std'].keys())
                avg_tax_rates = np.array([data['tax_rate_vs_percentile_last_100_avg_std'][p]['average'] for p in percentiles])
                # Connect points with lines, use different markers, and include LaTeX label
                ax.plot(percentiles, avg_tax_rates, label=label, color=color, marker=marker, linestyle='-', linewidth=1, markersize=8)
                
        # Custom titles and labels for plots 2 and 3
        if plot_index == 2:  # Custom title, xlabel, ylabel for plot 3
            ax.set_title("Average Wealth Evolution in Population Over Time")
            ax.set_xlabel("Simulation Steps")
            ax.set_ylabel("Wealth (log scale)")
        elif plot_index == 3:  # Already customized for "Tax Rates per Percentile"
            ax.set_title("Tax Rates Averages (Last 100 Steps) per Percentile")
            ax.set_xlabel("Wealth Percentile")
            ax.set_ylabel("Tax Rate")
        else:  # Default settings for other plots
            ax.set_title(plot_name.replace('_', ' ').title())
            ax.set_xlabel("Steps" if plot_index != 3 else "Average Sorted Wealth")
            ax.set_ylabel(plot_name.split('_')[0].title())
        
        # Adjust legend size using fontsize
        ax.legend(fontsize='x-small')
        
        plt.tight_layout()
        
        save_path = os.path.join(plot_directory, f"{base_name}_{plot_name}.png")
        plt.savefig(save_path)
        plt.close()
        # save_path = os.path.join(os.path.dirname(file_path), f"{base_name}_{plot_name}.png")
        # plt.savefig(save_path)
        # plt.close()  # Close the plot to free memory
        # print(f"Plot saved to {save_path}")
        
def plot_tax_parameters(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Define scaling factors for normalization and the parameters to plot
    scales = {'T_max': 10, 'T_esp': 1/3, 'T_scale': 1}
    parameters = ['T_max', 'T_esp', 'T_scale']
    colors = ['blue', 'red', 'green']
    legend_labels = ['$T_{max}$', '$T_{esp}$', '$T_{scale}$']
    
    alphas = [data['alpha'] for data in simulation_data]
    plt.figure(figsize=(5, 4))

    for param, color, label in zip(parameters, colors, legend_labels):
        means = np.array([data[f'{param}_last_100_avg'] for data in simulation_data])
        stds = np.array([data[f'{param}_last_100_std'] for data in simulation_data])
        scaled_means = means * scales[param]
        scaled_stds = stds * scales[param]
        
        # Ensure error bars don't suggest negative values
        lower_limits = np.maximum(scaled_means - scaled_stds, 0)
        upper_limits = scaled_means + scaled_stds
        corrected_stds_lower = scaled_means - lower_limits
        corrected_stds_upper = upper_limits - scaled_means

        plt.errorbar(alphas, scaled_means, yerr=[corrected_stds_lower, corrected_stds_upper], label=label, fmt='o', color=color, ecolor=color, elinewidth=3, capsize=5, alpha=0.75)
        plt.plot(alphas, scaled_means, color=color, linestyle='-', alpha=0.5)

    plt.title(f"Avg Tax Parameters (Last 100 Steps) vs. $\\alpha$")
    plt.xlabel('$\\alpha$')
    plt.ylabel('Normalized Values')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(plot_directory, f"{base_name}_tax_parameters_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Tax Parameters plot saved to {save_path}")


    #print(f"Tax Parameters plot saved to {save_path}")

def plot_w_scale(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    alphas = [data['alpha'] for data in simulation_data]
    means = np.array([data['W_scale_last_100_avg'] for data in simulation_data])
    stds = np.array([data['W_scale_last_100_std'] for data in simulation_data])

    # Correct the standard deviations to ensure they don't go below zero
    lower_limits = np.maximum(means - stds, 0)
    upper_limits = means + stds  # No need for an upper limit on log scale, but kept for consistency
    corrected_stds_lower = means - lower_limits
    corrected_stds_upper = upper_limits - means

    plt.figure(figsize=(5, 4))
    plt.errorbar(alphas, means, yerr=[corrected_stds_lower, corrected_stds_upper], label='$W_{scale}$', fmt='o', color='purple', ecolor='purple', elinewidth=3, capsize=5, alpha=0.75)
    plt.plot(alphas, means, color='purple', linestyle='-', alpha=0.5)

    plt.yscale('log')  # Set the y-axis to a logarithmic scale

    plt.title(f"Avg $W_{{scale}}$ (Last 100 Steps) vs. $\\alpha$")
    plt.xlabel('$\\alpha$')
    plt.ylabel('$W_{{scale}}$ (log scale)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(plot_directory, f"{base_name}_W_scale_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Avg $W_{{scale}}$ plot saved to {save_path}")


    #print(f"$W_{{scale}}$ plot saved to {save_path}")

def set_log_scale_xaxis(values):
    """Adjusts x-axis to log scale based on the range of values."""
    plt.xlim([min(values) * 0.9, max(values) * 1.1])
    plt.xscale('log')

def plot_tax_function_with_distributions(simulation_data, base_name, plot_directory):
    """Plots the tax function with wealth and expected salaries distributions for each alpha."""
    for data in simulation_data:
        alpha = data['alpha']
        final_wealth_sorted = data['final_wealth_sorted']
        expected_salaries = data['expected_salaries']
        tax_rates_final_wealth = data['tax_rates_final_wealth']
        
        fig, ax1 = plt.subplots(figsize=(6, 4.5))

        # Plot Tax Function
        ax1.plot(final_wealth_sorted, tax_rates_final_wealth, label='Tax Rates', color='blue', linewidth=2)
        ax1.set_xlabel("Wealth (log scale)")
        ax1.set_ylabel("Tax Rates", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xscale('log')

        # Instantiate a second y-axis to plot wealth and salary distributions
        ax2 = ax1.twinx()
        # Plotting Distributions
        logbins = np.logspace(np.log10(min(final_wealth_sorted)), np.log10(max(final_wealth_sorted)), 100)
        # linear_bins = np.linspace(np.log(min(final_wealth_sorted)), np.log(max(final_wealth_sorted)), 50)
        # logbins = np.exp(linear_bins)
        #logbins2 = np.logspace(np.log10(min(expected_salaries)), np.log10(max(expected_salaries)), 50)
        ax2.hist(final_wealth_sorted, bins=logbins, alpha=0.5, label='Final Wealth Distribution', color='green', density=True)
        #ax2.hist(expected_salaries, bins=logbins2, alpha=0.5, label='Expected Salaries Distribution', color='red', density=True)
        ax2.set_ylabel('Probability Density', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.title(f"Final Tax Function vs Wealth for $\\alpha$={alpha}")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

        plot_path = os.path.join(plot_directory, f"{base_name}_alpha_{alpha}_tax_function_distributions.png")
        plt.savefig(plot_path)
        plt.close()
        #print(f"Plot for alpha={alpha} saved to {plot_path}")

def plot_avg_rel_consumptions_and_gini_index_with_errors(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    
    alphas = [data['alpha'] for data in simulation_data]
    avg_rel_consumptions = [data['avg_rel_consumptions_last_100'] for data in simulation_data]
    std_avg_rel_consumptions = [data['std_avg_rel_consumptions_last_100'] for data in simulation_data]
    gini_index_last_100_avg = [data['gini_index_last_100_avg'] for data in simulation_data]
    gini_index_last_100_std = [data['gini_index_last_100_std'] for data in simulation_data]

    plt.figure(figsize=(5, 4))
    plt.errorbar(alphas, avg_rel_consumptions, yerr=std_avg_rel_consumptions, label='Rel Consumptions', fmt='o', color='navy', ecolor='lightblue', elinewidth=3, capsize=5)
    plt.errorbar(alphas, gini_index_last_100_avg, yerr=gini_index_last_100_std, label='Gini Index', fmt='o', color='darkgreen', ecolor='lightgreen', elinewidth=3, capsize=5)

    plt.title("Avg Rel Consumptions (Last 100 Steps) & Gini Index vs $\\alpha$")
    plt.xlabel('$\\alpha$')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(plot_directory, "avg_rel_consumptions_and_gini_index_vs_alpha.png")
    plt.savefig(save_path)
    plt.close()
    #print(f"Plot saved to {save_path}")

def plot_other_variables_separately(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    
    alphas = [data['alpha'] for data in simulation_data]
    variables = ['final_wealth_avg', 'gini_normalized_last_100_avg', 'reward_consumption_part_last_100_avg', 'cv_last_100_avg']
    std_variables = ['final_wealth_std', 'gini_normalized_last_100_std', 'reward_consumption_part_last_100_std', 'cv_last_100_std']
    labels = ['Avg Final Wealth Across Pop in simulation', 'Normalized Gini Index (Last 100 Steps)', 'Reward Consumption Part (Last 100 Steps)', 'Coeff. of Variation CV (Last 100 Steps)']
    #titles = ['Avg Final Wealth Across Pop in simulation', 'Normalized Gini Index (Last 100 Steps)', 'Reward Consumption Part (Last 100 Steps)', 'Coeff. of Variation CV (Last 100 Steps)']
    colors = ['blue', 'blue', 'blue', 'blue']

    for (var, std_var, label, color) in zip(variables, std_variables, labels, colors):
        means = np.array([data[var] for data in simulation_data])
        stds = np.array([data[std_var] for data in simulation_data])

        plt.figure(figsize=(5, 4))

        # Calculate the lower and upper limits of the standard deviation
        lower_std = means - stds
        upper_std = means + stds

        # Ensure that the lower limit is not less than a small positive value
        # This is to avoid negative values on a log scale
        lower_limit = np.maximum(lower_std, 1e-5)
        # The upper limit does not usually need adjustment for a log scale

        # Correct the standard deviation bars to not go below the lower limit
        corrected_stds_lower = means - lower_limit
        corrected_stds_upper = upper_std - means

        # For final_wealth_avg, use log scale for y-axis
        if var == 'final_wealth_avg':
            plt.yscale('log')
            plt.errorbar(alphas, means, yerr=[corrected_stds_lower, corrected_stds_upper], label=label, fmt='o', color=color, ecolor=color, elinewidth=3, capsize=5)
        else:
            plt.errorbar(alphas, means, yerr=stds, label=label, fmt='o', color=color, ecolor=color, elinewidth=3, capsize=5)

        plt.title(f"{label} vs $\\alpha$")
        plt.xlabel('$\\alpha$')
        if var == 'final_wealth_avg':
            plt.ylabel('Values (log scale)')
        else:
            plt.ylabel('Values')
        #plt.legend()
        plt.grid(True)
        save_path = os.path.join(plot_directory, f"{var}_vs_alpha.png")
        plt.savefig(save_path)
        plt.close()
        #print(f"Plot for {label} saved to {save_path}")


def ensure_directory_exists(directory):
    """Ensures the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    file_path = input("Enter the file path of the JSON data: ")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_directory = os.path.join(os.path.dirname(file_path), base_name)
    ensure_directory_exists(plot_directory)
    
    # Update plot saving paths in each plot function call to use `plot_directory`
    # plot_simulation_results(file_path, plot_directory)
    # plot_tax_parameters(file_path, plot_directory)
    # plot_w_scale(file_path, plot_directory)
    # Load simulation data once and pass it along to avoid multiple file reads
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    plot_tax_function_with_distributions(simulation_data, base_name, plot_directory)
    # plot_avg_rel_consumptions_and_gini_index_with_errors(file_path, plot_directory)
    # plot_other_variables_separately(file_path, plot_directory)