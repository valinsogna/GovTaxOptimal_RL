import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_wealth_distributions_vertical(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)

    # Prepare the colormap and figure
    cmap = plt.get_cmap('turbo')
    alphas = [data['alpha'] for data in simulation_data]
    num_plots = len(simulation_data)

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 2 * num_plots), sharex=True)

    # Find the global minimum and maximum wealth across all alphas to set the bins
    all_wealth = np.hstack([data['final_wealth_sorted'] for data in simulation_data])
    min_wealth = all_wealth.min()
    max_wealth = all_wealth.max()

    # Define the log-scaled bins
    logbins = np.logspace(np.log10(min_wealth), np.log10(max_wealth), 100)

    # Plot the PDF of final wealth distributions for each alpha in its own subplot
    for ax, data, alpha, color in zip(axes, simulation_data, alphas, cmap(np.linspace(0, 1, num_plots))):
        final_wealth_sorted = data['final_wealth_sorted']
        density, bins = np.histogram(final_wealth_sorted, bins=logbins, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(bin_centers, density, '-', color=color)
        ax.fill_between(bin_centers, density, color=color, alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(f'Alpha: {alpha}')
        ax.grid(True)

    fig.suptitle('Final Wealth Distributions across Alphas', y=1.02)
    fig.tight_layout()
    fig.supxlabel('Wealth')
    fig.supylabel('Probability Density')

    save_path = os.path.join(plot_directory, "wealth_distributions_vertical.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save the figure with a tight bounding box to include the legend
    plt.close()
    print(f"Wealth Distributions vertical plot saved to {save_path}")


def plot_wealth_distributions_vertical_overlap(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)

    cmap = plt.get_cmap('turbo')
    num_plots = len(simulation_data)
    
    # Adjust the figure size and spacing
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 2 + 0.5*num_plots), sharex=True)
    
    # Define the log-scaled bins
    all_wealth = np.hstack([data['final_wealth_sorted'] for data in simulation_data])
    logbins = np.logspace(np.log10(min(all_wealth)), np.log10(max(all_wealth)), 100)
    
    # Adjust margins to make room for y-axis label and ticks, also reduce the top margin
    plt.subplots_adjust(left=0.15, hspace=0, top=0.93)

    # Set larger font sizes for the title and axis labels
    main_title_fontsize = 14
    axis_label_fontsize = 12
    alpha_label_fontsize = 10
    tick_label_fontsize = 8

    # Set bold font weight
    fontweight = 'bold'

    for i, (ax, data) in enumerate(zip(axes, simulation_data)):
        alpha = data['alpha']
        color = cmap(i / num_plots)
        density, bins = np.histogram(data['final_wealth_sorted'], bins=logbins, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        ax.plot(bin_centers, density, '-', color=color)
        ax.fill_between(bin_centers, density, color=color, alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Set the alpha labels and adjust fontsize
        ax.text(0.05, 0.5, f'$\\alpha={alpha}$', fontsize=alpha_label_fontsize, transform=ax.transAxes)

        # Set y-ticks and labels for the maximum value
        max_density = max(density)
        ax.set_yticks([1e-8, max_density])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis='y', which='both', labelsize=tick_label_fontsize)

        # Remove x-ticks labels except for the last subplot
        if i < num_plots - 1:
            ax.xaxis.set_ticklabels([])

    axes[-1].set_xlabel('Wealth', fontsize=axis_label_fontsize, fontweight=fontweight)
    
    fig.suptitle('Final Wealth Distributions across Alphas', fontsize=main_title_fontsize, fontweight=fontweight)
    plt.figtext(0.01, 0.5, 'Probability Density', va='center', ha='center', rotation='vertical', fontsize=axis_label_fontsize, fontweight=fontweight)

    # Save the plot ensuring all labels are within the figure
    save_path = os.path.join(plot_directory, "wealth_distributions_vertical_overlap.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Wealth Distributions vertical overlap plot saved to {save_path}")

# Example usage:
# plot_wealth_distributions_vertical_overlap('/path/to/json', '/path/to/save/plots')

def plot_wealth_distribution_subplots(file_path, plot_directory):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    
    # Calculate number of rows needed for subplots based on the number of alphas
    num_alphas = len(simulation_data)
    fig, axes = plt.subplots(num_alphas, 1, figsize=(6, 2 * num_alphas), sharex=True)

    # If there's only one alpha, axes will not be an array but a single object.
    if num_alphas == 1:
        axes = [axes]

    # Prepare the colormap
    cmap = plt.get_cmap('turbo')
    colors = cmap(np.linspace(0, 1, num_alphas))
    
    # Define the log-scaled bins using the global wealth range
    all_wealth = np.hstack([data['final_wealth_sorted'] for data in simulation_data])
    logbins = np.logspace(np.log10(all_wealth.min()), np.log10(all_wealth.max()), 100)
    
    # Plot each subplot
    for ax, data, color, alpha_val in zip(axes, simulation_data, colors, [d['alpha'] for d in simulation_data]):
        final_wealth_sorted = data['final_wealth_sorted']
        
        # Plot histogram with log-scaled bins
        ax.hist(final_wealth_sorted, bins=logbins, density=True, color=color, alpha=0.5, label=f'Alpha: {alpha_val}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(f'Alpha: {alpha_val}')
        ax.legend(loc='upper right')
        
    # Set common labels
    fig.text(0.5, 0.04, 'Wealth', ha='center', va='center')
    fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical')
    plt.suptitle('Final Wealth Distributions across Alphas')

    save_path = os.path.join(plot_directory, "wealth_distributions_subplots.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Wealth Distributions subplot saved to {save_path}")


def ensure_directory_exists(directory):
    """Ensures the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    file_path = input("Enter the file path of the JSON data: ")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_directory = os.path.join(os.path.dirname(file_path), base_name)
    ensure_directory_exists(plot_directory)
    plot_wealth_distributions_vertical_overlap(file_path, plot_directory)