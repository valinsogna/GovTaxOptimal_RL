import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_means_with_error_bars(file_path_means, file_path_stds, algorithm, reward_type, legend_labels=None, colors=None, title=''):
    """
    Reads data from two files: one containing mean values and the other containing standard deviations, sorts them by 'alpha', 
    and plots the averages with error bars representing the standard deviation for specified columns against 'alpha' after rescaling.
    The plot is saved with a filename that includes the algorithm and reward type.

    Parameters:
    - file_path_means: string, path to the data file with mean values.
    - file_path_stds: string, path to the data file with standard deviations.
    - algorithm: string, the name of the algorithm used.
    - reward_type: string, the type of reward used.
    - legend_labels: list of strings or None, labels for the legend corresponding to the columns to plot. Defaults to ['T_max', 'T_esp', 'T_scale'] if None.
    - colors: list of strings or None, colors for each plot. Defaults to ['blue', 'red', 'green'] if None.
    - title: string, title of the plot. If empty, a default title including algorithm and reward_type will be used.
    """
    if legend_labels is None:
        legend_labels = ['$T_{max}$', '$T_{esp}$', '$T_{scale}$']
    if colors is None:
        colors = ['blue', 'red', 'green']
    if len(legend_labels) != len(colors):
        raise ValueError("legend_labels and colors must have the same length.")
    
    # Define scaling factors for normalization
    scales = {'T_max': 10, 'T_esp': 1/3, 'T_scale': 1}
    
    # Load the data and sort by 'alpha'
    df_means = pd.read_csv(file_path_means).sort_values(by='alpha')
    df_stds = pd.read_csv(file_path_stds).sort_values(by='alpha')

    # Setting up the plot
    plt.figure(figsize=(5, 4))

    # Plotting each column against 'alpha' with error bars, applying scaling
    for y_label, legend_label, color in zip(['T_max', 'T_esp', 'T_scale'], legend_labels, colors):
        scaled_means = df_means[y_label] * scales[y_label]
        scaled_stds = df_stds[y_label] * scales[y_label]
        # Correcting std to not go below 0 or above 1 after scaling
        lower_limits = np.maximum(scaled_means - scaled_stds, 0)
        upper_limits = np.minimum(scaled_means + scaled_stds, 1)
        corrected_stds_lower = scaled_means - lower_limits
        corrected_stds_upper = upper_limits - scaled_means

        plt.errorbar(df_means['alpha'], scaled_means, yerr=[corrected_stds_lower, corrected_stds_upper], label=legend_label, fmt='o', color=color, ecolor=color, elinewidth=3, capsize=5, alpha=0.75)
        plt.plot(df_means['alpha'], scaled_means, color=color, linestyle='-', alpha=0.5)

    # Setting the title
    if not title:
        title = f"$T_{{max}}$, $T_{{esp}}$, $T_{{scale}}$ Averages (Last 100 Steps)"# vs $\\alpha$ in {algorithm} {reward_type}"
    plt.title(title)

    # Adding legend, grid, labels
    plt.legend()
    plt.grid(True)
    plt.xlabel('$\\alpha$', fontsize=14)
    plt.ylabel('Normalized Values')

    # Saving the plot
    filename = f"{algorithm}_{reward_type}_normalized_plot.png"
    plt.savefig(filename)

    # Optionally, display the plot
    #plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 5:
        reward_type = sys.argv[1]
        algorithm = sys.argv[2]
        file_path_means = sys.argv[3]
        file_path_stds = sys.argv[4]
    else:
        raise RuntimeError("You must provide reward type, algorithm, file_path for means, and file_path for stds as command line arguments")
    
    if not (os.path.exists(file_path_means) and os.path.exists(file_path_stds)):
        raise RuntimeError("One or both of the specified files do not exist")

    plot_means_with_error_bars(file_path_means=file_path_means, file_path_stds=file_path_stds, algorithm=algorithm, reward_type=reward_type)
