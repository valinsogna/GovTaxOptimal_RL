import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation_results(file_path, optional_file_path=None):
    with open(file_path, 'r') as file:
        simulation_data = json.load(file)
    
    if optional_file_path:
        with open(optional_file_path, 'r') as optional_file:
            optional_simulation_data = json.load(optional_file)
    else:
        optional_simulation_data = None

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    optional_base_name = os.path.splitext(os.path.basename(optional_file_path))[0] if optional_file_path else ""
    
    # Prepare for cumulative reward vs. alpha plot
    alphas = [data['alpha'] for data in simulation_data]
    cumulative_rewards = [data['final_cumulative_reward'] for data in simulation_data]
    
    if optional_simulation_data:
        optional_alphas = [data['alpha'] for data in optional_simulation_data]
        optional_cumulative_rewards = [data['final_cumulative_reward'] for data in optional_simulation_data]
    
    # Plotting cumulative reward vs. alpha
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(alphas, cumulative_rewards, label=f"{base_name}", color='#FF5733')  # Vibrant Orange
    ax.plot(alphas, cumulative_rewards, color='#FF5733')  # Connect points for the first algorithm
    
    if optional_simulation_data:
        ax.scatter(optional_alphas, optional_cumulative_rewards, label=f"{optional_base_name}", color='#9D00FF')  # Distinct Purple
        ax.plot(optional_alphas, optional_cumulative_rewards, color='#9D00FF')  # Connect points for the second algorithm
    
    ax.set_title("Cumulative Reward in 1 Simulation vs " + r"$\alpha$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(file_path), f"cumulative_reward_vs_alpha_comparison.png")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
    print(f"Comparative plot saved to {save_path}")

if __name__ == "__main__":
    file_path = input("Enter the file path of the JSON data: ")
    optional_file_path = input("Enter the optional file path of the JSON data for another algorithm (leave blank if not applicable): ")
    plot_simulation_results(file_path, optional_file_path if optional_file_path.strip() != "" else None)
