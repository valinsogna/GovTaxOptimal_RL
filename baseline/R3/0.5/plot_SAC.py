import pandas as pd
import matplotlib.pyplot as plt

# Load SAC algorithm results from a CSV string for demonstration. Replace this with your actual CSV file path.
sac_data = pd.read_csv('results_SAC.csv')

# Load MOPSO algorithm results from a CSV string for demonstration. Replace this with your actual CSV file path.
mopso_data = pd.read_csv('MOPSO.csv')

# Rename columns of MOPSO data to match those of SAC for consistency
mopso_data.rename(columns={"Objective 1 (e.g., -Inequality)": "objective1",
                           "Objective 2 (e.g., Consumptions)": "objective2"}, inplace=True)

# Aggregate SAC data by alpha, calculating the mean for each alpha
sac_data = sac_data.groupby('alpha')[['objective1', 'objective2']].mean().reset_index()

# Ensure the data is sorted by alpha
sac_data.sort_values('alpha', inplace=True)

# Plotting
plt.figure(figsize=(5, 4))
# Use a colormap
cmap = plt.get_cmap('turbo')  # This colormap varies from blue (cool) to red (warm)

# Normalize alpha values to the [0, 1] range for the colormap
norm = plt.Normalize(sac_data['alpha'].min(), sac_data['alpha'].max())

# Plot SAC results with varying colors
for _, row in sac_data.iterrows():
    plt.scatter(row['objective1'], row['objective2'],s=70, color=cmap(norm(row['alpha'])), label=f"SAC alpha={row['alpha']}")

# Since the labels will be redundant, we create a custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=rf'$\alpha$ = {alpha}', markerfacecolor=cmap(norm(alpha))) for alpha in sac_data['alpha'].unique()] + [Line2D([0], [0], marker='x', color='k', label='MOPSO', markersize=10)]
plt.legend(handles=legend_elements, loc='best', fontsize='xx-small')

# Plot MOPSO results
plt.scatter(mopso_data['objective1'], mopso_data['objective2'], color='k', marker='x', s=15)

plt.xlabel('Objective 1 (-Inequality)')
plt.ylabel('Objective 2 (Consumption)')
plt.title('Comparison of SAC and MOPSO')
plt.grid(True)
#plt.show()
plt.savefig("SAC_vs_MOPSO.png")