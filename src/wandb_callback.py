import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000, percentiles=None):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.percentiles = percentiles

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            # Get info from all environments
            infos = self.locals['infos']
            # Aggregate data
            aggregated_data = {
                'gini_index': [],
                'reward': [],
                'T_max': [],
                'T_esp': [],
                'T_scale': [],
                'total_wealth': [],
                'avg_rel_consumptions': [],
                #'total_spending': [],
                'wealth_mean': [],
                'wealth_median': [],
                'states': [[] for _ in self.percentiles],
                'perc_consumptions': [[] for _ in self.percentiles],
                'tax_rates_perc': [[] for _ in self.percentiles],
                'average_tax_rate': [],
                'W_scale': [],
                'cv': []
            }

            for info in infos:
                if 'gini_index' in info:
                    aggregated_data['gini_index'].append(info['gini_index'])
                    aggregated_data['reward'].append(info['reward'])
                    aggregated_data['T_max'].append(info['T_max'])
                    aggregated_data['T_esp'].append(info['T_esp'])
                    aggregated_data['T_scale'].append(info['T_scale'])
                    aggregated_data['W_scale'].append(info['W_scale'])
                    aggregated_data['total_wealth'].append(info['total_wealth'])
                    aggregated_data['avg_rel_consumptions'].append(info['avg_rel_consumptions'])
                    aggregated_data['wealth_mean'].append(info['wealth_mean'])
                    aggregated_data['wealth_median'].append(info['wealth_median'])
                    aggregated_data['average_tax_rate'].append(info['average_tax_rate'])
                    for idx, percentile in enumerate(self.percentiles):
                        aggregated_data['states'][idx].append(info['new_state'][idx])
                        aggregated_data['perc_consumptions'][idx].append(info['perc_consumptions'][idx])
                        aggregated_data['tax_rates_perc'][idx].append(info['tax_rates_perc'][idx])

            # Log aggregated data to wandb
            for key, values in aggregated_data.items():
                if key == 'states':
                    for idx, percentile in enumerate(self.percentiles):
                        wandb.log({f'States/Wealth Percentile {percentile}': np.mean(aggregated_data['states'][idx])}, commit=False)
                elif key == 'perc_consumptions':
                    for idx, percentile in enumerate(self.percentiles):
                        wandb.log({f'Consumptions/Wealth Percentile {percentile}': np.mean(aggregated_data['perc_consumptions'][idx])}, commit=False)
                elif key == 'tax_rates_perc':
                    for idx, percentile in enumerate(self.percentiles):
                        wandb.log({f'Tax Rates/Wealth Percentile {percentile}': np.mean(aggregated_data['tax_rates_perc'][idx])}, commit=False)
                elif key in ['T_max', 'T_esp', 'T_scale', 'W_scale', 'cv']:
                    wandb.log({f'Actions/{key}': np.mean(values)}, commit=False)
                else:
                    wandb.log({f'Gini & al/{key}': np.mean(values)}, commit=False)

            wandb.log({'_': None}, commit=True)  # Commit all data

        return True
