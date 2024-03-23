import numpy as np
#from TaxationEnv import TaxationEnv as te
import matplotlib.pyplot as plt
# x_range = np.linspace(0, 10000000, 1000)

x_range = np.logspace(0, 11, 1000)
# massimo, esponente, scala = parametri_tasse
# ts = massimo * (1 - 1 / (1 + (wealth / np.exp(scala*25) )**esponente))

# Define a function to convert numbers to Unicode superscript
def to_superscript(n):
    superscript_map = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
    }
    return ''.join(superscript_map.get(char, char) for char in str(n))



def _tasse(self, wealth, parametri_tasse):
    # Calcola le tasse qui, basandoti sulla funzione tasse del tuo script
    massimo, esponente, scala = parametri_tasse
    ts = massimo * (1 - 1 / (1 + (wealth / np.exp(scala*25))**esponente))
    return ts

#tasse = _tasse(None, x_range, [0.2, 0.8, 0.07])#SAC-alpha_0-T_max=0.5_R2
#tasse = _tasse(None, x_range, [0.5, 0.0001, 0.26]) #PPO-alpha_0-T_max=0.5_R2
#tasse = _tasse(None, x_range, [0.14, 0.9, 0.95]) #PPO-alpha_0-T_max=0.5_R1
#tasse = _tasse(None, x_range, [0.05, 0.9, 0.1]) #SAC/PPO-alpha_0-T_max=0.05_R2
#tasse = _tasse(None, x_range, [0.047, 0.85, 0.13]) #SAC/PPO-alpha_0-T_max=0.05_R2
#tasse = _tasse(None, x_range, [0.86, 0.9, 0.46]) #SAC_alpha_1-T_max=0.99_meanR=0.02_R1/R2
#tasse = _tasse(None, x_range, [0.15, 0.89, 0.94]) #SAC_alpha_0-T_max=0.99_meanR=0.02_R1
#tasse = _tasse(None, x_range, [0.2, 0.75, 0.09]) #R2 sbagliato vecchio
tasse = _tasse(None, x_range, [0.000047, 0.29, 0.48])#PPO_10000000_steps_MlpPolicy_alpha=0.0_envs=4_R1_T_max=0.05
#tasse = _tasse(None, x_range, [0.02, 0.89, 0.96])#SAC_10000000_steps_MlpPolicy_alpha=0.0_envs=4_R1_T_max=0.05
#tasse = _tasse(None, x_range, [0.047, 0.49, 0.10])#PPO_10000000_steps_MlpPolicy_alpha=0.0_envs=4_R2_ABScons


plt.plot(x_range, tasse)
plt.xscale('log')
plt.xlabel('Wealth')
plt.ylabel('Tax')
plt.title('Taxation function')
plt.grid(True)
# Set x-ticks for log scale
tick_values = [10**i for i in range(0, 12, 1)]  # Powers of 10 from 10^0 to 10^10
tick_labels = ['10{}'.format(to_superscript(i)) for i in range(0, 12, 1)]  # Label them as 10^0, 10^1, etc.
plt.xticks(tick_values, tick_labels)
# Set y-ticks from 0 to 0.05
plt.yticks(np.arange(0, 0.05, 0.005))
# Adding a vertical red line at x = 8886110.52051
plt.axvline(x=8886110.52051, color='red', linestyle='-', linewidth=2)

#plt.savefig("tasse.png")
plt.show()


# μ_r = 0.02
# σ_r = 0.2

# y1 =  np.random.lognormal(μ_r, σ_r, size=10000)

# μ_r = 0.04
# σ_r = 0.2
# y2 =  np.random.lognormal(μ_r, σ_r, size=10000)

# plt.hist(y1, bins=100)
# plt.hist(y2, bins=100)
# plt.show()

# μ_p = 13
# σ_p = 1
# y3 =  np.random.lognormal(μ_p, σ_p, size=10000)
# μ_p2 = 12
# σ_p2 = 1
# y4 =  np.random.lognormal(μ_p2, σ_p2, size=10000)
# plt.hist(y4, bins=1000, alpha=0.5, label="12")
# plt.hist(y3, bins=1000, alpha=0.5, label="13")
# #aggiungi legenda al plot
# plt.legend()
# plt.show()