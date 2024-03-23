import numpy as np
#from TaxationEnv import TaxationEnv as te
import matplotlib.pyplot as plt
# x_range = np.linspace(0, 10000000, 1000)

# sol = np.random.lognormal(10, 1, size=1)
# print(sol)

x_range = np.logspace(0, 11, 1000)
parametri_default_consumi=[0.8, 5], # esponente eta, moltiplicatore K
stipendi_attesi = 30000
stipendi_attesi2 = 80000

def _tasse(wealth, parametri_tasse):
    # Calcola le tasse qui, basandoti sulla funzione tasse del tuo script
    max, esp, scale = parametri_tasse
    # old version:
    # ts = max * (1 - 1 / (1 + (wealth / np.exp(scale*25))**esp))
    # Pick the true scale parameter as the wealth at the scale_percentile
    # scale = np.percentile(wealth, scale_percentile*100)# returns a variable of dimension 1
    # new version:
    W_scale = np.percentile(wealth, scale*100)
    ts = max * (1 - 1 / (1 + (wealth / W_scale)**esp))
    return ts

def _consumi(wealth, stipendi_attesi, parametri_tasse):
    # Calcola i consumi relativi alla ricchezza 
    η = 0.8 # esponente
    K = 5 # moltiplicatore
    cons = 1 / (1 + ((wealth * (1/(K * stipendi_attesi) + 10**-5)) *
                     (1 - _tasse(wealth, parametri_tasse)))**η)
    return cons

consumi = _consumi(x_range, stipendi_attesi, [0.5, 0.5, 0.5])
consumi2 = _consumi(x_range, stipendi_attesi2, [0.5, 0.5, 0.5])
consumi_abs = consumi*x_range
consumi_abs2 = consumi2*x_range
# plt.figure(figsize=(7, 4.2))
# plt.plot(x_range, consumi, label='Expected salary: 30000')
# plt.plot(x_range, consumi2, label='Expected salary: 80000')
# plt.xscale('log')
# plt.xlabel('Wealth (log scale)')
# plt.ylabel('Relative Consumption')
# plt.title('Relative Consumption for $T_{max}$, $T_{esp}$, $T_{scale}=0.5$')

# plt.grid(True)
# plt.legend()
# plt.savefig("rel_consumpt")

plt.plot(x_range, consumi_abs, label='Expected salary: 30000')
plt.plot(x_range, consumi_abs2, label='Expected salary: 80000')
plt.xscale('log')
plt.xlabel('Wealth (log scale)')
plt.ylabel('Absolute Consumption')
plt.title('Absolute Consumption for $T_{max}$, $T_{esp}$, $T_{scale}=0.5$')
plt.grid(True)
plt.legend()
plt.savefig("abs_consumpt")

