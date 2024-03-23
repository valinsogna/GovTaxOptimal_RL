import numpy as np
import matplotlib.pyplot as plt

x_range = np.logspace(0, 11, 1000)

# Define a function to convert numbers to Unicode superscript
def to_superscript(n):
    superscript_map = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
    }
    return ''.join(superscript_map.get(char, char) for char in str(n))


def _taxes(wealth, taxes_params):
    max, esp, scale_percentile = taxes_params
    # old version:
    # ts = max * (1 - 1 / (1 + (wealth / np.exp(scale*25))**esp))
    # Pick the true scale parameter as the wealth at the scale_percentile
    scale = np.percentile(wealth, scale_percentile*100)# returns a variable of dimension 1
    print(scale)
    # new version:
    ts = max * (1 - 1 / (1 + (wealth / scale)**esp))
    return ts

_taxes_params = [0.064, 0.63, 0.74]  # PPO_10000000_steps_MlpPolicy_alpha=0.0_envs=4_R1_T_max=0.1

wealth = np.random.lognormal(12,2,1000)
#sort wealth
wealth = np.sort(wealth)
#taxes
tasse = _taxes(wealth, _taxes_params)


plt.plot(wealth, tasse)
plt.xscale('log')
plt.xlabel('Wealth')
plt.ylabel('Tax')
plt.title('Taxation function')
plt.grid(True)
# Set x-ticks for log scale
# tick_values = [10**i for i in range(0, 12, 1)]  # Powers of 10 from 10^0 to 10^10
# tick_labels = ['10{}'.format(to_superscript(i)) for i in range(0, 12, 1)]  # Label them as 10^0, 10^1, etc.
# plt.xticks(tick_values, tick_labels)
# Set y-ticks from 0 to 0.1
plt.yticks(np.arange(0, 0.1, 0.02))
# # Adding a vertical red line at x = 8886110.52051
# plt.axvline(x=8886110.52051, color='red', linestyle='-', linewidth=2)

#plt.savefig("tasse.png")
plt.show()
