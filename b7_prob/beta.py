import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

fig, axes = plt.subplots(1, 3, figsize=(12,4))

probs1 = [[10,10], [2,2], [1,1], [.2,.2]]
probs2 = [[4,12], [2,6], [1,3], [.25,.75]]
probs3 = [[12,4], [6,2], [3,1], [.75,.25]]

colors = ['g', 'b', 'k', 'm']

for ax, probs in zip(axes, [probs1, probs2, probs3]):
    for prob, color in zip(probs, colors):
        a, b = prob
        x = np.linspace(0.01, 0.99, 100)
        y = beta.pdf(x, a, b)
        ax.plot(x, y, c=color, lw=2, label=r'$\beta({},{})$'.format(a,b))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    ax.set_ylim([-.5, 4.5])
plt.suptitle('beta distribution')
plt.show()