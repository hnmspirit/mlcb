import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

probs = [
    [0, 1],
    [.5, .5],
    [-1, 2]
]

colors = ['b', 'r', 'g']

for prob, color in zip(probs, colors):
    mean, std = prob
    x = np.linspace(mean-3*std, mean+3*std, 200)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y, color)
plt.plot([-8, 8], [0, 0], 'k',)

plt.text(-5.25, .35, r'$\mu=0, \sigma=1$', fontsize=15, color='b')
plt.text(1.25, .5, r'$\mu=0.5, \sigma=0.5$', fontsize=15, color='r')
plt.text(-8.25, .15, r'$\mu=-1, \sigma=2$', fontsize=15, color='g')

plt.grid(alpha=.2)
plt.title('normal distribution')
plt.show()