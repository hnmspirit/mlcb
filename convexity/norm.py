import numpy as np
from numpy import abs, log, exp, maximum, sin, sqrt
import matplotlib.pyplot as plt

fig = plt.figure()

x = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
x, y = np.meshgrid(x, y)
levels = np.array([0, 1])

labels = ['1/8', '1/4', '1/2', '2/3', '4/5',
          '1', '4/3', '2', '4', 'inf']
ps = [1/8, 1/4, 1/2, 2/3, 4/5, 1, 4/3, 2, 4, 100]

for j, p in enumerate(ps):
    plt.subplot(2, 5, j+1)
    z = (abs(x)**p + abs(y)**p)**(1/p)

    plt.arrow(0, -1.1, 0, 2.2, width=0.001,
              head_width=0.05, overhang=1, alpha=0.8)
    plt.arrow(-1.1, 0, 2.2, 0, width=0.001,
              head_width=0.05, overhang=1, alpha=0.8)

    plt.contourf(x, y, z, levels=levels, colors='gray', alpha=0.5)
    plt.contour(x, y, z, levels=levels, colors='b')

    print('j = {}'.format(j))
    plt.title('p = {}'.format(labels[j]), fontsize=10)
    plt.axis('square')
    plt.axis('off')

plt.suptitle(r'$(|x|^p + |y|^p)^{1/p} \leq 1$', ha='center', y=0.94)
plt.show()
