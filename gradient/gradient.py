from __future__ import division, print_function, unicode_literals
import numpy as np
from numpy import sin, cos, abs
import matplotlib.pyplot as plt
np.random.seed(21)


def func(x): return x**2 + 5*sin(x)


def grad(x): return 2*x + 5*cos(x)


def gd1(x=None, lr=0.1, eps=1e-4):
    if x == None:
        x = np.random.randn()
    xs = [x]
    for it in range(100):
        gradient = grad(x)
        if abs(gradient) < eps:
            break
        x = x - lr*gradient
        xs.append(x)
    print('gd has stoped after %d iter' % (it+1))
    return xs, it


x1, _ = gd1(-5, .1)
x1 = np.array(x1)
y1 = func(x1)

x2, _ = gd1(5, .1)
x2 = np.array(x2)
y2 = func(x2)

# visualize
x0 = np.linspace(-6, 6, 100)
y0 = func(x0)

plt.plot(x0, y0, 'c', alpha=0.6)
plt.plot(x1, y1, 'r.', markersize=10)
plt.plot(x2, y2, 'b.', markersize=10)
plt.grid(0.5), plt.show()
