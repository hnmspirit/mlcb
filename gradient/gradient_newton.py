from __future__ import division, print_function, unicode_literals
import numpy as np
from numpy import sin, cos, abs
import matplotlib.pyplot as plt
np.random.seed(21)


def func(x): return x**2 + 10*sin(x)


def grad(x): return 2*x + 10*cos(x)


def gradgrad(x): return 2 - 10*sin(x)


def gd_newton(x=None, eps=1e-3):
    if x == None:
        x = np.random.randn()
    xs = [x]
    for it in range(100):
        g1 = grad(x)
        g2 = gradgrad(x)
        if abs(g1) < eps or abs(g2) < eps:
            break
        x = x - g1/g2
        xs.append(x)
    print('gd newton has stoped after %d iter at %f' % (it+1, func(x)))
    return xs, it


x2, _ = gd_newton(-5)
x2 = np.array(x2)
y2 = func(x2)

# visualize
x0 = np.linspace(-6, 6, 100)
y0 = func(x0)

for i in range(len(x2)):
    if i == 0:
        plt.cla()
        plt.plot(x0, y0, 'y', alpha=0.7)
    else:
        xp, yp = x2[i-1:i+1], y2[i-1:i+1]
        plt.plot(xp, yp, 'r.-')
    plt.plot(x2[i], y2[i], 'ro')
    plt.grid(alpha=0.5), plt.axis([-12, 12, -12, 50])
    plt.pause(0.1)
plt.show()
