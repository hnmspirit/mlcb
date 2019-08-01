from __future__ import division, print_function, unicode_literals
import numpy as np
from numpy import sin, cos, abs
import matplotlib.pyplot as plt
np.random.seed(21)


def func(x): return x**2 + 10*sin(x)


def grad(x): return 2*x + 10*cos(x)


def gd1(x=None, lr=0.1, eps=1e-3):
    if x == None:
        x = np.random.randn()
    xs = [x]
    for it in range(10000):
        gradient = grad(x)
        if abs(gradient) < eps:
            break
        x = x - lr*gradient
        xs.append(x)
    print('gd has stoped after %d iter' % (it+1))
    return xs, it


def gd_momentum(x=None, lr=0.1, gamma=0.9, eps=1e-2):
    if x == None:
        x = np.random.randn()
    xs = [x]
    v = np.zeros_like(x)
    for it in range(10000):
        gradient = grad(x)
        if abs(gradient) < eps:
            break
        v = gamma * v + lr*gradient
        x = x - v
        xs.append(x)
    print('gd momentum has stoped after %d iter' % (it+1))
    return xs, it


def gd_nag(x=None, lr=0.1, gamma=0.9, eps=1e-2):
    if x == None:
        x = np.random.randn()
    xs = [x]
    v = np.zeros_like(x)
    for it in range(10000):
        gradient = grad(x - gamma*v)
        if abs(gradient) < eps:
            break
        v = gamma*v + lr*gradient
        x = x - v
        xs.append(x)
    print('gd nag has stoped after %d iter' % (it+1))
    return xs, it


x0 = 5

x, _ = gd1(x=x0, lr=0.1)
x1 = np.array(x)
y1 = func(x1)

x, _ = gd_momentum(x=x0, lr=0.09, gamma=0.9)
x2 = np.array(x)
y2 = func(x2) - 1

x, _ = gd_nag(x=x0, lr=0.1, gamma=0.9)
x3 = np.array(x)
y3 = func(x3) + 1

# visualize
x0 = np.linspace(-6, 6, 100)
y0 = func(x0)

n_steps = max(len(x1), len(x2), len(x3))
for i in range(n_steps+1):
    plt.clf()
    plt.plot(x0, y0, 'y', alpha=0.1, linewidth=15)
    title = 'iter %d/%d' % (i+1, n_steps+1)
    if i >= len(x1):
        title += ' (x1 stopped)'
        plt.plot(x1[-1], y1[-1], 'r.')
    else:
        plt.plot(x1[i-1:i+1], y1[i-1:i+1], 'r.-')

    if i >= len(x2):
        title += ' (x2 stopped)'
        plt.plot(x2[-1], y2[-1], 'g.')
    else:
        plt.plot(x2[i-1:i+1], y2[i-1:i+1], 'g.-')

    if i >= len(x3):
        title += ' (x3 stopped)'
        plt.plot(x3[-1], y3[-1], 'b.')
    else:
        plt.plot(x3[i-1:i+1], y3[i-1:i+1], 'b.-')
    plt.grid(alpha=0.3), plt.title(title),
    plt.axis([-7, 7, -15, 45]), plt.pause(0.1)
plt.show()
