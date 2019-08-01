from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fanimation
np.random.seed(21)


def func(w):
    x, y = w
    return (x**2 + y - 7)**2 + (x-y+1)**2


def grad(w):
    x, y = w
    g = np.zeros_like(w)
    g[0] = 4*(x**2 + y - 7)*x + 2*(x-y+1)
    g[1] = 2*(x**2 + y - 7) - 2*(x-y+1)
    return g


def gd1(w_init, alpha=0.1, eps=1e-3, n_epochs=100):
    w = w_init
    ws = [w]
    for it in range(n_epochs):
        gradient = grad(w)
        if np.linalg.norm(gradient)/len(w) < eps:
            break
        w = w - alpha*gradient
        ws.append(w)
    print('gd has stoped after %d iters' % (it+1))
    return w, ws


def gd_momentum(w_init, alpha=0.1, gamma=0.9, eps=1e-2, n_epochs=100):
    w = w_init
    v = np.zeros_like(w)
    ws = [w]
    for it in range(n_epochs):
        gradient = grad(w)
        if np.linalg.norm(gradient)/len(w) < eps:
            break
        v = gamma * v + alpha*gradient
        w = w - v
        ws.append(w)
    print('gd momentum has stoped after %d iters' % (it+1))
    return w, ws


# data
x = np.arange(-6, 5, 0.025)
y = np.arange(-20, 15, 0.025)
x, y = np.meshgrid(x, y)
z = func((x, y))

# gd
w0 = np.array([[0], [-3]])
w, ws = gd_momentum(w0, 0.005)
ws = np.array(ws).reshape(-1, 2)

# visualize
levels = np.concatenate((np.arange(0.1, 50, 5), np.arange(50, 150, 20)))


def update(i):
    w0, w1 = ws[i, 0], ws[i, 1]
    title = 'iter %4d/%4d: point=(%.1f,%.1f)' % (i+1, len(ws), w0, w1)
    if i == 0:
        plt.cla()
        CS = plt.contour(x, y, z, levels=levels)
    else:
        wp0, wp1 = ws[i-2:i+1, 0], ws[i-2:i+1, 1]
        plt.plot(wp0, wp1, 'r.-')
    plt.plot(w0, w1, 'r.')
    plt.grid(0.1), plt.title(title)


for i in range(0, len(ws), 2):
    update(i)
    plt.pause(0.1)
plt.show()
