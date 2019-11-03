import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fanim
np.random.seed(21)


def func(w):
    x, y = w
    return (x**2 + y - 7)**2 + (x-y+1)**2

def grad(w):
    x, y = w
    g = np.zeros(2)
    g[0] = 4*(x**2 + y - 7)*x + 2*(x-y+1)
    g[1] = 2*(x**2 + y - 7) - 2*(x-y+1)
    return g


def gd_vanilla(w_init, lr=0.1, epochs=100, eps=1e-3):
    w = w_init.copy()
    ws = [w]
    for it in range(epochs):
        w_check = w
        gw = grad(w)
        w = w - lr * gw
        if np.linalg.norm(w - w_check) / w.size < eps:
            break
        ws.append(w)
    print('gd vanilla stopped after %d iter' % (it+1))
    return np.array(ws), it


def gd_momentum(w_init, lr=0.1, gamma=0.9, epochs=100, eps=1e-3):
    w = w_init.copy()
    v = np.zeros_like(w)
    ws = [w]
    for it in range(epochs):
        w_check = w
        gw = grad(w)
        v = gamma*v + lr*gw
        w = w - v
        if np.linalg.norm(w - w_check) / w.size < eps:
            break
        ws.append(w)
    print('gd momentum stopped after %d iter' % (it+1))
    return np.array(ws), it


def gd_nesterov(w_init, lr=0.1, gamma=0.9, epochs=100, eps=1e-3):
    w = w_init.copy()
    v = np.zeros_like(w)
    ws = [w]
    for it in range(epochs):
        w_check = w
        gw = grad(w - gamma*v)
        v = gamma*v + lr*gw
        w = w - v
        if np.linalg.norm(w - w_check) / w.size < eps:
            break
        ws.append(w)
    print('gd momentum stopped after %d iter' % (it+1))
    return np.array(ws), it


w0 = np.array([-1, 10])
w1, _ = gd_vanilla(w0, 0.005, epochs=150)
w2, _ = gd_momentum(w0, 0.005, epochs=150)
w3, _ = gd_nesterov(w0, 0.005, epochs=150)

x1, y1 = w1.T
x2, y2 = w2.T
x3, y3 = w3.T

xv = np.arange(-6, 5, 0.025)
yv = np.arange(-20, 15, 0.025)
xv, yv = np.meshgrid(xv, yv)
zv = func((xv, yv))


fig, ax = plt.subplots(figsize=(9,6))
levels = np.concatenate((np.arange(0.1, 50, 5), np.arange(50, 150, 20)))
CS = ax.contour(xv, yv, zv, levels=levels, alpha=.4)

l1, = ax.plot([], [], 'r', lw=2, alpha=.5, label='vanilla')
p1, = ax.plot([], [], 'r.')
l2, = ax.plot([], [], 'g', lw=2, alpha=.5, label='momentum')
p2, = ax.plot([], [], 'g.')
l3, = ax.plot([], [], 'b', lw=2, alpha=.5, label='nesterov')
p3, = ax.plot([], [], 'b.')

ax.grid(alpha=.3)
ax.legend()

def init_anim():
    l1.set_data([], [])
    p1.set_data([], [])

    l2.set_data([], [])
    p2.set_data([], [])

    l3.set_data([], [])
    p3.set_data([], [])

def anim(i):
    ax.set_title('epoch: %d/%d' % (i+1, frames))

    t1 = min(i, len(x1)-1)
    l1.set_data(x1[:t1], y1[:t1])
    p1.set_data(x1[t1] , y1[t1])

    t2 = min(i, len(x2)-1)
    l2.set_data(x2[:t2], y2[:t2])
    p2.set_data(x2[t2] , y2[t2])

    t3 = min(i, len(x3)-1)
    l3.set_data(x3[:t3], y3[:t3])
    p3.set_data(x3[t3] , y3[t3])

frames = max(len(x1), len(x2), len(x3))
anim = fanim(fig, anim, frames, init_anim, interval=100, repeat_delay=1000)
plt.show()