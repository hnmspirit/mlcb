from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fanimation
np.random.seed(21)

# data
N = 1000
X = np.random.rand(N, 1)
y = 4 + 3*X + .2*np.random.randn(N, 1)
one = np.ones((N, 1))
X1 = np.hstack((one, X))


def func(w): return .5/N * np.linalg.norm(y - X1.dot(w), 2)**2


def grad(w): return 1./N * X1.T.dot(X1.dot(w) - y)


def grad_numeric(w, eps=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy()
        wn = w.copy()
        wp[i] += eps
        wn[i] -= eps
        g[i] = (func(wp) - func(wn))/(2*eps)
    return g


def check_grad(w, eps=1e-4):
    w = np.random.rand(*w.shape)
    g1 = grad(w)
    g2 = grad_numeric(w)
    norm = np.linalg.norm(g1 - g2)
    return True if norm < eps else False


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
    print('\tf = {:.2f}*x + {:.2f}'.format(w[1][0], w[0][0]))
    return w, ws


# check
w0 = np.random.rand(2, 1)
print('Checking gradient ... {}'.format(check_grad(w0)))

# gd
w0 = np.array([[2], [1]])
w, ws = gd1(w0, 0.1)
w, ws = gd1(w0, 2)
w, ws = gd1(w0, 1)
ws = np.array(ws).reshape(-1, 2)


# ***********************************
# animation 2d
def update(i):
    plt.cla()
    w0, w1 = ws[i, 0], ws[i, 1]
    x1 = np.array([[0], [1]])
    y1 = x1*w1 + w0
    title = 'iter %4d/%4d: f = %.2f*x + %.2f' % (i+1, len(ws), w1, w0)

    plt.plot(X, y, 'c.', alpha=0.5)
    plt.plot(x1, y1, 'm')
    plt.grid(alpha=0.3), plt.axis([-.25, 1.25, 2, 8]), plt.title(title)


# visualize
plt.figure(figsize=(4, 4))
for i in range(0, len(ws), 5):
    update(i)
    plt.pause(0.1)
plt.close()
# savegif
# fig2 = plt.figure(figsize=(8,6))
# anim = fanimation(fig2, update, np.arange(0, len(ws), 5), interval=100)
# anim.save('gradient_linreg.gif', dpi=200, writer='pillow')


# ***********************************
# animation 3d
a1 = np.linalg.norm(y, 2)**2/N
b1 = 2*np.sum(X)/N
c1 = np.linalg.norm(X, 2)**2/N
d1 = -2*np.sum(y)/N
e1 = -2*X.T.dot(y)/N

xm = np.arange(1.5, 6.0, 0.025)
ym = np.arange(0.5, 4.5, 0.025)
xm, ym = np.meshgrid(xm, ym)
Z = a1 + xm**2 + b1*xm*ym + c1*ym**2 + d1*xm + e1*ym


def update_2d(i):
    w0, w1 = ws[i, 0], ws[i, 1]
    title = 'iter %4d/%4d: f = %.2f*x + %.2f' % (i+1, len(ws), w1, w0)
    if i == 0:
        plt.cla()
        CS = plt.contour(xm, ym, Z, 100)
        manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
        plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
    else:
        wp0, wp1 = ws[i-1:i+1, 0], ws[i-1:i+1, 1]
        plt.plot(wp0, wp1, 'r-')
    plt.plot(w0, w1, 'r.')
    plt.axis([1.5, 6, 0.5, 4.5]), plt.title(title)


# visualize
plt.figure(figsize=(4, 4))
for i in range(0, len(ws), 1):
    update_2d(i)
    plt.pause(0.1)
plt.show()
