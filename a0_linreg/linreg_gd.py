import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fanim
np.random.seed(21)


N = 200
X = np.random.rand(N, 1)
y = 4 + 3*X + 0.2*np.random.randn(N, 1)

X1 = np.hstack((np.ones((N, 1)), X))
w0 = np.random.rand(2, 1)
print(w0[:,0])

def func(w): return .5/N * np.linalg.norm(y - X1.dot(w), 2)**2
def grad(w): return 1./N * X1.T.dot(X1.dot(w) - y)


def grad_numeric(w):
    eps = 1e-6
    g = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy()
        wn = w.copy()
        wp[i] += eps
        wn[i] -= eps
        g[i] = (func(wp) - func(wn)) / (2*eps)
    return g


g0 = grad(w0)
g1 = grad_numeric(w0)
diff = np.linalg.norm(g0 - g1)
print('Check diff gradient: ', diff)


def gradient_descent(w_init, lr=0.1, epochs=1000, eps=1e-4):
    w = w_init.copy()
    ws = [w[:,0]]
    for epoch in range(epochs):
        gw = grad(w)
        if np.linalg.norm(gw) / gw.size < eps:
            break
        w = w - lr * gw
        ws.append(w[:,0])
    print('epoch %4d, w: %.3e  %.3e' % (epoch+1, w[0, 0], w[1, 0]))
    return w, np.array(ws)


w, ws = gradient_descent(w0, 0.1)
w, ws = gradient_descent(w0, 2)
w, ws = gradient_descent(w0, 1)


# *****************
# anim 2d

def update(i):
    plt.cla()
    w0, w1 = ws[i, 0], ws[i, 1]
    xv = np.array([0, 1])
    yv = w0 + w1 * xv

    plt.plot(X, y, 'c.', alpha=0.5)
    plt.plot(xv, yv, 'm')
    plt.grid(alpha=0.5)
    plt.title('it %#d/%3d, w= %.2f: %.2f' % (i+1, len(ws), w0, w1))


plt.figure(figsize=(4, 4))
for i in range(0, len(ws), 3):
    update(i)
    # print(ws[i, :])
    plt.pause(0.1)
plt.close()

# fig2 = plt.figure(figsize=(4,4))
# anim = fanim(fig2, update, range(0, len(ws), 3), interval=100)
# anim.save('linreg_gd.gif', dpi=200, writer='pillow')


# *****************
# anim 3d

k1 = (y**2).sum()/N
k2 = -2*y.sum()/N
k3 = -2*(X*y).sum()/N
k4 = 2*X.sum()/N
k5 = (X**2).sum()/N

bmin = ws.min(axis=0) - 1
bmax = ws.max(axis=0) + 1
W0 = np.arange(bmin[0], bmax[0], 0.025)
W1 = np.arange(bmin[1], bmax[1], 0.025)
W0, W1 = np.meshgrid(W0, W1)
L = k1 + W0**2 + k2*W0 + k3*W1 + k4*W0*W1 + k5*W1**2


def update3d(i):
    w0, w1 = ws[i, 0], ws[i, 1]
    if i == 0:
        plt.cla()
        CS = plt.contour(W0, W1, L, 100)
        manual_locations = [(5.4, 2.2), (4.6, 2.4)]
        plt.clabel(CS, inline=.1, fontsize=9, manual=manual_locations)
    else:
        wp0, wp1 = ws[i-1:i+1, 0], ws[i-1:i+1, 1]
        plt.plot(wp0, wp1, 'r-', alpha=0.5)
    plt.plot(w0, w1, 'r.')
    plt.grid(alpha=0.5)
    plt.title('it %3d/%3d, w= %.2f: %.2f' % (i+1, len(ws), w0, w1))


plt.figure(figsize=(4, 4))
for i in range(0, len(ws), 1):
    update3d(i)
    # print(ws[i, :])
    plt.pause(0.1)
# plt.show()
