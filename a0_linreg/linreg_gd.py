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


def gradient_descent(w_init, lr=0.1, epochs=1000, eps=1e-3):
    w = w_init.copy()
    ws = [w[:,0]]
    for epoch in range(epochs):
        gw = grad(w)
        if np.linalg.norm(gw) / gw.size < eps:
            break
        w = w - lr * gw
        ws.append(w[:,0])
    print('lr: {:.2f}, epoch {:4d}, w: {:.3e}  {:.3e}'.format(lr, epoch+1, w[0, 0], w[1, 0]))
    return w, np.array(ws)

w, ws = gradient_descent(w0, 0.1)
w, ws = gradient_descent(w0, 2)
w, ws = gradient_descent(w0, 1)


# *****************
# anim 2d
def init_anim2d():
    line.set_data([], [])

def anim2d(i):
    w0, w1 = ws[i, 0], ws[i, 1]
    xv = np.array([0, 1])
    yv = w0 + w1 * xv
    line.set_data(xv, yv)
    ax.set_title('it %#d/%3d, w= %.1f: %.1f' % (i+1, len(ws), w0, w1))

fig, ax = plt.subplots(figsize=(4, 4))
ax.axis([-.25, 1.25, 3.5, 7.5])
ax.plot(X, y, 'c.', alpha=0.5)
line, = ax.plot([], [], 'm')
ax.grid(alpha=0.5)

frames = range(0, len(ws), 3)
anim = fanim(fig, anim2d, frames, init_anim2d, interval=100, repeat_delay=1000)
plt.pause(2)
# anim.save('linreg_gd.gif', dpi=200, writer='pillow')



# *****************
# anim 3d
k1 = (y**2).sum()/N
k2 = -2*y.sum()/N
k3 = -2*(X*y).sum()/N
k4 = 2*X.sum()/N
k5 = (X**2).sum()/N

xmin, xmax, ymin, ymax = 0, 6, -1, 5
W0 = np.arange(xmin, xmax, 0.025)
W1 = np.arange(ymin, ymax, 0.025)
W0, W1 = np.meshgrid(W0, W1)
L = k1 + W0**2 + k2*W0 + k3*W1 + k4*W0*W1 + k5*W1**2

def init_anim3d():
    point.set_data([], [])
    line.set_data([], [])

def anim3d(i):
    w0, w1 = ws[i, 0], ws[i, 1]
    wp0, wp1 = ws[:i+1, 0], ws[:i+1, 1]
    point.set_data(w0, w1)
    line.set_data(wp0, wp1)
    plt.title('it %3d/%3d, w= %.1f: %.1f' % (i+1, len(ws), w0, w1))

fig, ax = plt.subplots(figsize=(5, 4))
ax.axis([xmin, xmax, ymin, ymax])
CS = ax.contour(W0, W1, L, 100)
manual_locations = [(5.4, 2.2), (4.6, 2.4)]
ax.clabel(CS, inline=.1, fontsize=9, manual=manual_locations)
point, = ax.plot([], [], 'r.')
line, = ax.plot([], [], 'r-', alpha=0.5)
ax.grid(alpha=0.5)

frames = range(0, len(ws))
anim = fanim(fig, anim3d, frames, init_anim3d, interval=100, repeat_delay=1000)
plt.show()