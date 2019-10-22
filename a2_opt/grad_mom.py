import numpy as np
import matplotlib.pyplot as plt
np.random.seed(21)

def f(x): return x**2 + 10*np.sin(x)
def g(x): return 2*x + 10*np.cos(x)


def gd_vanilla(x=None, lr=0.1, epochs=100, eps=1e-3):
    if not x:
        x = np.random.randn()
    xs = [x]
    for it in range(epochs):
        x_check = x
        gx = g(x)
        x -= lr * gx
        if np.abs(x - x_check) < eps:
            break
        xs.append(x)
    print('gd vanilla stopped after %d iter' % (it+1))
    return np.array(xs), it


def gd_momentum(x=None, lr=0.1, gamma=0.9, epochs=100, eps=1e-3):
    if not x:
        x = np.random.randn()
    v = np.zeros_like(x)
    xs = [x]
    for it in range(epochs):
        x_check = x
        gx = g(x)
        v = gamma*v + lr*gx
        x -= v
        if np.abs(x - x_check) < eps:
            break
        xs.append(x)
    print('gd momentum stopped after %d iter' % (it+1))
    return np.array(xs), it


def gd_nesterov(x=None, lr=0.1, gamma=0.9, epochs=100, eps=1e-3):
    if not x:
        x = np.random.randn()
    v = np.zeros_like(x)
    xs = [x]
    for it in range(epochs):
        x_check = x
        gx = g(x - gamma*v)
        v = gamma*v + lr*gx
        x -= v
        if np.abs(x - x_check) < eps:
            break
        xs.append(x)
    print('gd momentum stopped after %d iter' % (it+1))
    return np.array(xs), it


x0 = 5
x1, _ = gd_vanilla(x0, 0.1, epochs=200)
y1 = f(x1)

x2, _ = gd_momentum(x0, 0.1, epochs=200)
y2 = f(x2) - 1 # shift for visualize

x3, _ = gd_nesterov(x0, 0.1, epochs=200)
y3 = f(x3) + 1 # shift for visualize

xv = np.linspace(-6, 6, 90)
yv = f(xv)


steps = max(len(x1), len(x2), len(x3))
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(xv, yv, 'y', alpha=0.2, linewidth=15)

l1, = ax.plot([], [], 'r', label='vanilla')
p1, = ax.plot([], [], 'r.')

l2, = ax.plot([], [], 'g', label='momentum')
p2, = ax.plot([], [], 'g.')

l3, = ax.plot([], [], 'b', label='nesterov')
p3, = ax.plot([], [], 'b.')

plt.grid(alpha=0.2)
plt.axis([-7, 7, -15, 45])
plt.legend()

def anim(i):
    ax.set_title('epoch: %d' % i)

    t1 = min(i, len(x1)-1)
    l1.set_data(x1[t1-1:t1+1], y1[t1-1:t1+1])
    p1.set_data(x1[t1], y1[t1])

    t2 = min(i, len(x2)-1)
    l2.set_data(x2[t2-1:t2+1], y2[t2-1:t2+1])
    p2.set_data(x2[t2], y2[t2])

    t3 = min(i, len(x3)-1)
    l3.set_data(x3[t3-1:t3+1], y3[t3-1:t3+1])
    p3.set_data(x3[t3], y3[t3])


for i in range(steps):
    anim(i)
    plt.pause(0.05)
# plt.show()
