import numpy as np
import matplotlib.pyplot as plt

def plot(ax, X,y, title):
    ax.plot(X[y==0,0], X[y==0,1], 'bo')
    ax.plot(X[y==1,0], X[y==1,1], 'ro')
    ax.set(xlim=[-1,2], ylim=[-1,2])
    ax.axis(option='equal')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def fill(ax, x, y):
    ax.plot(x, y, 'k')
    ax.fill_betweenx(y, x, -1, color='r', alpha=.1)
    ax.fill_betweenx(y, x,  2, color='b', alpha=.1)

def fill_xor(ax, x1, x2, y):
    if (x2 - x1 < 0).all():
        x1, x2 = x2, x1
    [ax.plot(x_, y, 'k') for x_ in [x1, x2]]
    ax.fill_betweenx(y, x1, x2, color='r', alpha=.1)
    ax.fill_betweenx(y, x1, -1, color='b', alpha=.1)
    ax.fill_betweenx(y, x2,  2, color='b', alpha=.1)


X0 = np.array([[0,0], [1,0], [0,1], [1,1]])
X1 = np.array([[0,.5], [1,.5]])
yv = np.array([-1,2])

fig, ax = plt.subplots(2,2, figsize=(6,6), dpi=100)

# AND: 2x+2y-3=0
fill(ax[0,0], 1.5-yv, yv)
plot(ax[0,0], X0, np.array([0,0,0,1]), 'AND')

# OR: 2x+2y-1=0
fill(ax[0,1], .5-yv, yv)
plot(ax[0,1], X0, np.array([0,1,1,1]), 'OR')

# NOT: 2x-1=0
fill(ax[1,0], [.5,.5], yv)
plot(ax[1,0], X1, np.array([0,1]), 'NOT')

# XOR
fill_xor(ax[1,1], 0.5 - yv, 1.5 - yv, yv)
plot(ax[1,1], X0, np.array([1,0,0,1]), 'XOR')

fig.savefig('xor.jpg', dpi=100)
plt.show()
