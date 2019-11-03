import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
from itertools import zip_longest
from optim import *

def func(w):
    x, y = w
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def grad(w):
    x, y = w
    g = np.zeros(2)
    k1 = 2*(1.5 - x + x*y)
    k2 = 2*(2.25 - x + x*y**2)
    k3 = 2*(2.625 - x + x*y**3)

    g[0] = (y-1)*k1 + (y**2-1)*k2 + (y**3-1)*k3
    g[1] = x*k1 + 2*x*y*k2 + 3*x*y**2*k3
    return g



class OptAnim(animation.FuncAnimation):
    def __init__(self, paths, labels, fig, ax, frames=None,
                 interval=60, repeat_delay=None, blit=True, **kwargs):

        if frames is None:
            frames = max(path.shape[1] for path in paths)
        self.paths = paths
        self.lines = [ax.plot([], [], label=label, lw=2, alpha=.7)[0]
                      for _, label in zip_longest(paths, labels)]
        self.points = [ax.plot([], [], '.', color=line.get_color(), alpha=.7)[0]
                       for line in self.lines]

        super(OptAnim, self).__init__(fig, self.animate, frames, self.init_anim,
            interval=interval, blit=blit, repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])
        return self.lines + self.points

    def animate(self, i):
        for line, point, path in zip(self.lines, self.points, self.paths):
            line.set_data(*path[:, :i])
            point.set_data(*path[:, i-1:i])
        return self.lines + self.points


minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)
x_init = np.array([0.9, 1.6])
x_init_ = x_init.reshape(-1, 1)


# paths
epochs = 2000
x1, it1 = vanilla(x_init, grad, lr=0.01, epochs=epochs)
x2, it2 = momentum(x_init, grad, lr=0.005, epochs=epochs)
x3, it3 = nesterov(x_init, grad, lr=0.005, epochs=epochs)
x4, it4 = adagrad(x_init, grad, lr=1, epochs=epochs)
x5, it5 = rmsprop(x_init, grad, lr=0.05, epochs=epochs)
x6, it6 = adam(x_init, grad, lr=0.1, epochs=epochs)

labels = ['vanilla', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam']
paths = [x1,x2,x3,x4,x5,x6]
idxs = [list(range(50)) + list(range(50,len(path),5)) for path in paths]

for label, path, idx in zip(labels, paths, idxs):
    print('{:9}:{:6} epoch ({:4})'.format(label, len(path), len(idx)))

paths = [path[idx].T for path, idx in zip(paths, idxs)]


# figure
xmin, xmax, xstep = -4.5, 4.5, .045
ymin, ymax, ystep = -4.5, 4.5, .045
x, y = np.meshgrid(np.arange(xmin, xmax+xstep, xstep), np.arange(ymin, ymax+ystep, ystep))
z = func((x, y))
zeps = 1.1e-0 # shift to show on log scale
z += zeps

logzmax = np.log(z.max() - z.min() + zeps)
levels = np.logspace(0, logzmax//2, 35)
fps = 10

fig = plt.figure(figsize=(9, 6), tight_layout=True)
ax = plt.gca()

cm = ax.contour(x, y, z, levels=levels, norm=LogNorm(), cmap='jet', alpha=.5)
ax.plot(*x_init_, 'k+', markersize=10, alpha=0.5)
ax.plot(*minima_, 'k*', markersize=10, alpha=0.5)

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# for label, path in zip(labels, paths):
#     ax.plot(*path, ':', label=label, lw=2, alpha=.7)

anim = OptAnim(paths, labels, fig, ax, interval=1000//fps)
ax.legend()

print('save anim ...')
# anim.save('opt2d.gif', fps=fps, writer='pillow')
# anim.save('opt2d2.mp4', fps=fps, writer='ffmpeg', codec='h264')

plt.show()