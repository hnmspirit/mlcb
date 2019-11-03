import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
from itertools import zip_longest
import torch

def func(w):
    x, y = w
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def train(loss, opt, x_init, epochs, eps=1e-3):
    x = torch.tensor(x_init, requires_grad=True)
    opt_ = opt([x])

    paths = [x_init]
    for epoch in range(epochs):
        x_check = x.clone().detach()
        loss_ = loss(x)
        loss_.backward()
        opt_.step()
        paths.append(x.clone().detach().numpy())
        if torch.norm(x.data - x_check) < eps:
            break
        opt_.zero_grad()
    return np.array(paths), epoch


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


xmin, xmax, xstep = -4.5, 4.5, .045
ymin, ymax, ystep = -4.5, 4.5, .045
x, y = np.meshgrid(np.arange(xmin, xmax+xstep, xstep), np.arange(ymin, ymax+ystep, ystep))
z = func((x, y))
zeps = 1.1e-0 # shift to show on log scale
z += zeps

minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)
x_init = np.array([0.9, 1.6])
x_init_ = x_init.reshape(-1, 1)


methods = dict(
    vanilla  = lambda x: torch.optim.SGD(x, lr=0.01),
    momentum = lambda x: torch.optim.SGD(x, lr=0.005, momentum=0.9, nesterov=False),
    nesterov = lambda x: torch.optim.SGD(x, lr=0.005, momentum=0.9, nesterov=True),
    adagrad  = lambda x: torch.optim.Adagrad(x, lr=1),
    rmsprop  = lambda x: torch.optim.RMSprop(x, lr=0.05),
    adam     = lambda x: torch.optim.Adam(x, lr=0.2)
)

labels = []
paths = []
for label, opt in methods.items():
    path, epoch = train(func, opt, x_init, epochs=2000)
    idx = list(range(50)) + list(range(50,len(path),5))
    labels.append(label)
    paths.append(path[idx].T)
    print('{:9}:{:6} epoch ({:4})'.format(label, epoch, len(idx)))

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
anim.save('opt2d.mp4', fps=fps, writer='ffmpeg', codec='h264')

plt.show()