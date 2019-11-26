import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
np.random.seed(21)

K = 3
N = 50
means = [[1,1], [6,2], [3,4]]
cov = [[.9, 0], [0, .9]]
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.vstack([X0, X1, X2])
y = np.array([0]*N + [1]*N + [2]*N)


class KMeans(object):
    def __init__(self, K):
        self.K = K

    def init_c(self):
        ids = np.random.choice(self.X.shape[0], self.K, replace=False)
        return self.X[ids]

    def update_x(self, centers):
        D = cdist(self.X, centers, metric='euclidean')
        return np.argmin(D, axis=1)

    def update_c(self, y):
        centers = np.array([self.X[y==k].mean(axis=0) for k in range(self.K)])
        return centers

    def fit(self, X):
        self.X = X
        list_centers = []
        list_y = []

        centers = self.init_c()
        for it in range(100):
            y = self.update_x(centers)
            list_centers.append(centers)
            list_y.append(y)

            centers_new = self.update_c(y)
            diff = np.linalg.norm(centers_new - centers)
            print('it {}: diff = {:.4f}'.format(it, diff))
            if diff < 0.001:
                break
            centers = centers_new
        return list_centers, list_y, it


KMC = KMeans(K=3)
list_centers, list_y, it = KMC.fit(X)

# sampling centers moving
list_centers1 = []
nsample = 5
for i in range(len(list_y)-1):
    list_centers1.extend(np.linspace(list_centers[i], list_centers[i+1], nsample, endpoint=True))
list_centers1.append(list_centers[-1])

# fig
fig = plt.figure(figsize=(6,6), tight_layout=True)
ax = plt.gca()

ct1, = ax.plot([], [], 'r*', mec='k', mew=2, ms=12)
ct2, = ax.plot([], [], 'b*', mec='k', mew=2, ms=12)
ct3, = ax.plot([], [], 'g*', mec='k', mew=2, ms=12)

pt1, = ax.plot([], [], 'r.', alpha=0.3)
pt2, = ax.plot([], [], 'b.', alpha=0.3)
pt3, = ax.plot([], [], 'g.', alpha=0.3)

txt = ax.set_title('')


ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-2, 7])
ax.set_xlim([-3, 9])
ax.grid(alpha=0.1)


def init_anim():
    ct1.set_data([], [])
    ct2.set_data([], [])
    ct3.set_data([], [])

    pt1.set_data([], [])
    pt2.set_data([], [])
    pt3.set_data([], [])

    txt.set_text('')


def update_anim(i):
    cs = list_centers1[i]
    ct1.set_data(cs[0,0], cs[0,1])
    ct2.set_data(cs[1,0], cs[1,1])
    ct3.set_data(cs[2,0], cs[2,1])

    y = list_y[i//nsample]
    pt1.set_data(X[y==0][:,0], X[y==0][:,1])
    pt2.set_data(X[y==1][:,0], X[y==1][:,1])
    pt3.set_data(X[y==2][:,0], X[y==2][:,1])

    txt.set_text('iter: {}'.format(i//nsample + 1))

anim = FuncAnimation(fig, update_anim, len(list_centers1), init_anim, interval=100, repeat_delay=2000)
anim.save('kmeans.gif', writer='pillow')
plt.show()