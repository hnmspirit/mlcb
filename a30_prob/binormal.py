from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-2, 2, 0.025)
Y = np.arange(-2, 2, 0.025)
X, Y = np.meshgrid(X, Y)
Z = np.exp(-(X**2 + Y**2 - .5*X*Y))*1.5

ax.plot_surface(X, Y, Z, alpha=1, cmap='jet')
cset = ax.contour(X, Y, Z, zdir='z', offset=-1, cmap='jet')
ax.set_zlim(-1, 1)

ax.set_title('bivariate normal distribution', fontsize=10)
plt.show()
