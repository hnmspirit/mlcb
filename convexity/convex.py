import numpy as np
from numpy import abs, log, maximum
from numpy import concatenate as cat, arange
import matplotlib.pyplot as plt


def fig1():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)
    z = abs(x) + abs(y)

    levels = arange(0, 1, 0.15)
    plt.contour(x, y, z, levels=levels, cmap='jet')

    txt = r'$|x| + |y|$'
    return txt


def fig2():
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    x, y = np.meshgrid(x, y)
    z = x**2 + y**2

    levels = cat((arange(0, 0.1, 0.05), arange(0.1, 1, 0.15)))
    plt.contour(x, y, z, levels=levels, cmap='jet')

    txt = r'$x^2 + y^2$'
    return txt


def fig3():
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    x, y = np.meshgrid(x, y)
    z = maximum(2*x**2 + y**2 - x*y, abs(x) + 2*abs(y))

    levels = cat((arange(0, 4, 1), arange(4, 16, 2)))
    plt.contour(x, y, z, levels=levels, cmap='jet')

    txt = r'$max(2x^2 + y^2 - xy, |x| + 2|y|)$'
    return txt


def fig4():
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    x, y = np.meshgrid(x, y)
    z = x + y

    levels = arange(-2, 2, 0.4)
    plt.contour(x, y, z, levels=levels, cmap='jet')

    txt = r'$x + y$'
    return txt


def fig5():
    x = np.linspace(0.001, 2, 50)
    y = np.linspace(0.001, 2, 50)
    x, y = np.meshgrid(x, y)
    z = x*log(x) + y*log(y)

    levels = cat((arange(-1, 0.2, 0.1), arange(0.2, 1, 0.2)))
    plt.contour(x, y, z, levels=levels, cmap='jet')

    txt = r'$xlog(x) + ylog(y)$'
    return txt


def fig6():
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    x, y = np.meshgrid(x, y)
    z = x**2 - y**2

    levels = arange(-1.5, 1.5, 0.4)
    plt.contour(x, y, z, levels=levels, cmap='jet')

    txt = r'$x^2 - y^2$'
    return txt


for j, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6]):
    plt.subplot(2, 3, j+1)
    label = fig()
    plt.title(label, fontsize=10)
    plt.axis('square',)
    plt.xticks([])
    plt.yticks([])


plt.show()
