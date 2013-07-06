from numpy import linspace
import holopy as hp
from holopy.core import Optics
from holopy.propagation import propagate
from holopy.core.tests.common import get_example_data
from holopy.core import load
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

lamb=0.632
ind=1
#one='withObject1.png'
img='holoTest.png'
#two='withoutObject1.png'
pixels=6.8

optics = hp.core.Optics(lamb, ind,
                        polarization=[1.0, 0.0])

holo = hp.load(img, spacing = pixels,  optics = optics)
rec_vol = propagate(holo, linspace(0.01, 1, 7))

a=rec_vol.real
xSize=rec_vol.shape[0]
ySize=rec_vol.shape[1]
zSize=rec_vol.shape[2]

x=[i for i in xrange(xSize)]
y=[i for i in xrange(ySize)]


def slices(b):
    z=[]
    for i in x:
        for j in y:
            z.append(a[i][j][b])

    z=np.array(z)
    z=np.reshape(z,(ySize,xSize))

    return z


for i in xrange(zSize):
    print i
    z=slices(i)
    xx,yy=np.meshgrid(x,y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx,yy,z,cmap=plt.cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.view_init(elev=90.,azim=0)

    plt.show()
