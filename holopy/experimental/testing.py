from __future__ import division, print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt


def loadHolo():
    optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

    magnification = 20
    Spacing = 6.8 / magnification

    obj = hp.load('P1000018.tiff', spacing=Spacing, optics=optics)
    ref = hp.load('P1000019.tiff', spacing=Spacing, optics=optics)

    holo = obj - ref

    return holo

holo = loadHolo()

#hp.show(holo)
#plt.show()

#holo = holo[1100:1700,1200:1800]
rec = hp.propagate(holo, np.linspace(1950, 2000, 5))
#rec = hp.propagate(holo, 16000)


hp.show(rec)
plt.show()

''' 3D plotting - useful? '''
#from mayavi import mlab
#mlab.contour3d(recInt)
##    mlab.pipeline.volume(mlab.pipeline.scalar_field(newImg))
#mlab.axes(x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
#mlab.outline()
#mlab.show()
