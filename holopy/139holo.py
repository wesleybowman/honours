from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt

''' my holographic setup, which I think works since we use plane waves '''

optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

magnification = 40
Spacing = 6.8 / magnification

obj = hp.load('image0139.tif', spacing=Spacing, optics=optics)
ref = hp.load('image0146.tif', spacing=Spacing, optics=optics)

holo = obj - ref

focus = hp.load('image0136.tif', spacing=Spacing, optics=optics)

hp.show(focus)
plt.show()

hp.show(holo)
plt.show()

rec = hp.propagate(holo, np.linspace(50, 200, 50))

hp.show(rec)
plt.show()

''' 3D plotting - useful? '''
#from mayavi import mlab
#mlab.contour3d(recInt)
##    mlab.pipeline.volume(mlab.pipeline.scalar_field(newImg))
#mlab.axes(x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
#mlab.outline()
#mlab.show()
