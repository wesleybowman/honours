from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

# setting up optics
#optics = hp.core.Optics(wavelen=.405, index=1.33, polarization=[1.0, 0.0])

optics = hp.core.Optics(wavelen=.635, index=1.33, polarization=[1.0, 0.0])

# loading the images
#obj = hp.load('jerichoObject.bmp',spacing=6,optics=optics)
#ref = hp.load('jerichoRef.bmp',spacing=6,optics=optics)

obj = hp.load('fibre1.png',spacing=7.6,optics=optics)
ref = hp.load('refFibre1.png',spacing=7.6,optics=optics)

# contrast image
holo=obj-ref

# reconstruction, same image for all slices though
#rec = hp.propagate(holo, np.linspace(200,13e7,10))
rec = hp.propagate(holo, np.linspace(3.5e4,5.5e4,100))

# intensity so pyplot can plot it
recInt=abs(rec)*abs(rec)

hp.show(recInt)
plt.show()

''' 3D plotting - useful '''
#mlab.contour3d(recInt)
##    mlab.pipeline.volume(mlab.pipeline.scalar_field(newImg))
#mlab.axes(x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
#mlab.outline()
#mlab.show()
