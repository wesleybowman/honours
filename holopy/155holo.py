from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt

''' my holographic setup, which I think works since we use plane waves '''

optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

magnification = 60
Spacing = 6.8 / magnification

obj = hp.load('image0100.tif', spacing=Spacing, optics=optics)
ref = hp.load('../image0156.tif', spacing=Spacing, optics=optics)

holo = obj - ref

rec = hp.propagate(holo, np.linspace(300, 400, 20))

#hp.show(holo)
#plt.show()

hp.show(rec)
plt.show()

recInt = abs(rec) * abs(rec)

from mayavi import mlab
mlab.contour3d(recInt)
#mlab.pipeline.volume(mlab.pipeline.scalar_field(recInt))
mlab.axes(x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
mlab.outline()
mlab.show()
