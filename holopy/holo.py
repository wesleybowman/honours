from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt

def myHolo():
    ''' my holographic setup, which I think works since we use plane waves '''

    optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

    magnification = 40
    Spacing = 6.8 / magnification

    obj = hp.load('image0139.tif', spacing=Spacing, optics=optics)
    ref = hp.load('image0146.tif', spacing=Spacing, optics=optics)

    holo = obj - ref

    hp.show(holo)
    plt.show()

    rec = hp.propagate(holo, np.linspace(1, 500, 50))

    return rec

rec = myHolo()

#recInt = abs(rec) * abs(rec)

#print('Amplitude')
#hp.show(recInt)
hp.show(rec)
plt.show()

#print('Imaginary')
#hp.show(rec.imag)
#plt.show()
#
#print('Phase')
#phase=np.arctan(rec.imag/rec.real)
##phase=np.unwrap(phase)
#hp.show(phase)
##hp.show(np.angle(rec))
#plt.show()

''' 3D plotting - useful? '''
#from mayavi import mlab
#mlab.contour3d(recInt)
##    mlab.pipeline.volume(mlab.pipeline.scalar_field(newImg))
#mlab.axes(x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
#mlab.outline()
#mlab.show()
