from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt

def myHolo():

    optics = hp.core.Optics(wavelen=.632, index=1.33, polarization=[1.0, 0.0])

    #magnification = 40
    #Spacing = 6.8 / magnification
    Spacing = 0.1

    obj = hp.load('cgh.png', spacing=Spacing, optics=optics)
    #ref = hp.load('image0146.tif', spacing=Spacing, optics=optics)

    #holo = obj - ref
    holo = obj

    hp.show(holo)
    plt.show()

    rec = hp.propagate(holo, np.linspace(100, 150, 50))

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
