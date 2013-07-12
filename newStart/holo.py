from __future__ import division,print_function
import holopy as hp
import matplotlib.pyplot as plt

''' setting up optics '''
optics = hp.core.Optics(wavelen=.405, index=1.33,
                        polarization=[1.0, 0.0])

holo = hp.core.load('jerichoObject.bmp',spacing=6,optics=optics)

'''reconstruction? '''
''' For now doing individual slices. One of these 3 should work. Is this in
    meters? '''
#rec = hp.propagate(holo, 13e-3)
#rec = hp.propagate(holo, 13e-3-250e-6)
rec = hp.propagate(holo, 250e-6)

''' intensity so pyplot can plot it '''
recInt=rec.real*rec.real+rec.imag*rec.imag

''' hp.show doesn't show me anything '''
#hp.show(holo)
hp.show(recInt)

''' how I have to view it '''
#plt.imshow(recInt)
plt.show()
