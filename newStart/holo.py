from __future__ import division,print_function
import holopy as hp
import matplotlib.pyplot as plt

''' setting up optics '''
optics = hp.core.Optics(wavelen=.405, index=1.33,
                        polarization=[1.0, 0.0])

''' loading the images '''
obj = hp.core.load('jerichoObject.bmp',spacing=6,optics=optics)
ref = hp.core.load('jerichoRef.bmp',spacing=6,optics=optics)

''' contrast image '''
holo=obj-ref

'''reconstruction '''
#rec = hp.propagate(holo, 13e-3)
#rec = hp.propagate(holo, 13e-3-250e-6)
rec = hp.propagate(holo, 250e-6)

''' intensity so pyplot can plot it '''
recInt=rec.real*rec.real+rec.imag*rec.imag

''' hp.show doesn't show me anything,unless plt.show() is also present '''
#hp.show(holo)
#hp.show(recInt)

''' how I have to view it '''
plt.imshow(recInt)
plt.show()
