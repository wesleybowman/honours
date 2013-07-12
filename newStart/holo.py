from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt

# setting up optics
optics = hp.core.Optics(wavelen=.405, index=1.33,
                        polarization=[1.0, 0.0])

# loading the images
obj = hp.load('jerichoObject.bmp',spacing=6,optics=optics)
ref = hp.load('jerichoRef.bmp',spacing=6,optics=optics)

# contrast image
holo=obj-ref

# reconstruction, same image for all slices though
#rec = hp.propagate(holo, 13e-3)
#rec = hp.propagate(holo, 13e-3-250e-6)
rec = hp.propagation.propagate(holo, np.linspace(200e-6,300e-6,10))

# intensity so pyplot can plot it
recInt=abs(rec)*abs(rec)

hp.show(recInt)
plt.show()
