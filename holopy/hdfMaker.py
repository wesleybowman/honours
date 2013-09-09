from __future__ import division,print_function
import holopy as hp
import numpy as np
import tables as tb
#import matplotlib.pyplot as plt

optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

magnification = 60
Spacing = 6.8 / magnification

ref = hp.load('../image0156.tif', spacing=Spacing, optics=optics)

f = tb.openFile('154.hdf','w')

matrix = f.createCArray(f.root, 'complexHologram',tb.ComplexAtom(itemsize=8), shape=(1024,1024,100))

for i in xrange(1,101):

    image = 'image0{0:03}.tif'.format(i)
    print(image)

    obj = hp.load(image, spacing=Spacing, optics=optics)

    holo = obj - ref

    rec = hp.propagate(holo, np.linspace(400, 500, 20))

    matrix[..., i-1] = rec[..., 15]

f.close()
