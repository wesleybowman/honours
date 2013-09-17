from __future__ import division, print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt
import tables as tb

def loadHolo():
    optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

    magnification = 20
    Spacing = 6.8 / magnification

    obj = hp.load('P1000018.tiff', spacing=Spacing, optics=optics)
    ref = hp.load('P1000019.tiff', spacing=Spacing, optics=optics)

    holo = obj - ref

    return holo

holo = loadHolo()

slices = np.arange(100,10100,100)

f = tb.openFile('P1819.hdf','w')
matrix = f.createCArray(f.root, 'complexHologram',tb.ComplexAtom(itemsize=8), shape=(4016,3016,100))

count = 0
for i in slices:

    print(i)
    rec = hp.propagate(holo, i)
    matrix[..., count] = rec
    count += 1

