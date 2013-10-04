from __future__ import division,print_function
import holopy as hp
import numpy as np
import tables as tb
from scipy import ndimage
#import matplotlib.pyplot as plt

def sobHypot(rec):
    a, b, c = rec.shape
    hype = np.ones((a,b,c))

    for i in xrange(c):
        x=ndimage.sobel(abs(rec[...,i])**2,axis=0, mode='constant')
        y=ndimage.sobel(abs(rec[...,i])**2,axis=1, mode='constant')
        hype[...,i] = np.hypot(x,y)
        hype[...,i] = hype[...,i].mean()

    index = hype.argmax()
    return index

def main():
    optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

    magnification = 60
    Spacing = 6.8 / magnification

    ref = hp.load('../image0156.tif', spacing=Spacing, optics=optics)

    f = tb.openFile('154testauto.hdf','w')

    matrix = f.createCArray(f.root, 'complexHologram',tb.ComplexAtom(itemsize=8), shape=(1024,1024,100))

    for i in xrange(1,101):

        image = 'image0{0:03}.tif'.format(i)
        print(image)

        obj = hp.load(image, spacing=Spacing, optics=optics)

        holo = obj - ref

        rec = hp.propagate(holo, np.linspace(475, 505, 20))

        index = sobHypot(rec)
        print(index)

        matrix[..., i-1] = rec[..., index]

    f.close()

if __name__ == '__main__':
    main()
