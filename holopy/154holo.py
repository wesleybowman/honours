from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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


def TestsobHypot(rec):

    x=ndimage.sobel(abs(rec)**2,axis=0, mode='constant')
    y=ndimage.sobel(abs(rec)**2,axis=1, mode='constant')
    hype = np.hypot(x,y)
    hype = hype.mean(axis=0).mean(axis=0)

    index = hype.argmax()
    return index

optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

magnification = 60
Spacing = 6.8 / magnification

obj = hp.load('image0100.tif', spacing=Spacing, optics=optics)
#obj = hp.load('image0001.tif', spacing=Spacing, optics=optics)
#obj = hp.load('image0020.tif', spacing=Spacing, optics=optics)
ref = hp.load('../image0156.tif', spacing=Spacing, optics=optics)

holo = obj - ref

rec = hp.propagate(holo, np.linspace(475, 505, 20))

#hp.show(holo)
#plt.show()

hp.show(rec)
plt.show()

a, b, c = rec.shape

hype = np.ones((a,b,c))


index = sobHypot(rec)
print(index)
index = TestsobHypot(rec)
print(index)


for i in xrange(c):
    x=ndimage.sobel(abs(rec[...,i])**2,axis=0, mode='constant')
    y=ndimage.sobel(abs(rec[...,i])**2,axis=1, mode='constant')
    hype[...,i] = np.hypot(x,y)
    #hype[...,i] = hype[...,i].mean()

#print('Max')
#print(hype.max())
#print(hype.argmax())

index=hype.argmax()


#index = np.where(hype==hype.max())
#print(index)

#print('All')
#print(hype[...,:])
for i in xrange(c):
    print(hype[...,i].mean())
    plt.imshow(hype[...,i])
    plt.show()


#recInt = abs(rec) * abs(rec)

#from mayavi import mlab
#mlab.contour3d(recInt)
##mlab.pipeline.volume(mlab.pipeline.scalar_field(recInt))
#mlab.axes(x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
#mlab.outline()
#mlab.show()
