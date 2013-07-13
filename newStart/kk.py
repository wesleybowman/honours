from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import numexpr as ne
#from numba import autojit



obj=plt.imread('jerichoObject.bmp')
ref=plt.imread('jerichoRef.bmp')

holo=obj-ref

temp=np.empty(holo.shape)+0j
K=np.empty(holo.shape)+0j
Kreal=np.empty(holo.shape)+0j
Kimag=np.empty(holo.shape)+0j

wavelength=405e-9
k=2*np.pi/(wavelength)

''' L is the distance from the source to the screen '''
L=13e-3
z=250e-6
#z=13e-3
#z=13e-3-250e-6

dx=6e-6
dy=6e-6

n,m=holo.shape

first=time.time()

k,l=np.mgrid[0:n,0:m]
k2=k*k
l2=l*l
dx2=dx*dx
dy2=dy*dy
pi=np.pi #here for numexpr to evaluate

img=ne.evaluate('ref*holo*exp((-1j*pi)/(wavelength*z)*(k2*dx2+l2*dy2))')

rec=np.fft.ifft2(img)
recInt=abs(rec)*abs(rec)

plt.imshow(recInt,cmap=plt.cm.Greys_r)
plt.show()
