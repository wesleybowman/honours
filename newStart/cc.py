from __future__ import division,print_function
import matplotlib.pyplot as plt
#import numexpr as ne
#import time

import scipy.fftpack as fftpack
import numpy as np

obj=plt.imread('jerichoObject.bmp')
print(obj.shape)
ref=plt.imread('jerichoRef.bmp')
print(ref.shape)
img=obj-ref

#start=200e-6
#stop=13e-3
#stepSize=1e-6
#distance=np.arange(start,stop,stepSize)

def ifft(a, overwrite=False, shift=True):
    if shift:
        res = fftpack.ifft2(fftpack.fftshift(a, axes=[0,1]), axes=[0, 1],
                            overwrite_x=overwrite)
    else:
        res = fftpack.ifft2(a, overwrite_x=overwrite)

    return res

holoInt=img
raw_input('correct?')

distX=6e-6
distY=6e-6

wavelength=405e-9
k=2*np.pi/wavelength

L=13e-3-250e-6

temp=np.empty(holoInt.shape)*0j

for (k,l),value in np.ndenumerate(holoInt):
    print(k,l)
    temp[k,l]=holoInt[k,l]*np.exp(-1j*np.pi*(k*k*distX*distX+l*l*distY*distY)/(wavelength*L))

rec=ifft(temp)

recInt=rec.real*rec.real+rec.imag*rec.imag

plt.imshow(recInt,cmap=plt.cm.Greys_r)
plt.show()

#    plt.ion()
#    fig=plt.figure()
#    ax=fig.gca()
#    for i,value in enumerate(distance):
#        print i, value
##        name='figure%d.png'%(i,)
##        plt.savefig(name,bbox_inches=0)
#        plt.clf()
#
#        recInt,rec=reconstruction(holoInt,wavelength=405e-9,z=value,px=distX,py=distY,show=0)
#
#        plt.imshow(recInt,cmap=plt.cm.Greys_r)
#        fig.canvas.draw()
#        time.sleep(1e-3)
#
