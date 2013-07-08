from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import autojit

@autojit
def main():

    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')

    img=obj-ref

    temp=np.empty(img.shape)+0j

    wavelength=405e-9
    k=2*np.pi/(wavelength)

    ''' L is the distance from the source to the screen '''
    L=13e-3

    #distX=6e-6
    #distY=6e-6

    n=float(img.shape[0])
    m=float(img.shape[1])

    first=time.time()

    a=np.mgrid[0:n,0:m][0]
    b=np.mgrid[0:n,0:m][1]

    r=np.sqrt(L*L+a*a+b*b)
    Xprime=(a*L)/r
    Yprime=(b*L)/r
    Rprime=(L*L)/r
    xx=(Xprime*L)/Rprime
    yy=(Yprime*L)/Rprime
    xx=xx.astype(int)
    yy=yy.astype(int)

    ''' z is the slice we want to look at '''
    z=250e-6
    #z=13e-3-250e-6
    #z=13e-3

    print(z)

    temp[xx,yy]=img[xx,yy]*(L/Rprime)**4*np.exp((1j*k*z*Rprime)/L)

    print('temp')

    K=np.fft.fft2(temp)

    print('fft')

    print(time.time()-first)

    Kint=K.real*K.real+K.imag+K.imag
    print('Kint')

    #plt.imshow(np.log(Kint+1),cmap=plt.cm.Greys_r)
    plt.imshow(Kint,cmap=plt.cm.Greys_r)
    plt.show()

main()

