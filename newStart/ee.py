from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RectBivariateSpline as rbs
from numba import autojit

@autojit
def main():
    '''Using numba to try and make this as fast as possible, still very slow. '''

    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')

    img=obj-ref

    K=np.empty(img.shape)+0j
    temp=np.empty(img.shape)+0j

    wavelength=405e-9
    k=2*np.pi/(wavelength)
    z=250e-6
    #z=13e-3-250e-6

    distX=6e-6
    distY=6e-6

    n=float(img.shape[0])
    m=float(img.shape[1])

    a = np.arange(0,n)
    b = np.arange(0,m)

    first=time.time()

    for i in xrange(K.shape[0]):
        for j in xrange(K.shape[1]):

            print(i,j)
            '''create an r vector '''
            r=(i*distX,j*distY,z)

            for x in xrange(img.shape[0]):
                for y in xrange(img.shape[1]):
                    '''create an ksi vector, then calculate
                       it's norm, and the dot product of r and ksi'''
                    ksi=(x*distX,y*distY,z)
                    ksiNorm=np.linalg.norm(ksi)
                    ksiDotR=float(np.dot(ksi,r))

                    '''calculate the integrand'''
                    temp[x,y]=img[x,y]*np.exp(1j*k*ksiDotR/ksiNorm)

            '''interpolate so that we can do the integral and take the integral'''
            temp2=rbs(a,b,temp.real)
            K[i,j]=temp2.integral(0,n,0,m)

            timeTook=time.time()-first

            print(timeTook)


    K.dump('K.dat')

    kInt=K.real*K.real+K.imag*K.imag

    plt.imshow(kInt,cmap=plt.cm.Greys_r)

main()
