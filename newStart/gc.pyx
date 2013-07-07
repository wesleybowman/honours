from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RectBivariateSpline as rbs
from numba import autojit
from math import sqrt
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def main():

    cdef np.ndarray[double, ndim=2] obj
    cdef np.ndarray[double, ndim=2] ref
    cdef np.ndarray[double, ndim=2] img
    cdef np.ndarray[double, ndim=2] K

    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')

    img=obj-ref

    K=np.empty(img.shape)+0j
#    temp=np.empty(img.shape)+0j
#    tempDot=np.empty(img.shape)+0j
#    tempNorm=np.empty(img.shape)+0j

    cdef double wavlength,k,z,distX,distY
    wavelength=405e-9
    k=2*np.pi/(wavelength)
    z=250e-6
    #z=13e-3
    #z=13e-3-250e-6

    distX=6e-6
    distY=6e-6

    cdef np.ndarray[double, ndim=2] n
    cdef np.ndarray[double, ndim=2] m

    n=float(img.shape[0])
    m=float(img.shape[1])

#    a = np.arange(0,n)
#    b = np.arange(0,m)
#
#    xx=np.mgrid[0:n,0:m][0]
#    yy=np.mgrid[0:n,0:m][1]
#    ii=np.mgrid[0:n,0:m][0]
#    jj=np.mgrid[0:n,0:m][1]
#
#    pts=np.array((a.ravel(),b.ravel())).T

    first=time.time()

    cdef unsigned int i,j,x,y
    cdef double ksi,ksiDotR,ksiNorm

    for i in xrange(K.shape[0]):
        for j in xrange(K.shape[1]):
            print(i,j)
            r=(i*distX,j*distY,z)

            for x in xrange(img.shape[0]):
                for y in xrange(img.shape[1]):
                    ksi=(x*distX,y*distY,z)
                    ksiDotR=ksi[0]*r[0]+ksi[1]*r[1]+ksi[2]*r[2]
                    ksiNorm=sqrt(ksi[0]*ksi[0]+ksi[1]*ksi[1]+ksi[2]*ksi[2])

            timeTook=time.time()-first
            print(timeTook)

main()
