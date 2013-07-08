from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import autojit
from math import sqrt

def X(Xprime,Yprime,Rprime,L):
    return (Xprime*L)/Rprime

def Y(Xprime,Yprime,Rprime,L):
    return (Yprime*L)/Rprime

@autojit
def main():

    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')

    img=obj-ref

    K=np.empty(img.shape)+0j
    temp=np.empty(img.shape)+0j
    #tempDot=np.empty(img.shape)+0j
    #tempNorm=np.empty(img.shape)+0j

    wavelength=405e-9
    k=2*np.pi/(wavelength)
    L=250e-6
    #z=13e-3
    #z=13e-3-250e-6

    distX=6e-6
    distY=6e-6

    n=float(img.shape[0])
    m=float(img.shape[1])

    for x in xrange(img.shape[0]):
        for y in xrange(img.shape[1]):
            print(x,y)
            r=sqrt(L*L+x*x+y*y)
            Xprime=(x*L)/r
            Yprime=(y*L)/r
            Rprime=(L*L)/r
            xx=X(Xprime,Yprime,Rprime,L)
            yy=Y(Xprime,Yprime,Rprime,L)

            return img[xx,yy]


main()

