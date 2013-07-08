from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import autojit
from math import sqrt


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
    #L=250e-6
    L=13e-3
    #z=13e-3-250e-6

    distX=6e-6
    distY=6e-6

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

    print(img[xx,yy])

    temp[xx,yy]=img[xx,yy]*(L/Rprime)**4*np.exp((1j*k*z*Rprime)/L)
    print('temp')
    print(temp)

    print(time.time()-first)

#    for x in xrange(img.shape[0]):
#        for y in xrange(img.shape[1]):
#            print(x,y)
#            r=sqrt(L*L+x*x+y*y)
#            Xprime=(x*L)/r
#            Yprime=(y*L)/r
#            Rprime=(L*L)/r
#            xx=(Xprime*L)/Rprime
#            yy=(Yprime*L)/Rprime
#
#            print(img[xx,yy])
#
#    print(time.time()-first)

main()

