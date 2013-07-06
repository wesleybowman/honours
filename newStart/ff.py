from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import LinearNDInterpolator as interpol
from scipy.integrate import dblquad

def main():
    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')

    img=obj-ref

    K=np.empty(img.shape)+0j
    temp=np.empty(img.shape)+0j

    wavelength=405e-9
    k=2*np.pi/(wavelength)
    z=250e-6
    #z=13e-3
    #z=13e-3-250e-6

    distX=6e-6
    distY=6e-6

#    n,m= img.shape

    n=img.shape[0]
    m=img.shape[1]
#    a,b = np.mgrid[0:n,0:m]
    a = np.mgrid[0:n,0:m][0]
    b = np.mgrid[0:n,0:m][1]

    pts=np.array((a.ravel(),b.ravel())).T

    first=time.time()

#    for (i,j),k in np.ndenumerate(K):
    for i in xrange(K.shape[0]):
        for j in xrange(K.shape[1]):

            print(i,j)
            r=(i*distX,j*distY,z)

#            for (x,y),value in np.ndenumerate(img):
            for x in xrange(img.shape[0]):
                for y in xrange(img.shape[1]):

                    ksi=(x*distX,y*distY,z)
                    ksiNorm=np.linalg.norm(ksi)
                    ksiDotR=np.dot(ksi,r)

                    temp[x,y]=img[x,y]*np.exp(1j*k*ksiDotR/ksiNorm)

            tempRavel=temp.ravel()
            surf=interpol(pts,tempRavel)
            func=lambda y,x: surf([[x,y]])

            K[i,j]=dblquad(func,0,m,lambda x:0, lambda x:n  )

    timeTook=time.time()-first

    print(timeTook)

main()
#main_numba = autojit(main)
#main_numba()

#a=np.arange(0,n)
#b=np.arange(0,m)
#
#ksi=(a*distX,b*distY,z)
#n,m=img.shape
#x,y=np.mgrid[0:n,0:m]
#i,j=np.mgrid[0:n,0:m]
#r=(i*distX,j*distY,z)
#print(ksi)
#ksiNorm=np.linalg.norm(ksi)
#ksiDotR=np.dot(ksi,r)
#
#img[x,y]*np.exp(1j*k*ksiDotR/ksiNorm)
#

