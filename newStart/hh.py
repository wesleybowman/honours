from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import autojit
from scipy.interpolate import RectBivariateSpline as rbs
from multiprocessing import Pool
import itertools


def func(smallX,smallY):

    print(smallX,smallY)

    temp2=temp[xx,yy]*np.exp((1j*k*(smallX*Xprime+smallY*Yprime))/L)
    temp3=rbs(i,j,temp2.real)
    K[smallX,smallY]=temp3.integral(0,kx,0,ky)

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return func(*a_b)

#global temp,temp2,temp3,xx,yy,Xprime,Yprime,k
#@autojit
#def main():

obj=plt.imread('jerichoObject.bmp')
ref=plt.imread('jerichoRef.bmp')

img=obj-ref

temp=np.empty(img.shape)+0j
K=np.empty(img.shape)+0j

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

kx=K.shape[0]
ky=K.shape[1]
i=np.arange(0,kx)
j=np.arange(0,ky)

pool=Pool()
pool.map(func_star,itertools.product(i,j))

#smallX=np.mgrid[0:K.shape[0],0:K.shape[1]][0]
#smallY=np.mgrid[0:K.shape[0],0:K.shape[1]][1]

#    for smallX in xrange(kx):
#        for smallY in xrange(ky):
#
#            print(smallX,smallY)
#
#            temp2=temp[xx,yy]*np.exp((1j*k*(smallX*Xprime+smallY*Yprime))/L)
#            temp3=rbs(i,j,temp2.real)
#            K[smallX,smallY]=temp3.integral(0,kx,0,ky)

print(K)
K.dump('K.dat')
#K=np.fft.fft2(temp)

#print('fft')

print(time.time()-first)

#Kint=K.real*K.real+K.imag+K.imag
#print('Kint')

#plt.imshow(np.log(Kint+1),cmap=plt.cm.Greys_r)
#plt.imshow(Kint,cmap=plt.cm.Greys_r)
#plt.show()




