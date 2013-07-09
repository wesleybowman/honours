from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RectBivariateSpline as rbs
from multiprocessing import Pool
import itertools

def func(smallX,smallY):
    ''' Function used to calculate the integral '''
    print(smallX,smallY)
    temp2=temp[xx,yy]*np.exp((1j*k*(smallX*Xprime+smallY*Yprime))/L)
    temp3=rbs(i,j,temp2.real)
    K[smallX,smallY]=temp3.integral(0,kx,0,ky)

def func_star(a_b):
    ''' Convert `f([1,2])` to `f(1,2)` call, so that pool can do multiple
        arguments. '''
    return func(*a_b)

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

n=img.shape[0]
m=img.shape[1]

first=time.time()

a,b=np.mgrid[0:n,0:m]

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

print('Distance: {0}'.format(z))

temp[xx,yy]=img[xx,yy]*(L/Rprime)**4*np.exp((1j*k*z*Rprime)/L)

kx=K.shape[0]
ky=K.shape[1]
i=np.arange(0,kx)
j=np.arange(0,ky)

pool=Pool()
pool.map(func_star,itertools.product(i,j))

print(K)
K.dump('K.dat')

print(time.time()-first)

#Kint=K.real*K.real+K.imag+K.imag
#print('Kint')

#plt.imshow(np.log(Kint+1),cmap=plt.cm.Greys_r)
#plt.imshow(Kint,cmap=plt.cm.Greys_r)
#plt.show()
