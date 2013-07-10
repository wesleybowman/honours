''' Trying to vectorize code to make it faster '''
from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as rbs
from multiprocessing import Pool
import itertools
import time

obj=plt.imread('jerichoObject.bmp')
ref=plt.imread('jerichoRef.bmp')

img=obj-ref

temp=np.empty(img.shape)+0j
K=np.empty(img.shape)+0j
Kreal=np.empty(img.shape)+0j
Kimag=np.empty(img.shape)+0j

wavelength=405e-9
k=2*np.pi/(wavelength)

''' L is the distance from the source to the screen '''
L=13e-3

#distX=6e-6
#distY=6e-6

n,m=img.shape

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

kx,ky=K.shape
i=np.arange(0,kx)
j=np.arange(0,ky)

smallX,smallY=a,b

temp2=temp[xx,yy]*np.exp((1j*k*(smallX*Xprime+smallY*Yprime))/L)
temp3=rbs(i,j,temp2.real)
Kreal[smallX,smallY]=temp3.integral(0,kx,0,ky)

temp4=rbs(i,j,temp2.imag)
Kimag[smallX,smallY]+=temp4.integral(0,kx,0,ky)
#
#''' Using multiprocessing.Pool to make this parallel. '''
#pool=Pool()
#pool.map(func_star,itertools.product(i,j))

print(Kreal.dtype)
print(Kimag.dtype)

Kreal=Kreal.astype(np.float64,copy=False)
Kimag=Kimag.astype(np.float64,copy=False)

print(Kreal.dtype)
print(Kimag.dtype)

K=Kreal*Kreal+Kimag*Kimag

print(K)
K.dump('K.dat')

print(time.time()-first)

plt.imshow(K,cmap=plt.cm.Greys_r)
plt.show()

