from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import LinearNDInterpolator as interpol
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.integrate import dblquad

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

#n,m= img.shape
n= float(img.shape[0])
m= float(img.shape[1])

#a,b = np.mgrid[0:n,0:m]
a=np.arange(0,n)
b=np.arange(0,m)

#pts=np.array((a.ravel(),b.ravel())).T

first=time.time()

total=n*m

for (i,j),k in np.ndenumerate(K):

    perc=i*j
    print(perc/total)
    r=(i*distX,j*distY,z)

    for (x,y),value in np.ndenumerate(img):
        ksi=(x*distX,y*distY,z)
        ksiNorm=np.linalg.norm(ksi)
        ksiDotR=np.dot(ksi,r)

        temp[x,y]=img[x,y]*np.exp(1j*k*ksiDotR/ksiNorm)

    temp.dump('temp.dat')
    #tempRavel=temp.ravel()
    #surf=interpol(pts,tempRavel)
    #func=lambda y,x: surf([[x,y]])

    #K[i,j]=dblquad(func,0.0,m,lambda x:0.0, lambda x:n)[0]
    temp2=rbs(a,b,temp.real)
    K[i,j]=temp2.integral(0,n,0,m)

timeTook=time.time()-first

print(timeTook)

K.dump('K.dat')
kInt=K.real*K.real+K.imag*K.imag

plt.imshow(kInt,cmap=plt.cm.Greys_r)
plt.imsave(kInt,cmap=plt.cm.Greys_r)


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
#img[x,y]*np.exp(1j*k*ksiDotR/ksiNorm)
#

