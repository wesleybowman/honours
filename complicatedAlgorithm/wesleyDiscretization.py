from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import numexpr as ne

def plot(img):
    plt.imshow(img,cmap=plt.cm.Greys_r)
    plt.show()

def logPlot(img):
    plt.imshow(np.log(img+1),cmap=plt.cm.Greys_r)
    plt.show()

obj=plt.imread('jerichoObject.bmp')
ref=plt.imread('jerichoRef.bmp')

img=obj-ref

temp=np.empty(img.shape)+0j
imgPrime=np.empty(img.shape)+0j
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

r=ne.evaluate('sqrt(L*L + a*a + b*b)')
Xprime=ne.evaluate('(a*L) / r')
Yprime=ne.evaluate('(b*L) / r')
Rprime=ne.evaluate('(L*L) / r')
xx=ne.evaluate('(Xprime*L) / Rprime')
yy=ne.evaluate('(Yprime*L) / Rprime')
xx=xx.astype(int)
yy=yy.astype(int)

#r=np.sqrt(L*L+a*a+b*b)
#Xprime=(a*L)/r
#Yprime=(b*L)/r
#Rprime=(L*L)/r
#xx=(Xprime*L)/Rprime
#yy=(Yprime*L)/Rprime
#xx=xx.astype(int)
#yy=yy.astype(int)

''' z is the slice we want to look at '''
z=250e-6
#z=13e-3-250e-6
#z=13e-3

print('Distance: {0}'.format(z))

imgPrime=ne.evaluate('img * (L / Rprime)**4 * exp((1j * k * z * Rprime) / L)')

K=np.fft.ifft2(imgPrime)

Kint=abs(K) * abs(K)

plot(Kint)
logPlot(Kint)
