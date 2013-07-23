from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
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

distX=6e-6
distY=6e-6

n,m=img.shape

#''' tested to make sure the vectorized code was the same as the for loops '''
#tempXprime=np.empty(img.shape)
#tempYprime=np.empty(img.shape)
#tempR=np.empty(img.shape)
#tempxx=np.empty(img.shape)
#tempyy=np.empty(img.shape)
#tempRprime=np.empty(img.shape)
#
#for x in xrange(n):
#    for y in xrange(m):
#        r=np.sqrt(L*L+x*x+y*y)
#        tempR[x,y]=r
#        tempRprime[x,y]=(L*L)/r
#        tempXprime[x,y]=(x*L)/r
#        tempYprime[x,y]=(y*L)/r
#        tempxx[x,y]=(tempXprime[x,y]*L)/tempRprime[x,y]
#        tempyy[x,y]=(tempYprime[x,y]*L)/tempRprime[x,y]
#
#tempxx=tempxx.astype(int)
#tempyy=tempyy.astype(int)

a,b=np.mgrid[0:n,0:m]

r=ne.evaluate('sqrt(L*L + a*a + b*b)')
Xprime=ne.evaluate('(a*L) / r')
Yprime=ne.evaluate('(b*L) / r')
Rprime=ne.evaluate('(L*L) / r')

''' xx is the inverse transform of Xprime, refered to as X in the paper '''
xx=ne.evaluate('(Xprime*L) / Rprime')
yy=ne.evaluate('(Yprime*L) / Rprime')
xx=xx.astype(int)
yy=yy.astype(int)

#test=(tempxx==xx).all() #True

''' z is the slice we want to look at '''
z=250e-6
z=13e-3-250e-6
#z=13e-3

print('Distance: {0}'.format(z))

#test=(img[xx,yy]==img).all() #This is False!
#print(test)

imgPrime=ne.evaluate('img * (L / Rprime)**4 * exp((1j * k * z * Rprime) / L)')

newimg=img[xx,yy]
NewimgPrime=ne.evaluate('newimg * (L / Rprime)**4 * exp((1j * k * z * Rprime) / L)')

#test=(imgPrime==NewimgPrime).all() #False
#bad=np.where(imgPrime!=NewimgPrime)
#imgPrime[bad[0],bad[1]]
#NewimgPrime[bad[0],bad[1]]

# Not exactly the same, but close enough that it doesn't change the appearance
# of the output image

K=np.fft.ifft2(imgPrime)
#K=np.fft.ifft2(NewimgPrime)

Kint=abs(K) * abs(K)

plot(Kint)
logPlot(Kint)
