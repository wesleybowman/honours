import numpy as np
import matplotlib.pyplot as plt
import time


#t1=time.time()
#
#img=plt.imread('output.png')
#img=np.mean(img,2)
#tau=img*img
#tau=img
#
#E=np.zeros((10,10))
#
#E=E+0j
#img=img+0j
#
#pi=np.pi
#z=2
#wavelength=632e-9
#
#dx=0.0254/600
#dy=0.0254/600
#
#N,M=tau.shape
#
#
#for (n,m),value in np.ndenumerate(E):
#    
#    print n,m
#    
#    for k in range(N):
#        for l in range(M):
#        
#            img[k,l]=tau[k,l]*np.exp((-1j*pi)/(wavelength*z)*
#            (k*k*dx*dx+l*l*dy*dy))*np.exp(1j*2*pi*((k*n)/N+(l*m)/M))
#        
#    E[n,m]=np.sum(img)
#
#oldImg=img

#image=plt.imread('successfulDotTest.png')
#image=np.mean(image,2)
#
#
#tau=image
#tau=tau+0j

tau=hologram

pi=np.pi
z=2
wavelength=632e-9

#dx=0.0254/600
#dy=0.0254/600

dx=0.0254/600
dy=0.0254/600

N,M=tau.shape

newimg=np.zeros((tau.shape))
newimg=newimg+0j

for (k,l),value2 in np.ndenumerate(tau):
    
    newimg[k,l]=tau[k,l]*np.exp((-1j*pi)/(wavelength*z)*
    ((k*k*dx*dx)+(l*l*dy*dy)))

Newhologram=np.fft.ifft2(newimg)

plt.imshow(Newhologram.real,cmap=plt.cm.Greys_r)
plt.colorbar()
plt.show()

#
#tau2=hologram
#h=np.array([])
#h=np.zeros((tau2.shape))
#h=h+0j
#
#
#for (x,y),value2 in np.ndenumerate(tau2):
#
#    h[x,y]=1/(1j*wavelength*z)*np.exp(2*pi*1j*(z/wavelength))*np.exp(1j*pi*(x*x+y*y)/(wavelength*z))
#
#f=np.fft.fft2(tau2)
#h=np.fft.fft2(h)
#g=f*h
#ng=np.fft.ifft2(g)
#nng=np.fft.fft2(g)
#
#plt.imshow(nng.real,cmap=plt.cm.Greys_r)
#plt.colorbar()
#plt.show()
#
#
#
#
