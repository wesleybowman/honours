import numpy as np
import matplotlib.pyplot as plt

img=plt.imread('smallA.png')
img=plt.imread('object.png')
img=np.mean(img,2)

Mx,My=img.shape

400e-6/Mx
64.8e-3/Mx

d0x=400e-6/Mx#0.0254/600#0.162e-3
d0y=400e-6/My#0.0254/600#0.162e-3
dx=64.8e-3/Mx#1#e-6
dy=64.8e-3/My#1#e-6

holo=np.zeros(img.shape)+0j

hologram=np.ones(img.shape)+0j

u=np.zeros(img.shape)+0j

uPrime=np.zeros(img.shape)+0j

U=np.zeros(img.shape)+0j

Uhat=np.zeros(img.shape)+0j

wavelength=632.8e-9
z=80e-3
k=2*np.pi/wavelength
k=1/wavelength


n,m=holo.shape

mx=np.arange(0,n)
my=np.arange(0,m)
x=np.arange(0,Mx)
y=np.arange(0,My)

#x=x*dx
#y=y*dy

x,y=np.meshgrid(x,y)
mx,my=np.meshgrid(mx,my)

mx,my=np.mgrid[0:n,0:m]
x,y=np.mgrid[0:Mx,0:My]

xs=mx*d0x
ys=my*d0y
v=xs*xs+ys*ys

fx=xs/(wavelength*z)
fy=ys/(wavelength*z)

w=x*x+y*y

uPrime[x,y]=img[x,y]*np.exp(1j*k*(w/(2*z)))

test=uPrime[x,y]*np.exp(-1j*2*np.pi*(fx*x+fy*y))

U[mx,my]=np.exp(1j*k*(v/(2*z)))*test.sum()

UInt=U.real*U.real+U.imag*U.imag

plt.imshow((UInt.real),cmap=plt.cm.Greys_r)
plt.show()


UInt[mx,my]*np.exp(-1j*np.pi/(wavelength*z)*(mx*mx*d0x*d0x+my*my*d0y*d0y))

Uhat[mx,my]=UInt[mx,my]*np.exp(1j*k*v/(wavelength*z))

Uhat[mx,my]=UInt[mx,my]*np.exp(-1j*np.pi/(wavelength*z)*(mx*mx*d0x*d0x+my*my*d0y*d0y))
rec=np.fft.ifft2(np.fft.ifft2(Uhat))
#rec=(np.fft.ifft2(Uhat))

temp=Uhat[mx,my]*np.exp(1j*2*np.pi*(fx*x+fy*y))
u[x,y]=temp.sum()

uint=u.real*u.real+u.imag*u.imag
uint=temp.real*temp.real+temp.imag*temp.imag

#rec=np.abs(rec)*np.abs(rec)
#rec=rec.real*rec.real+rec.imag*rec.imag
plt.imshow((rec.real),cmap=plt.cm.Greys_r)
plt.show()
#
#
#holoInt=UInt
#pxUhat[mx,my]=UInt[mx,my]*np.exp(1j*k*v/(wavelength*z))=d0x
#py=d0y
#n,m=holoInt.shape
#
#dx=px
#dy=py
#
#g=np.zeros(holoInt.shape)+0j
#rec=np.zeros(holoInt.shape)+0j
##    tau=np.zeros(holoInt.shape)+0j
#
#
#for (x,y),value in np.ndenumerate(holoInt):
#    g[x,y]=np.exp(-1j*np.pi/(wavelength*z)*(x*x*dx*dx+y*y*dy*dy))
#    rec[x,y]=holoInt[x,y]*g[x,y]
#
#tau=np.fft.fft2(holoInt)
#g=np.fft.fft2(g)
#
#rec=tau*g
#
#recShift=np.fft.ifft2(rec)
##    recShift=rec
#rec=np.fft.ifft2(recShift)
#recInt=rec.real*rec.real+rec.imag*rec.imag
#
#plt.imshow(np.log(recInt),cmap=plt.cm.Greys_r)
#plt.show()



'''
for (mx,my),value in np.ndenumerate(holo):
    print mx,my
    xs=mx*d0x
    ys=my*d0y
    
    v=xs*xs+ys*ys
    
    fx=xs/(wavelength*z)
    fy=ys/(wavelength*z)
    
    for (x,y),value2 in np.ndenumerate(img):

        w=x*x+y*y
        uPrime[x,y]=img[x,y]*np.exp(1j*k*(w/(2*z)))
        
        u[x,y]=np.exp(1j*k*(v/(2*z)))*uPrime[x,y]*np.exp(-1j*2*np.pi*(fx*x+fy*y))
    
    hologram[mx,my]=u[x,y]

fft=np.fft.fftshift(np.fft.fft2(hologram))


plt.imshow(hologram.real,cmap=plt.cm.Greys_r)

'''



#C=1/np.sqrt(1-((xs*xs)+(ys*ys))/(z*z))
#x0=C*xs
#y0=C*ys
#
#r0=np.sqrt((z*z)+(x0*x0)+(y0*y0))
#
#fx=x0/(wavelength*r0)
#fy=y0/(wavelength*r0)
#
#D=np.exp(-1j*2*np.pi*(fx+fy))
#E=np.exp(1j*2*np.pi*(fx+fy))
#
#uPrime=img*np.exp(1j*k*r0)
#
#newU=np.fft.fft2(uPrime)
#
#u=np.zeros(uPrime.shape)
#u=u+0j
#
#
#for (x,y),value in np.ndenumerate(img):
#    u[x,y]=uPrime[x,y]*np.exp(-1j*2*np.pi*(fx*x+fy*y))
#    holo[x,y]=img[x,y]*np.exp((1j*np.pi)/(wavelength*z)*(x*x+y*y))
#    
#    
#hologram=np.fft.fftshift(np.fft.fft2(holo))
#
#fft=u*np.exp(1j*k*r0)
#
#fft2=np.fft.fftshift(np.fft.fft2(img))
#
#U=np.fft.fftshift(np.fft.fft2(fft))
#
#Uprime=U*np.exp(-1j*k*r0)
#ifft=Uprime*D
#
#reconstruction=np.fft.ifft2(ifft)

