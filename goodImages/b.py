import numpy as np
import matplotlib.pyplot as plt

img=plt.imread('dot.png')
#img=plt.imread('blackA.jpg')

img=np.mean(img,2)

plt.imshow(img)
plt.colorbar()
plt.show()

N,M=img.shape

zz=np.linspace(1e-6,1,100)

z=2
wavelength=632e-9
k=2*np.pi/wavelength

g=np.zeros((N,M))
interference=np.zeros((N,M))
g=g+0j
interference=interference+0j



for (x,y),k in np.ndenumerate(img):
    if k<0.5:
        A=1

    else:
        A=0

    r=1
    E=A*np.exp(1j*k*r)
    interference[x][y]=A

E=interference

plt.imshow(E.imag,cmap=plt.cm.Greys_r)
plt.show()

for (x,y),value in np.ndenumerate(g):
    g[x][y]=np.exp(1j*k*z)/(1j*wavelength*z)*np.exp(1j*k*(x**2+y**2)/(2*z))

ffts=np.fft.fft2(img)*np.fft.fft2(g)
E=np.fft.ifft2(ffts)

plt.imshow(E.real,cmap=plt.cm.Greys_r)
plt.colorbar()
plt.show()
