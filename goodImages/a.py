import numpy as np
import matplotlib.pyplot as plt

def plotPhoto():
    '''plot the images'''
    plt.imshow(i)
    plt.show()


def holo(i,x,y,k,z):

    i=i+0j

#    for (x,y),k in np.ndenumerate(i):
#        if k.real==0:
#            A=0
#        else:
#            A=1
#
#        r=0
#        E=A*np.exp(1j*2*np.pi*r/wavelength)
#        i[x][y]=E

    for (a,b),value in np.ndenumerate(i):
        i[a][b]=i[a][b]*np.exp(-k*1j*(a**2*x**2+b**2*y**2)/(z))

    hologram=np.fft.fft2(i)


    plt.imshow(hologram.real,cmap=plt.cm.Greys_r)
    plt.show()
    return hologram

def plotFFT(i):

    i=i+0j
    for (k,l),value in np.ndenumerate(i):
        i[k][l]=i[k][l]*np.exp(-1*np.pi*1j*(k**2*7.4*10**(-6)+l**2*7.4*10**(-6))/(635*10**(-9)*6*10**(-3)))

    '''Get fft and ifft'''
    FFT1=(np.fft.fftshift(np.fft.fft2(i)))
    IFFT1=np.real(np.fft.ifft2(np.fft.ifftshift(FFT1)))


    '''plot fft and ifft '''
    plt.subplot(1,2,1)
    plt.imshow(np.log(abs(FFT1))**2)
    plt.subplot(1,2,2)
    plt.imshow(IFFT1)
    plt.show()


if __name__=='__main__':

#    img=raw_input('Use which image?: ')
    i=plt.imread('wireAndFrame.png')
    i=plt.imread('good2.png')
    i=plt.imread('farther2.png')
    i=plt.imread('screw1.png')
    i=plt.imread('screwfar1.png')
    i=plt.imread('dice.png')

    one=plt.imread('withObject1.png')
    two=plt.imread('withoutObject1.png')
    i=one*two
    i=plt.imread('holoTest.png')
    i=plt.imread('dot.png')
#    i=plt.imread('blackA.jpg')
    i=np.mean(i,2)
        
    i=plt.imread('thickerSinglePixel.png')



    plt.imshow(i,cmap=plt.cm.Greys_r)
    plt.show()
    plotFFT(i)
    x=4.2e-5
    y=4.2e-5
    wavelength=632.8e-9
    k=2*np.pi/wavelength
    z=4
    h=holo(i,x,y,k,z)

    h=np.fft.ifft2(h)
    plt.imshow(h.real,cmap=plt.cm.Greys_r)
    plt.colorbar()
    plt.show()
