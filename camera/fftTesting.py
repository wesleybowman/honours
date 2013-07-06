import matplotlib.pyplot as plt
import numpy as np

def fft(img):
    i=plt.imread(img)
    #i=np.mean(i,2)

    FFT=(np.fft.fftshift(np.fft.fft2(i)))
    IFFT=np.real(np.fft.ifft2(np.fft.ifftshift(FFT)))

    plt.subplot(1,2,1)
    plt.imshow(np.log(abs(FFT))**2)
    plt.subplot(1,2,2)
    plt.imshow(IFFT)
    #plt.imshow(i)
    plt.show()

def ifft(img):
    i=plt.imread(img)
    #i=np.mean(i,2)

    IFFT=(np.fft.ifftshift(np.fft.ifft2(i)))
    FFT=np.real(np.fft.fft2(np.fft.fftshift(IFFT)))

    plt.subplot(1,2,1)
    plt.imshow(np.log(abs(IFFT))**2)
    plt.subplot(1,2,2)
    plt.imshow(FFT)
    #plt.imshow(i)
    plt.show()

#img='testImage.png'
img='testImage2.jpg'
fft(img)
ifft(img)

#i=plt.imread('test.png')
##i=np.mean(i,2)
#
#FFT=(np.fft.fftshift(np.fft.fft2(i)))
#IFFT=np.real(np.fft.ifft2(np.fft.ifftshift(FFT)))
#
#plt.subplot(1,2,1)
#plt.imshow(np.log(abs(FFT))**2)
#plt.subplot(1,2,2)
#plt.imshow(IFFT)
##plt.imshow(i)
#plt.show()

