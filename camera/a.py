#import cv2.cv as cv
import cv
import numpy as np
import matplotlib.pyplot as plt
import time

def getPhoto():
    time.sleep(0.01)

    cv.NamedWindow("camera", 1)
    capture = cv.CaptureFromCAM(1)

    img = cv.QueryFrame(capture)
    cv.SaveImage("img1.png", img)

    img2 = cv.QueryFrame(capture)
    cv.SaveImage("img2.png", img2)

def plotPhoto():

    i=plt.imread('img1.png')
    i2=plt.imread('img2.png')

    '''plot the images'''
    plt.subplot(1,2,1)
    plt.imshow(i)
    plt.subplot(1,2,2)
    plt.imshow(i2)
    plt.show()


def plotFFT():

    i=plt.imread('img1.png')
    i2=plt.imread('img2.png')
    #i=np.mean(i,2)

    '''Get fft and ifft'''
    FFT1=(np.fft.fftshift(np.fft.fft2(i)))
    IFFT1=np.real(np.fft.ifft2(np.fft.ifftshift(FFT1)))

    FFT2=(np.fft.fftshift(np.fft.fft2(i2)))
    IFFT2=np.real(np.fft.ifft2(np.fft.ifftshift(FFT2)))

    '''plot fft and ifft '''
    plt.subplot(1,2,1)
    plt.imshow(np.log(abs(FFT1))**2)
    plt.subplot(1,2,2)
    plt.imshow(IFFT1)
    plt.show()

    plt.subplot(1,2,1)
    plt.imshow(np.log(abs(FFT2))**2)
    plt.subplot(1,2,2)
    plt.imshow(IFFT2)
    plt.show()

if __name__=='__main__':
    getPhoto()
    plotPhoto()
