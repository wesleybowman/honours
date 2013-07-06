import cv2.cv as cv
#import cv
import numpy as np
import matplotlib.pyplot as plt


def getPhoto():

    cv.NamedWindow("camera", 1)
    capture = cv.CaptureFromCAM(1)

    img = cv.QueryFrame(capture)
    cv.SaveImage("img1.png", img)
    img2 = cv.QueryFrame(capture)
    cv.SaveImage("img2.png", img2)
    img3 = cv.QueryFrame(capture)
    cv.SaveImage("img3.png", img3)

def plotPhoto():

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

def plot():

    i2=plt.imread('img2.png')
    i3=plt.imread('img3.png')
    i2=np.mean(i2,2)
    i3=np.mean(i3,2)

    plt.subplot(1,2,1)
    plt.imshow(i2,cmap=plt.cm.Greys_r)
    plt.subplot(1,2,2)
    plt.imshow(i3)#,cmap=plt.cm.Greys_r)
    plt.show()

if __name__=='__main__':
    getPhoto()
    plot()
