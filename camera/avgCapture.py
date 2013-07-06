import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt


def getPhoto():
    name=raw_input('Image Name: ')
    name=name+'.png'

    cv.NamedWindow("camera", 1)
    capture = cv.CaptureFromCAM(1)

    count=0

    while count<=100:
        print count

        img = cv.QueryFrame(capture)
        img=np.asarray(img[:,:])
        count+=1

    avgImg=np.average(img,axis=2)

    plt.imsave(name,avgImg)

    plt.imshow(avgImg,cmap=plt.cm.Greys_r)
    plt.show()

if __name__=='__main__':

    getPhoto()
