import numpy as np
import matplotlib.pyplot as plt


def normalizeImage(img):
    '''This is to make sure every input image is seen as the same by the
       program later on. This way, the image will be between 0 and 1.'''

    imgMax=np.amax(img)

    for (i,j),k in np.ndenumerate(img):
        img[i,j]=k/imgMax

    return img


def diffraction(img,show=None):

    diff=np.fft.fft2(img)

    if show:
        diffShift=np.fft.fftshift(diff)
        diffInt=diffShift.real*diffShift.real+diffShift.imag*diffShift.imag

        plt.imshow(np.log(diffInt.real+1e-3),cmap=plt.cm.Greys_r)
        plt.show()

    return diff


def propagation(diff,wavelength=632e-9,z=1,px=0.01,py=0.01,show=None):


    '''Not sure which k to use '''
    k=2*np.pi/wavelength
    k=1/wavelength

    a,b=diff.shape

    k2=k*k
    kx=np.fft.fftfreq(a,px/a)
    ky=np.fft.fftfreq(b,py/b)

    kx2=kx*kx
    ky2=ky*ky

    prop=np.empty(diff.shape)+0j

    for (x,y),value in np.ndenumerate(diff):
        kz=np.sqrt(k2-kx2[x]-ky2[y])
        prop[x,y]=diff[x,y]*np.exp(2*np.pi*1j*kz*z)

    if show:
        propShift=np.fft.fftshift(prop)
        propInt=propShift.real*propShift.real+propShift.imag*propShift.imag

        plt.imshow(np.log(propInt.real+1e-3),cmap=plt.cm.Greys_r)
        plt.show()

    return prop


def hologram(prop,show=None):
    holo=np.fft.ifft2(prop)
    holoInt=holo.real*holo.real+holo.imag*holo.imag
    if show:
        plt.imshow((holoInt),cmap=plt.cm.Greys_r)
        plt.show()

    return holo,holoInt


def reconstruction(holoInt,wavelength=632e-9,z=1,px=0.01,py=0.01,show=1):

    n,m=holoInt.shape

    dx=px/n
    dy=py/m

    g=np.empty(holoInt.shape)+0j
    rec=np.empty(holoInt.shape)+0j

    for (x,y),value in np.ndenumerate(holoInt):
        g[x,y]=np.exp(-1j*np.pi/(wavelength*z)*(x*x*dx*dx+y*y*dy*dy))
        rec[x,y]=holoInt[x,y]*g[x,y]

    recShift=np.fft.ifftshift(rec)
    rec=recShift
    rec=np.fft.ifft2(rec)

    recInt=rec.real*rec.real+rec.imag*rec.imag

    if show:
        plt.imshow(np.log(recInt+1),cmap=plt.cm.Greys_r)
        plt.show()

    return recInt

if __name__=='__main__':
    img=plt.imread('smallA.png')
    #img=plt.imread('onePixels.png')
    #img=plt.imread('dot.png')
    img=plt.imread('bar.png')
#    img=plt.imread('dot2.png')
#    img=plt.imread('offCenterDot.png')
#    img=plt.imread('farther2.png')
#    img=plt.imread('screwfar1.png')
#    img=plt.imread('thickerSinglePixel.png')
#    img=plt.imread('object.png')


    try:
        img=np.mean(img,2)
    except ValueError:
        pass

    img=normalizeImage(img)

    diff=diffraction(img)
    prop=propagation(diff)
    holo,holoInt=hologram(prop)

    recInt=reconstruction(holoInt)

#    modifiedReconstruction(holoInt)
