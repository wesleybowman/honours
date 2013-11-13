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

    a, b = diff.shape

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
        #propInt=propShift.real*propShift.real+propShift.imag*propShift.imag
        propInt = abs(propShift)

        #plt.imshow(np.log(propInt.real+1e-3),cmap=plt.cm.Greys_r)
        plt.imshow((propInt),cmap=plt.cm.Greys_r)
        plt.show()

    return prop


def hologram(prop,show=None):
    holo=np.fft.ifft2(prop)
    #holoInt=holo.real*holo.real+holo.imag*holo.imag
    holoInt = abs(holo)

    if show:
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(holoInt, aspect='auto',cmap=plt.cm.Greys_r)
        fig.savefig('cgh.png')

        #plt.imshow((holoInt),cmap=plt.cm.Greys_r)
        #plt.savefig('cgh.png', bbox_inches='tight')
        #plt.show()

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
    img=plt.imread('bar.png')
    img=plt.imread('whitebar.png')
    #img=plt.imread('whitecircle.png')


    try:
        img=np.mean(img,2)
    except ValueError:
        pass

    #img=normalizeImage(img)

    diff = diffraction(img)
    prop = propagation(diff, z=2, show=1)
    holo, holoInt = hologram(prop,show=1)

    #recInt=reconstruction(holoInt)

#    modifiedReconstruction(holoInt)
