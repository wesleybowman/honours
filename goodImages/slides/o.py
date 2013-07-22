from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne

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

    kx,ky=np.meshgrid(kx,ky)

    kx2=kx*kx
    ky2=ky*ky

    kz=np.sqrt(k2-kx2-ky2)

    n,m=diff.shape
    x,y=np.mgrid[0:n,0:m]
    prop=diff*np.exp(2*np.pi*1j*kz.T*z)

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

    dx=(px)*np.ones(m) #(px/n)*np.ones(n)
    dy=(py)*np.ones(n) #(py/m)*np.ones(m)

    dx,dy=np.meshgrid(dx,dy)

    x,y=np.mgrid[0:n,0:m]

    x2=ne.evaluate('x*x')
    y2=ne.evaluate('y*y')
    dx2=ne.evaluate('dx*dx')
    dy2=ne.evaluate('dy*dy')

    x2=x2
    y2=y2

    dx2=dx2
    dy2=dy2

    pi=np.pi

    xy=ne.evaluate('x2*dx2+y2*dy2')
    ev=ne.evaluate('-2j*pi*(xy)/(wavelength*z)')
    g=ne.evaluate('exp(ev)')

#    k=2*pi/wavelength
#    ksi=np.sqrt(x2+y2+z*z)
#    newref=ne.evaluate('1/(ksi)*exp(-1j*k*ksi)')
#    g=ne.evaluate('g*newref')

    rec=ne.evaluate('holoInt*g')
    recShift=np.fft.ifftshift(rec)
    recShift=rec
    rec=np.fft.ifft2(recShift)
    recInt=rec.real*rec.real+rec.imag*rec.imag

    if show:
        plt.imshow(recInt,cmap=plt.cm.Greys_r)
        plt.show()
        plt.imshow(np.log(recInt+1),cmap=plt.cm.Greys_r)
        plt.show()

    return recInt,rec

if __name__=='__main__':

    obj=plt.imread('fibre1.png')
    ref=plt.imread('refFibre1.png')

    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')
    img=obj-ref

    #img=np.mean(img,2)

    start=200e-6
    stop=13e-3
    stepSize=1e-6
    distance=np.arange(start,stop,stepSize)

#    diff=diffraction(img)
#    prop=propagation(diff)
#    holo,holoInt=hologram(prop)

    holoInt=img
    distX=6e-6
    distY=6e-6
    distance=13e-3
    distance=13e-3-250e-6
    #distance=250e-6

    recInt,rec=reconstruction(holoInt,wavelength=405e-9,z=distance,px=distX,py=distY,show=1)

#    plt.ion()
#    fig=plt.figure()
#    ax=fig.gca()
#    for i,value in enumerate(distance):
#        print i, value
##        name='figure%d.png'%(i,)
##        plt.savefig(name,bbox_inches=0)
#        plt.clf()
#
#        recInt,rec=reconstruction(holoInt,wavelength=405e-9,z=value,px=distX,py=distY,show=0)
#
#        plt.imshow(recInt,cmap=plt.cm.Greys_r)
#        fig.canvas.draw()
#        time.sleep(1e-3)
#
