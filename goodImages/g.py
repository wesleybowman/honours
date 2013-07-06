'''
Slides of the reconstruction
'''
import numpy as np
import matplotlib.pyplot as plt
import time
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


#def reconstruction(holoInt,wavelength=632e-9,z=1,px=0.01,py=0.01,show=1):
#
#    n,m=holoInt.shape
#
#    dx=px/n*np.ones(n)
#    dy=py/m*np.ones(m)
#    dx,dy=np.meshgrid(dx,dy)
#
#    x,y=np.mgrid[0:n,0:m]
#
#    x2=x*x
#    y2=y*y
#    dx2=dx*dx
#    dy2=dy*dy
#
#    g=np.exp(-1j*np.pi/(wavelength*z)*(x2*dx2.T+y2*dy2.T))
#
#    rec=holoInt*g
#    recShift=np.fft.ifftshift(rec)
#    recShift=rec
#    rec=np.fft.ifft2(recShift)
#    recInt=rec.real*rec.real+rec.imag*rec.imag
#
#    if show:
#        plt.imshow(np.log(recInt+1),cmap=plt.cm.Greys_r)
#        plt.show()
#
#    return recInt,rec
def reconstruction(holoInt,wavelength=632e-9,z=1,px=0.01,py=0.01,show=1):

    n,m=holoInt.shape

    dx=(px/n)*np.ones(n)
    dy=(py/m)*np.ones(m)
    dx,dy=np.meshgrid(dx,dy)

    x,y=np.mgrid[0:n,0:m]

    x2=ne.evaluate('x*x')
    y2=ne.evaluate('y*y')
    dx2=ne.evaluate('dx*dx')
    dy2=ne.evaluate('dy*dy')

    x2=x2
    y2=y2

    dx2=dx2.T
    dy2=dy2.T

    pi=np.pi
    xy=ne.evaluate('x2*dx2+y2*dy2')
    ev=ne.evaluate('-1j*pi/(wavelength*z)*(xy)')
    g=ne.evaluate('exp(ev)')

    rec=ne.evaluate('holoInt*g')
#    recShift=np.fft.ifftshift(rec)
    recShift=rec
    rec=np.fft.ifft2(recShift)
    recInt=rec.real*rec.real+rec.imag*rec.imag

    if show:
        plt.imshow(np.log(recInt+1),cmap=plt.cm.Greys_r)
        plt.show()

    return recInt,rec

if __name__=='__main__':

    object1=plt.imread('newredblood1.png')
    ref=plt.imread('newref1.png')

    object1=plt.imread('michaelhair1.png')
    ref=plt.imread('newref1.png')

    img=object1-ref

    try:
        img=np.mean(img,2)
    except ValueError:
        pass

    zstep=1000
    start=3.4e-2
    stop=3.8e-2
    distance,step=np.linspace(start,stop,zstep,retstep=True)

    print step

    distX=4.8e-3
    distY=3.6e-3

#    diff=diffraction(img)
#    prop=propagation(diff)
#    holo,holoInt=hologram(prop)

    holoInt=img

    recInt,rec=reconstruction(holoInt,z=1.27,show=0)#,wavelength=632e-9,z=1,px=7.5e-6,py=7.5e-6,show=1)

    count=0
    for i in distance:
        print i
        recInt,rec=reconstruction(holoInt,z=i,px=distX,py=distY,show=0)
        if count==0:
            temp=recInt
            count+=1
        if count==1:
            newImg=np.dstack((temp,recInt))
            count+=1
        else:
            newImg=np.dstack((newImg,recInt))

    print 'plotting'
    plt.ion()
    fig=plt.figure()
    ax=fig.gca()
    for i in xrange(zstep):
        print i
        plt.clf()
        plt.imshow(newImg[...,i])
        fig.canvas.draw()
        time.sleep(1e-3)

