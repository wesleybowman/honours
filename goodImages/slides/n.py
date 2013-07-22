import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import tables as tb
import time

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

    dx=(px/n)*np.ones(n)
    dy=(py/m)*np.ones(m)
    dy,dx=np.meshgrid(dy,dx)

    x,y=np.mgrid[0:n,0:m]

    x2=ne.evaluate('x*x')
    y2=ne.evaluate('y*y')
    dx2=ne.evaluate('dx*dx')
    dy2=ne.evaluate('dy*dy')

    pi=np.pi

    xy=ne.evaluate('x2*dx2+y2*dy2')
    ev=ne.evaluate('-1j*pi/(wavelength*z)*(xy)')
    g=ne.evaluate('exp(ev)')

    rec=ne.evaluate('holoInt*g')
    recShift=np.fft.ifftshift(rec)
    recShift=rec
    rec=np.fft.ifft2(recShift)
    recInt=rec.real*rec.real+rec.imag*rec.imag
    recPhase=np.arctan(rec.imag/rec.real)
    recPhase=np.unwrap(recPhase)

    if show:
        plt.imshow((recInt),cmap=plt.cm.Greys_r)
        plt.show()

    return recInt,recPhase,rec

if __name__=='__main__':

    obj=plt.imread('fibre1.png')
    ref=plt.imread('refFibre1.png')
    img=obj-ref

    img=np.mean(img,2)

    from scipy import sparse
    newImg=sparse.lil_matrix(img)
    print newImg.shape

    zstep=1000
    start=1.5e-2
    end=4.0e-2
    stepSize=1e-6

#    distance,step=np.linspace(start,end,zstep,retstep=True)
    distance=np.arange(start,end,stepSize)

    holoInt=img
    distX=4.8e-3
    distY=3.6e-3
    dataShape=(holoInt.shape[0],holoInt.shape[1],distance.shape[0])

    with tb.openFile('recon.h5','w') as data:
        root=data.root
        filters=tb.Filters(complevel=9)
        recData=data.createCArray(root,'recData',tb.ComplexAtom(itemsize=16),shape=dataShape,filters=filters)
        print 'created'

        for i,value in enumerate(distance):
            print i
            recInt,recPhase,rec=reconstruction(holoInt,z=value,px=distX,py=distY,show=0)
            recData[...,i]=rec


    print stepSize

#    plt.ion()
#    fig=plt.figure()
#    ax=fig.gca()
#    for i,value in enumerate(distance):
#        print i, value
##        name='figure%d.png'%(i,)
##        plt.savefig(name,bbox_inches=0)
#        plt.clf()
#
#        recInt,recPhase,rec=reconstruction(holoInt,z=value,px=distX,py=distY,show=0)
#
#
#        plt.subplot(221)
#        plt.imshow(img)
#        plt.subplot(222)
#        plt.imshow(recInt)
#        plt.subplot(223)
#        plt.imshow(rec.real)
#        plt.subplot(224)
#        plt.imshow(recPhase)
#        ''' Look at correcting the phase'''
#
#        fig.canvas.draw()
#        time.sleep(1e-3)
#
