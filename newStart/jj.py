from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as rbs
from mpi4py import MPI
import time
import numexpr as ne

def compute(kx,ky,start=0,step=1):
    for smallX in xrange(start,kx,step):
        for smallY in xrange(start,ky,step):
            print(smallX,smallY)
            temp2=ne.evaluate('temp*exp((1j*k*(smallX*Xprime+smallY*Yprime))/L)')
            temp3=rbs(i,j,temp2.real)
            Kreal[smallX,smallY]=temp3.integral(0,kx,0,ky)
            temp4=rbs(i,j,temp2.imag)
            Kimag[smallX,smallY]=temp4.integral(0,kx,0,ky)
    return Kreal,Kimag

comm=MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root=0

if rank==root:
    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')

    img=obj-ref

    temp=np.empty(img.shape)+0j
    K=np.empty(img.shape)+0j
    Kreal=np.empty(img.shape)+0j
    Kimag=np.empty(img.shape)+0j

    wavelength=405e-9
    k=2*np.pi/(wavelength)

    ''' L is the distance from the source to the screen '''
    L=13e-3

    n,m=img.shape

    first=time.time()

    a,b=np.mgrid[0:n,0:m]

    r=ne.evaluate('sqrt(L*L+a*a+b*b)')
    Xprime=ne.evaluate('(a*L)/r')
    Yprime=ne.evaluate('(b*L)/r')
    Rprime=ne.evaluate('(L*L)/r')
    xx=ne.evaluate('(Xprime*L)/Rprime')
    yy=ne.evaluate('(Yprime*L)/Rprime')
    xx=xx.astype(int)
    yy=yy.astype(int)

#    r=np.sqrt(L*L+a*a+b*b)
#    Xprime=(a*L)/r
#    Yprime=(b*L)/r
#    Rprime=(L*L)/r
#    xx=(Xprime*L)/Rprime
#    yy=(Yprime*L)/Rprime
#    xx=xx.astype(int)
#    yy=yy.astype(int)

    ''' z is the slice we want to look at '''
    z=250e-6
    #z=13e-3-250e-6
    #z=13e-3

    print('Distance: {0}'.format(z))

    #temp[xx,yy]=img[xx,yy]*(L/Rprime)**4*np.exp((1j*k*z*Rprime)/L)
    temp[xx,yy]=ne.evaluate('img*(L/Rprime)**4*exp((1j*k*z*Rprime)/L)')

    kx,ky=K.shape

    i=np.arange(kx)
    j=np.arange(ky)

else:
    pass
    kx=None
    ky=None
    L=None
    Xprime=None
    Yprime=None
    temp=None
    Kreal=None
    Kimag=None

comm.Barrier()

kx=comm.bcast(kx,root=0)
ky=comm.bcast(ky,root=0)
L=comm.bcast(L,root=0)
Xprime=comm.bcast(Xprime,root=0)
Yprime=comm.bcast(Yprime,root=0)
temp=comm.bcast(temp,root=0)
Kreal=comm.bcast(Kreal,root=0)
Kimag=comm.bcast(Kimag,root=0)

comm.Barrier()
Kreal,Kimag=compute(kx,ky,rank,size)

if rank==root:
    Kreal.dump('Kreal.dat')
    Kimag.dump('Kimag.dat')

print(time.time()-first)
