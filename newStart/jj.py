from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as rbs
from mpi4py import MPI
import time
import numexpr as ne

def main(slice,comm,rank,size):

    if rank==0:
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

        ''' z is the slice we want to look at '''
        z=250e-6
        #z=13e-3-250e-6
        #z=13e-3
        z=slice

        print('Distance: {0}'.format(z))

        temp[xx,yy]=ne.evaluate('img*(L/Rprime)**4*exp((1j*k*z*Rprime)/L)')

        kx,ky=K.shape

        ii=np.arange(kx)
        jj=np.arange(ky)

        print(comm.rank,comm.size)

    else:
        kx=None
        ky=None
        L=None
        Xprime=None
        Yprime=None
        temp=None
        Kreal=None
        Kimag=None
        ii=None
        jj=None
        k=None

    comm.Barrier()

    print('rank:{0} size:{1} \n'.format(comm.rank,comm.size))
    comm.Barrier()

    print('broadcasting')
    kx=comm.bcast(kx,root=0)
    ky=comm.bcast(ky,root=0)
    L=comm.bcast(L,root=0)
    Xprime=comm.bcast(Xprime,root=0)
    Yprime=comm.bcast(Yprime,root=0)
    temp=comm.bcast(temp,root=0)
    Kreal=comm.bcast(Kreal,root=0)
    Kimag=comm.bcast(Kimag,root=0)
    ii=comm.bcast(ii,root=0)
    jj=comm.bcast(jj,root=0)
    k=comm.bcast(k,root=0)

    print('done broadcasting')
    comm.Barrier()

    rows = [comm.rank + comm.size * aa for aa in range(int(kx/comm.size)+1) if comm.rank + comm.size*aa < kx]

    rows2 = [comm.rank + comm.size * bb for bb in range(int(ky/comm.size)+1) if comm.rank + comm.size*bb < ky]

    comm.Barrier()

    print('loops now')

    for smallX in rows:
        for smallY in rows2:
            print(smallX,smallY)
            temp2=ne.evaluate('temp*exp((1j*k*(smallX*Xprime+smallY*Yprime))/L)')
            temp3=rbs(ii,jj,temp2.real)
            Kreal[smallX,smallY]=temp3.integral(0,kx,0,ky)
            temp4=rbs(ii,jj,temp2.imag)
            Kimag[smallX,smallY]=temp4.integral(0,kx,0,ky)

    comm.Barrier()

    if rank==0:
        Kreal.dump('Kreal{}.dat'.format(slice))
        Kimag.dump('Kimag{}.dat'.format(slice))

    print(time.time()-first)

if __name__=='__main__':

    comm=MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    slices=[250e-6,13e-3-250e-6,13e-3]

    for slice in slices:
        print('on slice:{0}'.format(slice))
        main(slice,comm,rank,size)
