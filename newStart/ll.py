from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as rbs
from mpi4py import MPI


comm=MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

obj=plt.imread('jerichoObject.bmp')
ref=plt.imread('jerichoRef.bmp')

holo=obj-ref

Kreal=np.empty(holo.shape)+0j
Kimag=np.empty(holo.shape)+0j
temp=np.empty(holo.shape)+0j

wavelength=405e-9
k=2*np.pi/(wavelength)
z=250e-6
#z=13e-3-250e-6

distX=6e-6
distY=6e-6

n=float(holo.shape[0])
m=float(holo.shape[1])

a = np.arange(0,n)
b = np.arange(0,m)

#create all r vectors
R = np.empty((holo.shape[0], holo.shape[1], 3))
R[:,:,0] = np.repeat(np.arange(holo.shape[0]), holo.shape[1]).reshape(holo.shape) * distX
R[:,:,1] = np.arange(holo.shape[1]) * distY
R[:,:,2] = z

#create all ksi vectors
KSI = np.empty((holo.shape[0], holo.shape[1], 3))
KSI[:,:,0] = np.repeat(np.arange(holo.shape[0]), holo.shape[1]).reshape(holo.shape) * distX
KSI[:,:,1] = np.arange(holo.shape[1]) * distY
KSI[:,:,2] = z

# vectorized 2-norm; see http://stackoverflow.com/a/7741976/4323
KSInorm = np.sum(np.abs(KSI)**2,axis=-1)**(1./2)

# loop over entire holo which is same shape as holo, rows first
# this loop populates holo one pixel at a time (so can be parallelized)

rowsX = [comm.rank + comm.size * aa for aa in range(int(n/comm.size)+1) if comm.rank + comm.size*aa < n]
rowsY = [comm.rank + comm.size * bb for bb in range(int(m/comm.size)+1) if comm.rank + comm.size*bb < m]

for x in rowsX:
    for y in rowsY:

        print(x, y)

        KSIdotR = np.dot(KSI[x,y], R[x,y])
        temp = holo * np.exp(1j * k * KSIdotR / KSInorm)

        '''interpolate so that we can do the integral and take the integral'''
        temp2 = rbs(a, b, temp.real)
        Kreal[x,y] = temp2.integral(0, n, 0, m)
        temp3 = rbs(a, b, temp.imag)
        Kimag[x,y] = temp3.integral(0, n, 0, m)


Kreal.dump('Kreal.dat')
Kimag.dump('Kimag.dat')



