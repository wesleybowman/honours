from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import numexpr as ne
from numpy.core.umath_tests import inner1d

comm=MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    print('Processors used: {}'.format(size))

obj=plt.imread('jerichoObject.bmp')
ref=plt.imread('jerichoRef.bmp')

holo=obj-ref

temp=np.empty(holo.shape)+0j
reconstruction=np.empty(holo.shape)+0j

wavelength=405e-9
k=2*np.pi/(wavelength)
zz='250e-6'
zz='13e-3-250e-6'
z=eval(zz)

distX=6e-6
distY=6e-6

n=holo.shape[0]
m=holo.shape[1]

#create all r vectors
R = np.empty((n, m, 3))
R[:,:,0] = np.repeat(np.arange(n), m).reshape(holo.shape) * distX
R[:,:,1] = np.arange(m) * distY
R[:,:,2] = z

#create all ksi vectors
KSI = np.empty((n, m, 3))
KSI[:,:,0] = np.repeat(np.arange(n),m).reshape(holo.shape) * distX
KSI[:,:,1] = np.arange(m) * distY
KSI[:,:,2] = z

# vectorized 2-norm; see http://stackoverflow.com/a/7741976/4323
KSInorm = np.sum(np.abs(KSI)**2,axis=-1)**(1./2)

# vectorized dot product
KSIdotR=inner1d(KSI,R)

rowsX = [comm.rank + comm.size * aa for aa in range(int(n/comm.size)+1) if comm.rank + comm.size*aa < n]

comm.Barrier()

for x in rowsX:
    for y in xrange(m):

        print(x, y)

        #set a tempKSI so numexpr can work
        tempKSI=KSIdotR[x,y]
        temp = ne.evaluate('holo * exp(1j * k * tempKSI / KSInorm)')

        #Sum up temp, and multiply by the length and width to get the volume.
        reconstruction[x,y]=temp.sum()*distX*distY

comm.Barrier()

rec=comm.reduce(reconstruction, op=MPI.SUM, root=0)

name='{}reconstruction'.format(zz)
np.save(name,rec)
