from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import holopy as hp

comm=MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    print('Processors used: {}'.format(size))

optics = hp.core.Optics(wavelen=635e-9, index=1.33, polarization=[1.0, 0.0])

obj = hp.load('fibre1.png', spacing=7.6e-6, optics=optics)
ref = hp.load('refFibre1.png', spacing=7.6e-6, optics=optics)

holo = obj - ref
n,m = holo.shape

distance = np.linspace(2.5e-2, 7.5e-2, 200)
d=distance.shape

rowsX = [comm.rank + comm.size * aa for aa in range(int(n/comm.size)+1) if comm.rank + comm.size*aa < d]
reconstruction = np.empty((n,m))

comm.Barrier()

for z in rowsX:
    print(distance[z])
    rec = hp.propagate(holo, distance[z])
    reconstruction = np.dstack((reconstruction,rec))

comm.Barrier()

recInt = abs(reconstruction) * abs(reconstruction)

hp.show(recInt)
plt.show()



