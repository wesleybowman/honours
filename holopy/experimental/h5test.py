from __future__ import division, print_function
import holopy as hp
import numpy as np
import tables as tb
from mpi4py import MPI

def loadHolo():
    optics = hp.core.Optics(wavelen=.66, index=1.33, polarization=[1.0, 0.0])

    magnification = 20
    Spacing = 6.8 / magnification

    obj = hp.load('P1000018.tiff', spacing=Spacing, optics=optics)
    ref = hp.load('P1000019.tiff', spacing=Spacing, optics=optics)

    holo = obj - ref

    return holo

holo = loadHolo()

comm=MPI.COMM_WORLD

slices = np.arange(100,10100,100)
n=100
newSlices = [comm.rank + comm.size * aa for aa in range(int(n/comm.size)+1) if comm.rank + comm.size*aa < n]

comm.Barrier()


with tb.openFile('P1819.hdf','w') as f:
    matrix = f.createCArray(f.root, 'complexHologram',tb.ComplexAtom(itemsize=8), shape=(4016,3016,100))

    #count = 0
    for i in newSlices:

        print(i, slices[i])
        rec = hp.propagate(holo, slices[i])
        matrix[..., i] = rec
        #matrix[..., count] = rec
        #count += 1

