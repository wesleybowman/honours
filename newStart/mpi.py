from __future__ import division,print_function
from mpi4py import MPI
import numpy as np

comm=MPI.COMM_WORLD
size=comm.Get_size

if comm.rank==0:
    print('Processors used: {}'.format(size))

comm.Barrier()

n=28
m=10

tempX=np.zeros((n,m))
buf=np.zeros((n,m))

#rowsX = [comm.rank + comm.size * aa for aa in xrange(int(n/comm.size)+1) if comm.rank + comm.size*aa < n]
#rowsY = [comm.rank + comm.size * bb for bb in xrange(int(m/comm.size)+1) if comm.rank + comm.size*bb < m]

comm.Barrier()

print(tempX)

comm.Scatter( [tempX, MPI.DOUBLE], [buf, MPI.DOUBLE])

for x in xrange(n):
    for y in xrange(m):

        buf[x,y]=x+1

#buf += 1

comm.Allgather([buf, MPI.DOUBLE], [tempX, MPI.DOUBLE])

print(tempX)


#for x in rowsX:
#    for y in xrange(m):
#
#        print(x, y)
#        tempX[x,y]=x+1
#
#comm.Barrier()
#
##print(tempX)
#comm.allgather(buf,tempX)
#print(tempX)
#comm.Barrier()
#print('buf')
#print(buf)
#bad=np.where(tempX!=0)
#print(bad[0].shape)
#
