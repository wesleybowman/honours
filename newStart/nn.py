from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
from numpy.core.umath_tests import inner1d

obj=plt.imread('jerichoObject.bmp')
ref=plt.imread('jerichoRef.bmp')

holo=obj-ref

temp=np.empty(holo.shape)+0j
reconstruction=np.empty(holo.shape)+0j

wavelength=405e-9
k=2*np.pi/(wavelength)
z=250e-6
#z=13e-3-250e-6

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

KSIdotR=inner1d(KSI,R)

inner_loops = 251
hl = int(m/inner_loops)

print("starting loops")

for x in xrange(n):
    for i in xrange(inner_loops):
        print (x,i)

        ksiTemp = KSIdotR[x,hl*i:hl*(i+1)]
        neTemp = ksiTemp[:,None,None]

        arg = ne.evaluate("neTemp/KSInorm")
        temp = ne.evaluate("holo * exp(1j * k * arg)")
        temp2 = ne.evaluate("sum(temp, axis=2)")

        reconstruction[x,hl*i:hl*(i+1)]=temp2.sum(axis=1)

reconstruction = ne.evaluate("reconstruction *distX * distY")

reconstruction.dump('reconstruction.dat')
