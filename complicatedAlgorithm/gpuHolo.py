from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
from numpy.core.umath_tests import inner1d
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import scikits.cuda.integrate as integrate

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

mod = SourceModule('''
#include <stdio.h>
__global__ void loop_pixels(int n_end, int m_end)
{
    for (int n=0; n<n_end; n++)
    {
        for (int m=0; m<m_end; m++)
        {
            printf("x: %d y: %d\\n", n, m);
        }
    }
}
''')


#loop_pixels = mod.get_function('loop_pixels')

#loop_pixels( np.int32(n), np.int32(m), block = (256,4,1))

for x in xrange(n):
    for y in xrange(m):

        print(x, y)

        #set a tempKSI so numexpr can work
        tempKSI=KSIdotR[x,y]
        temp = ne.evaluate('holo * exp(1j * k * tempKSI / KSInorm)')

        #Sum up temp, and multiply by the length and width to get the volume.
        #reconstruction[x,y]=temp.sum()*distX*distY
        temp_gpu =  gpuarray.to_gpu(temp)
        print(temp_gpu.gpudata)
        reconstruction[x,y] = integrate.trapz2d(temp_gpu, 6e-6, 6e-6)



name='{}reconstruction'.format(zz)
np.save(name,reconstruction)
