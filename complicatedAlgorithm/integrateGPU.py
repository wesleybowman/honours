import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import scikits.cuda.integrate as integrate

integrate.init()
a = np.asarray(np.random.rand(10), np.float32)
x = np.asarray(np.random.rand(10, 10), np.float32)
x_gpu = gpuarray.to_gpu(x)
a_gpu = gpuarray.to_gpu(a)

b = integrate.trapz(a_gpu)
z = integrate.trapz2d(x_gpu,1,1)
np.allclose(np.trapz(np.trapz(x)), z)
