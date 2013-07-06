import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



i=plt.imread('good.png')
#i=np.mean(i,2)

ifft=np.fft.ifftshift(np.real(np.fft.ifft2(i)))

plt.subplot(1,1,1)
plt.imshow(ifft)
plt.show()

'''
FFT1=(np.fft.fftshift(np.fft.fft2(i)))
IFFT1=np.real(np.fft.ifft2(np.fft.ifftshift(FFT1)))

plt.subplot(1,2,1)
plt.imshow(np.log(abs(FFT1))**2)
plt.subplot(1,2,2)
plt.imshow(IFFT1)
plt.show()
'''
