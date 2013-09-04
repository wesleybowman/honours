from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt

def myHolo():
    ''' my holographic setup, which I think works since we use plane waves '''

    optics = hp.core.Optics(wavelen=635e-9, index=1.33, polarization=[1.0, 0.0])

    #obj = hp.load('tryit.png', spacing=7.6e-6, optics=optics)
    #ref = hp.load('ref.png', spacing=7.6e-6, optics=optics)

    obj = hp.load('fibreBright.png', spacing=7.6e-6, optics=optics)
    ref = hp.load('refBright.png', spacing=7.6e-6, optics=optics)

    #obj = hp.load('singleBright.png', spacing=7.6e-6, optics=optics)
    #ref = hp.load('singleRef.png', spacing=7.6e-6, optics=optics)

    holo = obj - ref

    ''' between 4 cm and 10 cm, should be about 5.5 cm '''
    #rec = hp.propagate(holo, np.linspace(4e-2, 10e-2, 200))
    rec = hp.propagate(holo, np.linspace(5e-2, 6e-2, 200))
    #rec = hp.propagate(holo, np.linspace(0.5e-2, 2e-2, 200))

    return rec

rec = myHolo()

recInt = abs(rec) * abs(rec)

print('Amplitude')
hp.show(recInt)
plt.show()

print('Imaginary')
hp.show(rec.imag)
plt.show()

print('Phase')
phase=np.arctan(rec.imag/rec.real)
#phase=np.unwrap(phase)
hp.show(phase)
#hp.show(np.angle(rec))
plt.show()

#import os
#os.chdir('saves/')
#count=0
#for i in recInt:
#    filename = 'figure{}'.format(count)
#    plt.imsave(filename, i)
#    count+=1
