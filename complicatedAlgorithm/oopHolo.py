from __future__ import division,print_function
import holopy as hp
import numpy as np
import matplotlib.pyplot as plt

class reconstruction():

    def __init__(self,name):

        if name=='myHolo':
            self.rec=self.myHolo()

        else:
            self.rec=self.jericho()


    def jericho(self):
        ''' using Dr. Jericho's setup and images, currently doesn't work.
            I think this is because Jericho using a spherical wave. '''

        optics = hp.core.Optics(wavelen=405e-9, index=1.33, polarization=[1.0, 0.0])

        obj = hp.load('jerichoObject.bmp', spacing=6e-6, optics=optics)
        ref = hp.load('jerichoRef.bmp', spacing=6e-6, optics=optics)

        holo=obj-ref

        #rec = hp.propagate(holo, np.linspace(200e-6, 300e-6, 10))
        rec = hp.propagate(holo, np.linspace(12.5e-3, 13e-3, 20))

        return rec

    def myHolo(self):
        ''' my holographic setup, which I think works since we use plane waves '''

        optics = hp.core.Optics(wavelen=635e-9, index=1.33, polarization=[1.0, 0.0])

        obj = hp.load('fibre1.png', spacing=7.6e-6, optics=optics)
        ref = hp.load('refFibre1.png', spacing=7.6e-6, optics=optics)

        holo = obj - ref

        rec = hp.propagate(holo, np.linspace(2.5e-2, 7.5e-2, 20))

        return rec

if __name__=='__main__':

    rec=reconstruction('jericho')
    rec=rec.rec

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
