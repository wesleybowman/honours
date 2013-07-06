from __future__ import print_function
import numpy as np
import holopy as hp
from holopy.propagation import propagate
from holopy.core import load
import matplotlib.pyplot as plt
import time

optics = hp.core.Optics(wavelen=.405, index=1.33,
                        polarization=[1.0, 0.0])

holo = load('jerichoObject.bmp', spacing = 6e-6, optics =optics)
bg = load('jerichoRef.bmp', spacing = 6e-6, optics =optics)

holo=holo - bg

rec_vol = propagate(holo, np.linspace(245e-6, 250e-6, 5))

plt.ion()
fig=plt.figure()
ax=fig.gca()
for i in xrange(rec_vol.shape[2]):
    print(i)
#        name='figure%d.png'%(i,)
#        plt.savefig(name,bbox_inches=0)
    plt.clf()

    plt.imshow(rec_vol[...,i].real,cmap=plt.cm.Greys_r)
    fig.canvas.draw()
    time.sleep(1)

