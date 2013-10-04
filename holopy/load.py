from __future__ import division,print_function
#import holopy as hp
#import numpy as np
#import matplotlib.pyplot as plt
import tables as tb
from mayavi import mlab

def loadArray():

    matrix = f.getNode('/complexHologram')
    array = matrix.read()

    return array

#with tb.openFile('154.hdf') as f:
with tb.openFile('154sobHypot.hdf') as f:

    hologram = loadArray()
    recInt = abs(hologram) * abs(hologram)

    mlab.contour3d(recInt)
    #mlab.pipeline.volume(mlab.pipeline.scalar_field(recInt))

    mlab.axes(x_axis_visibility=True, y_axis_visibility=True,
              z_axis_visibility=True, nb_labels = 16, zlabel='Time')

    #mlab.outline()
    mlab.show()

print('closed')

