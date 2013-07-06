import scipy
from holopy.scattering.scatterer import Sphere, Spheres
from holopy.core import ImageSchema, Optics
from holopy.scattering.theory import Mie
import matplotlib.pyplot as plt

sphere = Sphere(n = 1.59+.0001j, r = .5, center = (4, 3, 5))

schema = ImageSchema(shape = 100, spacing = .1,
                     optics = Optics(wavelen = .660, index = 1.33,
                                     polarization = [1,0]))

s1 = Sphere(center=(5, 5, 5), n = 1.59, r = .5)
s2 = Sphere(center=(4, 4, 5), n = 1.59, r = .5)
collection = Spheres([s1, s2])
holo = Mie.calc_holo(collection, schema)

scipy.misc.imsave('holoTest.png',holo)
plt.imshow(holo)
plt.show()


