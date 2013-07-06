import numpy
from functools import *
from math import *
import matplotlib.pyplot as pyplot
import matplotlib.cm as colormaps

class Fraunhofer1D:
    def __init__(self, aperture, length, wavelength):
        '''
        Initialize the Fraunhofer algorithm with an aperture, total
        length, and wavelength of monochromatic plane waves.
        '''
        self.apt = aperture
        self.width = aperture.get_max_width()
        self.length = length
        self.lmbda = wavelength
        self.scr = numpy.zeros((5, self.width))
        self.field = numpy.zeros((self.length, self.width))

    def run(self):
        '''
        Run the manual integration over the aperture and every point
        on the aperture.
        '''
        for i in range(self.width):
            for j in range(self.length):
                for u in range(self.width):

                    # We only consider this point if it is not opaque.
                    if self.apt.get_opacity(u) == 1:
                        # Calculate distance from aperture to screen.
                        dx = abs(i - u)
                        dy = abs(j - self.length)
                        d = sqrt(dx * dx + dy * dy)
                        
                        # Add the contribution of this point to the
                        # field.
                        self.field[j, i] += cos(2.0 * pi * d / self.lmbda)

        # Square all points on the screen to get intensity.
        self.scr[0:5,:] += self.field[0,:]
        self.scr = self.scr ** 2

    def display(self):
        '''
        Show the field and screen in 9:1 ratio with pyplot.
        '''
        pyplot.subplot(10, 1, (1,9))
        pyplot.imshow(self.field, cmap = colormaps.gray)
        pyplot.subplot(10, 1, 10)
        pyplot.imshow(self.scr, cmap = colormaps.gray)
        pyplot.show()

class Aperture:
    def __init__(self, max_width, func):
        '''
        Create an aperture, positioned at (0, 0) in world space, of
        maximum width `max_width` with an opacity function `func`, which
        accepts an x-coordinate as its argument.
        '''
        self.max_width = max_width
        self.func = func

    def get_max_width(self):
        '''
        Returns the width of the aperture plane.
        '''
        return self.max_width

    def get_opacity(self, x):
        '''
        Return the result of the aperture function evaluated at
        the point `x`.
        '''
        return self.func(x)

class ShapeFunctions:
    @staticmethod
    def doubleslit(x1, x2, d, x):
        if x >= x1 - d / 2 and x <= x1 + d / 2:
            return 1
        elif x >= x2 - d / 2 and x <= x2 + d / 2:
            return 1
        else:
            return 0

    @staticmethod
    def singleslit(x0, d, x):
        if x >= x0 - d / 2 and x <= x0 + d / 2:
            return 1
        else:
            return 0

if __name__ == '__main__':
    apt = Aperture(200, partial(ShapeFunctions.doubleslit, 80, 120, 1))
    fraun = Fraunhofer1D(apt, 200, 10)

    fraun.run()
    fraun.display()
    
    '''FFT testing '''
    i=fraun.field
    FFT=(numpy.fft.fftshift(numpy.fft.fft2(i)))
    IFFT=numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(i)))
    
    pyplot.subplot(1,2,1)
    pyplot.imshow(numpy.log(abs(FFT))**2)
    pyplot.subplot(1,2,2)
    pyplot.imshow(IFFT)
    #plt.imshow(i)
    pyplot.show()
