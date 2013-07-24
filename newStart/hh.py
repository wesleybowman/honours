from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as rbs
from multiprocessing import Pool
import itertools
import time
import numexpr as ne

def func(smallX,smallY):
    ''' Function used to calculate the integral '''
    print(smallX,smallY)
    temp2=ne.evaluate('temp*exp((1j*k*(smallX*Xprime+smallY*Yprime))/L)')
    temp3=rbs(i,j,temp2.real)
    Kreal[smallX,smallY]=temp3.integral(0,kx,0,ky)
    temp4=rbs(i,j,temp2.imag)
    Kimag[smallX,smallY]=temp4.integral(0,kx,0,ky)

def func_star(a_b):
    ''' Convert `f([1,2])` to `f(1,2)` call, so that pool can do multiple
        arguments. '''
    return func(*a_b)

#@autojit
def main():
    ''' Using multiprocessing.Pool to make this parallel. '''
    pool=Pool()
    pool.map(func_star,itertools.product(i,j))

    Kreal.dump('Kreal.dat')
    Kimag.dump('Kimag.dat')

    print(time.time()-first)

if __name__=='__main__':

    obj=plt.imread('jerichoObject.bmp')
    ref=plt.imread('jerichoRef.bmp')

    img=obj-ref

    temp=np.empty(img.shape)+0j
    K=np.empty(img.shape)+0j
    Kreal=np.empty(img.shape)+0j
    Kimag=np.empty(img.shape)+0j

    wavelength=405e-9
    k=2*np.pi/(wavelength)

    ''' L is the distance from the source to the screen '''
    L=13e-3

    #distX=6e-6
    #distY=6e-6

    n,m=img.shape

    first=time.time()

    a,b=np.mgrid[0:n,0:m]

    r=ne.evaluate('sqrt(L*L+a*a+b*b)')
    Xprime=ne.evaluate('(a*L)/r')
    Yprime=ne.evaluate('(b*L)/r')
    Rprime=ne.evaluate('(L*L)/r')
    xx=ne.evaluate('(Xprime*L)/Rprime')
    yy=ne.evaluate('(Yprime*L)/Rprime')
    xx=xx.astype(int)
    yy=yy.astype(int)

    ''' z is the slice we want to look at '''
    z=250e-6
    #z=13e-3-250e-6
    #z=13e-3

    print('Distance: {0}'.format(z))

    temp[xx,yy]=ne.evaluate('img*(L/Rprime)**4*exp((1j*k*z*Rprime)/L)')

    kx,ky=K.shape
    i=np.arange(0,kx)
    j=np.arange(0,ky)

    main()
