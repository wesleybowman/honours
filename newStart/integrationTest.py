from __future__ import division, print_function
import numpy as np
from scipy.interpolate import LinearNDInterpolator as interpol
import scipy.integrate as sciInt


a,b=np.mgrid[0:11,0:11]
Z=np.ones((11,11))*10

pts=np.array((a.ravel(),b.ravel())).T
zr=Z.ravel()

surf=interpol(pts,zr)

func=lambda y,x: surf([[x,y]])

res=np.empty(Z.shape)
for (i,j),k in np.ndenumerate(res):
    res[i,j]=sciInt.dblquad(func,0.0,10.0,lambda x:0.0,lambda x:10.0)[0]

print(res[0])


