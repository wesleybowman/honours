from __future__ import division, print_function
import numpy as np
import numexpr as ne

a=np.ones((10,10))*10
temp=np.empty(a.shape)

n,m=a.shape

A=(1/n)*(1/m)
print(a.sum())
s=np.sum(a)
print(s)

S=ne.evaluate('sum(a)')
print(S)

#a1=a.sum(axis=1)
#a0=a.sum(axis=0)

#a1=a1[...,np.newaxis]
#a0=a0[...,np.newaxis]

#S=(a1+a0)*A
#print(a0)
#print(a1)
#print(S)

#S=(a.sum(axis=1)*a.sum(axis=0))*A

#print(S.sum())

