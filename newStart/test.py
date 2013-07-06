from __future__ import print_function
import numpy as np
from numba import autojit

@autojit
def test():
    a=np.arange(0,100,1)
    func=lambda y,x: y+x

    print(func(a,a))

test()
