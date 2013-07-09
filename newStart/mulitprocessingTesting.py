import itertools
from multiprocessing import Pool
import numpy as np

def func(a, b):
    print a, b

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return func(*a_b)

def main():
    pool = Pool()
    a=np.arange(0,20)
    b=np.arange(0,20)
    #pool.map(func_star, itertools.izip(a, itertools.repeat(b)))
    pool.map(func_star, itertools.product(a, b))

if __name__=="__main__":
    main()
