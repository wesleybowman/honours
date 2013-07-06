import multiprocessing

def do_calculation(data,a=1,b=2):
    return data * 2,a+1,b+1
    


if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)
    
    inputs = list(range(10))
    print 'Input   :', inputs
    
    builtin_outputs = map(do_calculation, inputs,inputs,inputs)
    print 'Built-in:', builtin_outputs
    
    pool_outputs = pool.map(functools.partial(do_calculation,data=1),inputs)
    print 'Pool    :', pool_outputs