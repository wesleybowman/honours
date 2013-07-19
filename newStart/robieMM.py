nner_loops = 251
hl = int(m/inner_loops)
print "starting loops"
for x in xrange(n):
        for i in xrange(inner_loops):
                print (x,i)
                ksiTemp = KSIdotR[x,hl*i:hl*(i+1)]
                a= datetime.now()
                neTemp = ksiTemp[:,None,None]
                arg = ne.evaluate("neTemp/KSInorm")
                temp = ne.evaluate("holo * exp(1j * k * arg)")
                temp2 = ne.evaluate("sum(temp, axis=2)")
                reconstruction[x,hl*i:hl*(i+1)]=temp2.sum(axis=1)
                b=datetime.now()
                print (b-a).seconds, (b-a).microseconds

reconstruction = ne.evaluate("reconstruction *distX * distY")

reconstruction.dump('reconstruction.dat')
