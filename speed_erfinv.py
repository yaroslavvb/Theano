import time

import numpy as np

import theano
import theano.tensor as tt
v = tt.fvector()

for p in range(10, 28):
    N = 2 ** p
    d = theano.shared(np.random.rand(N).astype("float32"))
    f = theano.function([], [], updates=([d, tt.erfinv(d)],), name=str(p))
    if p == 10:
        print f.maker.fgraph.toposort()

    t0 = time.time()
    for i in range(100):
        f()
    theano.sandbox.cuda.synchronize()
    t1 = time.time()
    del f, d
    print N, t1 - t0
