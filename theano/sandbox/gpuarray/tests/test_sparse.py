#TODO skip if scipy.sparse isn't present.
import cPickle
import os

import numpy
import scipy

from theano.gof.python25 import any
import theano.sandbox.cuda as cuda_ndarray
import theano.sandbox.gpuarray
from theano.sandbox.gpuarray.sparse import GpuDotCsrDense, GpuDotCsrCsrCsr
import theano.sparse as sparse
import theano.sparse.tests.test_basic
from theano.tests.unittest_tools import SkipTest
from theano.tests import unittest_tools as utt

if theano.sandbox.gpuarray.pygpu is None:
    raise SkipTest("pygpu not installed")

if not theano.sparse.enable_sparse:
    raise SkipTest('Optional package sparse disabled')

if cuda_ndarray.cuda_available and not theano.sandbox.gpuarray.pygpu_activated:
    if not cuda_ndarray.use.device_number:
        #We should not enable all the use like the flag device=gpu,
        #as many tests don't work in that setup.
        cuda_ndarray.use('gpu',
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False)
    theano.sandbox.gpuarray.init_dev('cuda')

if not theano.sandbox.gpuarray.pygpu_activated:
    raise SkipTest("pygpu disabled")

from pygpu import gpuarray

utt.seed_rng()
rng = numpy.random.RandomState(seed=utt.fetch_seed())


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including(
        'gpuarray')
    mode_with_cuda = theano.compile.mode.get_mode('FAST_RUN').including(
        'gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding(
        'gpuarray')
else:
    mode_with_cuda = theano.compile.mode.get_default_mode().including(
        'gpu')
    mode_with_gpu = theano.compile.mode.get_default_mode().including(
        'gpuarray')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding(
        'gpuarray')


def te_dot(sym_inputs, out, op, inputs, wrong_inputs):
    f = theano.function(sym_inputs, out, mode=mode_without_gpu)
    f_gpu = theano.function(sym_inputs, out, mode=mode_with_gpu)
    #theano.printing.debugprint(f)
    #theano.printing.debugprint(f_gpu)
    assert not any(isinstance(x.op, op)
                   for x in f.maker.fgraph.toposort())
    assert any(isinstance(x.op, op)
               for x in f_gpu.maker.fgraph.toposort())
    res = f(*inputs)
    res_gpu = f_gpu(*inputs)
    if scipy.sparse.issparse(res):
        res = res.todense()
        res_gpu = res_gpu.todense()
    utt.assert_allclose(res, res_gpu)

    #Assert not compatible shape raise error
    y_val = numpy.random.rand(5, 5).astype('float32')
    try:
        res_gpu = f_gpu(*wrong_inputs)
        assert False
    except ValueError:
        pass


def test_local_gpu_dot_csr_dense():
    x = theano.sparse.csr_matrix(dtype='float32')
    y = theano.tensor.fmatrix()
    for op in [
        theano.sparse.basic._dot,
        theano.sparse.basic.structured_dot
    ]:
        out = op(x, y)
        x1 = theano.sparse.tests.test_basic.sparse_random_inputs(
            'csr', (3, 4), out_dtype='float32')[1][0]
        y1 = numpy.random.rand(4, 5).astype('float32')
        y1 = numpy.asfortranarray(y1)
        y_val = numpy.random.rand(5, 5).astype('float32')
        te_dot([x, y], out, GpuDotCsrDense, [x1, y1], [x1, y_val])

        # Test case where the input is f order
        out = op(x, y.T)
        y1 = y1.T
        y1 = numpy.ascontiguousarray(y1)
        te_dot([x, y], out, GpuDotCsrDense, [x1, y1], [x1, y_val])

        # Test case where the input is f order and subsampled
        out = op(x, y.T[:, ::2])
        y1 = numpy.random.rand(4, 10).astype('float32')
        y1 = y1.T
        y1 = numpy.ascontiguousarray(y1)
        te_dot([x, y], out, GpuDotCsrDense, [x1, y1], [x1, y_val])


def test_dot_dense_csc():
    x = theano.tensor.fmatrix()
    y = theano.sparse.csc_matrix(dtype='float32')
    for op in [
        theano.sparse.basic._dot,
        theano.sparse.basic.structured_dot
    ]:
        out = op(x, y)

        x1 = numpy.random.rand(3, 4).astype('float32')
        x1 = numpy.asfortranarray(x1)
        y1 = theano.sparse.tests.test_basic.sparse_random_inputs(
            'csc', (4, 5), out_dtype='float32')[1][0]
        x_val = numpy.random.rand(5, 5).astype('float32')
        te_dot([x, y], out, GpuDotCsrDense, [x1, y1], [x_val, y1])


def test_local_gpu_dot_csr_csr_csr():
    x = theano.sparse.csr_matrix(dtype='float32')
    y = theano.sparse.csr_matrix(dtype='float32')
    out = theano.sparse.basic.true_dot(x, y)

    x1 = theano.sparse.tests.test_basic.sparse_random_inputs(
        'csr', (3, 4), out_dtype='float32')[1][0]
    y1 = theano.sparse.tests.test_basic.sparse_random_inputs(
        'csr', (4, 5), out_dtype='float32')[1][0]
    y_val = numpy.random.rand(5, 5).astype('float32')
    te_dot([x, y], out, GpuDotCsrCsrCsr, [x1, y1], [x1, y_val])

    # Test case where the input is transposed. It shouldn't make a difference
    x = theano.sparse.csr_matrix(dtype='float32')
    y = theano.sparse.csc_matrix(dtype='float32')
    out = theano.sparse.basic.true_dot(x, y.T)
    y1 = y1.T
    te_dot([x, y], out, GpuDotCsrCsrCsr, [x1, y1], [x1, y_val])

    x = theano.sparse.csc_matrix(dtype='float32')
    y = theano.sparse.csr_matrix(dtype='float32')
    out = theano.sparse.basic.true_dot(x.T, y)
    x1 = x1.T
    y1 = y1.T
    te_dot([x, y], out, GpuDotCsrCsrCsr, [x1, y1], [x1, y_val])


def speed():
    u"""
     A dense * B sparse

A = 4000 x 4000 ou 2000 x 2000  ou 1000 x 1000
B = n x 128

B sparse * D sparse --> E dense

D.T comme B
E comme nxn

A, E: dense
B, D: 1% a 10%; tu peux essayer 5%.

$CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS=allow_gc=False,ode=DebugMode,cuda.root=~/lisa/cuda_5.5.22/install/ PYTHONPATH=$PYTHONPATH:~/repos/compyte  LD_LIBRARY_PATH=$MY_PREFIX/lib/:$LD_LIBRARY_PATH LIBRARY_PATH=$MY_PREFIX/lib/:$LIBRARY_PATH CPATH=$CPATH:$MY_PREFIX/include  theano-nose theano/sandbox/gpuarray/tests/test_sparse.py:speed --pdb --pdb-failure -s
Using gpu device 0: GeForce GTX 470
dot(csr, dense)
%sparse nnz    m n   k    cpu(s) gpu(s) speed up(op), speed up(fct), speed up(op, dense cpu/gpu)
<function ascontiguousarray at 0x2999b90>
dense     all 128 1000 1000 0.014 0.001 25.34  4.20
WARNING (theano.gof.cmodule): not all the following op(s) implement c_code_cache_version(). This makes them recompiled for each process.[<theano.sandbox.gpuarray.sparse.GpuDotCsrDense object at 0x8cd6890>]
0.001     110 128 1000 1000 0.001 0.008  0.07  0.01  1.74/ 0.07
0.005     641 128 1000 1000 0.001 0.010  0.10  0.02  1.37/ 0.05
0.010    1332 128 1000 1000 0.002 0.008  0.21  0.03  1.84/ 0.07
0.020    2609 128 1000 1000 0.004 0.008  0.49  0.07  1.80/ 0.07
0.050    6344 128 1000 1000 0.007 0.009  0.79  0.09  1.64/ 0.06
0.100   12724 128 1000 1000 0.013 0.008  1.68  0.18  1.76/ 0.07
dense     all 128 2000 2000 0.047 0.002 22.90  4.49
0.001     268 128 2000 2000 0.001 0.010  0.10  0.01  4.68/ 0.20
0.005    1293 128 2000 2000 0.003 0.016  0.19  0.05  2.86/ 0.12
0.010    2558 128 2000 2000 0.006 0.012  0.51  0.12  4.01/ 0.18
0.020    5099 128 2000 2000 0.012 0.012  0.99  0.24  3.96/ 0.17
0.050   12688 128 2000 2000 0.028 0.014  1.98  0.47  3.36/ 0.15
0.100   25483 128 2000 2000 0.055 0.016  3.52  0.97  3.00/ 0.13
dense     all 128 4000 4000 0.190 0.006 29.62  5.23
0.001     513 128 4000 4000 0.003 0.023  0.13  0.04  8.21/ 0.28
0.005    2556 128 4000 4000 0.012 0.026  0.46  0.18  7.24/ 0.24
0.010    5207 128 4000 4000 0.024 0.029  0.82  0.34  6.58/ 0.22
0.020   10114 128 4000 4000 0.045 0.035  1.30  0.63  5.45/ 0.18
0.050   25534 128 4000 4000 0.125 0.037  3.35  1.69  5.11/ 0.17
0.100   51055 128 4000 4000 0.229 0.053  4.32  2.28  3.57/ 0.12
dense     all 256 4000 4000 0.360 0.012 30.47  8.49
0.001    1034 256 4000 4000 0.007 0.023  0.30  0.13 15.36/ 0.50
0.005    5109 256 4000 4000 0.024 0.030  0.80  0.39 12.20/ 0.40
0.010   10168 256 4000 4000 0.046 0.033  1.41  0.73 10.99/ 0.36
0.020   20365 256 4000 4000 0.091 0.042  2.14  1.24  8.47/ 0.28
0.050   50999 256 4000 4000 0.228 0.050  4.52  2.76  7.14/ 0.23
0.100  101988 256 4000 4000 0.449 0.075  5.97  4.23  4.79/ 0.16
dense     all 128 8000 8000 0.747 0.023 31.88  5.95
0.001     932 128 8000 8000 0.012 0.084  0.14  0.06  8.94/ 0.28
0.005    5227 128 8000 8000 0.046 0.096  0.48  0.23  7.78/ 0.24
0.010   10187 128 8000 8000 0.089 0.106  0.84  0.43  7.04/ 0.22
0.020   20204 128 8000 8000 0.184 0.123  1.50  0.81  6.08/ 0.19
0.050   51174 128 8000 8000 0.508 0.176  2.89  1.82  4.25/ 0.13
0.100  102494 128 8000 8000 0.904 0.211  4.29  2.85  3.55/ 0.11
dense     all 256 8000 8000 1.416 0.046 30.87  9.32
0.001    1957 256 8000 8000 0.020 0.078  0.26  0.11 18.17/ 0.59
0.005   10239 256 8000 8000 0.099 0.104  0.96  0.48 13.61/ 0.44
0.010   20483 256 8000 8000 0.179 0.137  1.31  0.73 10.36/ 0.34
0.020   41057 256 8000 8000 0.355 0.163  2.18  1.33  8.68/ 0.28
0.050  102775 256 8000 8000 0.889 0.252  3.53  2.48  5.62/ 0.18
0.100  205208 256 8000 8000 1.906 0.287  6.63  4.82  4.93/ 0.16
<function asfortranarray at 0x2999c08>
dense     all 128 1000 1000 0.012 0.001 21.53  1.38
0.001     124 128 1000 1000 0.006 0.000 31.21  0.31 64.22/ 2.98
0.005     642 128 1000 1000 0.008 0.000 27.78  0.31 42.36/ 1.97
0.010    1282 128 1000 1000 0.007 0.000 18.59  0.46 31.24/ 1.45
0.020    2575 128 1000 1000 0.009 0.001 13.71  0.23 19.04/ 0.88
0.050    6364 128 1000 1000 0.012 0.001 13.15  0.31 13.06/ 0.61
0.100   12830 128 1000 1000 0.019 0.001 12.51  0.37  8.08/ 0.38
dense     all 128 2000 2000 0.047 0.002 22.87  1.32
0.001     279 128 2000 2000 0.026 0.001 49.48  1.68 88.85/ 3.88
0.005    1321 128 2000 2000 0.028 0.001 24.08  1.45 39.90/ 1.74
0.010    2515 128 2000 2000 0.031 0.002 16.06  1.21 24.41/ 1.07
0.020    5102 128 2000 2000 0.036 0.002 17.91  1.71 23.06/ 1.01
0.050   12803 128 2000 2000 0.053 0.004 12.42  2.17 10.99/ 0.48
0.100   25684 128 2000 2000 0.087 0.006 13.88  2.80  7.47/ 0.33
dense     all 128 4000 4000 0.189 0.006 29.62  0.99
0.001     476 128 4000 4000 0.156 0.001 122.83  5.28 149.01/ 5.03
0.005    2492 128 4000 4000 0.165 0.005 33.56  5.13 38.44/ 1.30
0.010    5121 128 4000 4000 0.176 0.007 25.15  4.75 27.01/ 0.91
0.020   10043 128 4000 4000 0.198 0.012 16.20  4.44 15.48/ 0.52
0.050   25651 128 4000 4000 0.266 0.016 16.57  5.51 11.79/ 0.40
0.100   51500 128 4000 4000 0.380 0.032 11.69  5.58  5.83/ 0.20
dense     all 256 4000 4000 0.360 0.012 30.54  1.80
0.001    1064 256 4000 4000 0.165 0.002 73.71  4.67 160.81/ 5.27
0.005    5097 256 4000 4000 0.177 0.009 20.24  4.44 41.22/ 1.35
0.010   10229 256 4000 4000 0.214 0.012 18.47  4.94 31.07/ 1.02
0.020   20402 256 4000 4000 0.245 0.022 10.90  4.63 16.04/ 0.53
0.050   51077 256 4000 4000 0.382 0.030 12.89  6.55 12.16/ 0.40
0.100  102876 256 4000 4000 0.608 0.054 11.19  7.03  6.62/ 0.22
dense     all 128 8000 8000 0.764 0.023 32.65  0.88
0.001    1053 128 8000 8000 0.748 0.005 142.76  6.78 145.79/ 4.47
0.005    5036 128 8000 8000 0.797 0.020 40.37  6.37 38.68/ 1.18
0.010   10142 128 8000 8000 0.823 0.037 22.27  5.93 20.66/ 0.63
0.020   20379 128 8000 8000 0.911 0.054 16.97  5.68 14.22/ 0.44
0.050   50929 128 8000 8000 1.184 0.105 11.32  5.68  7.30/ 0.22
0.100  102220 128 8000 8000 1.621 0.124 13.05  7.12  6.15/ 0.19
dense     all 256 8000 8000 1.439 0.046 31.36  1.61
0.001    2090 256 8000 8000 0.775 0.009 85.31  6.52 158.32/ 5.05
0.005   10410 256 8000 8000 0.880 0.036 24.64  6.23 40.32/ 1.29
0.010   20462 256 8000 8000 0.917 0.061 14.92  5.48 23.43/ 0.75
0.020   40621 256 8000 8000 1.099 0.092 11.89  5.62 15.58/ 0.50
0.050  102010 256 8000 8000 1.647 0.183  8.99  5.61  7.86/ 0.25
0.100  205099 256 8000 8000 2.667 0.215 12.41  8.27  6.70/ 0.21
dot(dense, csc)
%sparse nnz    m n   k    cpu(s) gpu(s) speed up(op), speed up(fct), speed up(op, dense cpu/gpu)
<function ascontiguousarray at 0x2999b90>
dense     all 128 1000 1000 0.012 0.001 20.85  3.34
0.001     996 128 1000 1000 0.001 0.000  4.43  0.05 65.19/ 3.13
0.005    5091 128 1000 1000 0.002 0.000  6.48  0.09 49.64/ 2.38
0.010    9928 128 1000 1000 0.002 0.000  5.96  0.06 35.16/ 1.69
0.020   20053 128 1000 1000 0.003 0.001  6.70  0.15 22.41/ 1.07
0.050   49876 128 1000 1000 0.007 0.001  7.94  0.29 12.60/ 0.60
0.100   99755 128 1000 1000 0.015 0.001 10.32  0.61  7.92/ 0.38
dense     all 128 2000 2000 0.047 0.002 22.85  4.43
0.001    4051 128 2000 2000 0.002 0.000  5.28  0.19 148.39/ 6.49
0.005   20168 128 2000 2000 0.004 0.001  6.07  0.46 74.54/ 3.26
0.010   39903 128 2000 2000 0.006 0.001  6.20  0.44 44.84/ 1.96
0.020   80233 128 2000 2000 0.012 0.002  6.66  0.77 26.16/ 1.14
0.050  200415 128 2000 2000 0.028 0.004  7.68  1.10 12.76/ 0.56
0.100  399520 128 2000 2000 0.055 0.006  9.89  2.01  8.47/ 0.37
dense     all 128 4000 4000 0.187 0.006 29.32  5.48
0.001   15954 128 4000 4000 0.004 0.001  7.49  1.03 312.69/10.67
0.005   79691 128 4000 4000 0.013 0.002  6.60  1.64 92.75/ 3.16
0.010  159238 128 4000 4000 0.024 0.004  6.30  1.92 48.74/ 1.66
0.020  320259 128 4000 4000 0.046 0.007  6.69  3.11 27.20/ 0.93
0.050  800683 128 4000 4000 0.114 0.014  7.96  3.62 13.06/ 0.45
0.100 1598362 128 4000 4000 0.225 0.025  8.86  5.32  7.37/ 0.25
dense     all 256 4000 4000 0.358 0.012 30.37  8.51
0.001   15895 256 4000 4000 0.011 0.001 10.52  1.22 354.25/11.66
0.005   80271 256 4000 4000 0.029 0.004  7.38  2.22 92.43/ 3.04
0.010  160316 256 4000 4000 0.050 0.007  6.82  3.33 48.43/ 1.59
0.020  318694 256 4000 4000 0.093 0.013  6.88  4.03 26.64/ 0.88
0.050  801644 256 4000 4000 0.255 0.029  8.79  6.24 12.37/ 0.41
0.100 1600025 256 4000 4000 0.456 0.052  8.84  6.94  6.94/ 0.23
dense     all 128 8000 8000 0.806 0.023 34.45  6.40
0.001   63585 128 8000 8000 0.015 0.002  8.39  1.72 447.95/13.00
0.005  320248 128 8000 8000 0.052 0.008  6.66  2.83 102.50/ 2.98
0.010  639628 128 8000 8000 0.113 0.015  7.74  4.40 55.16/ 1.60
0.020 1277676 128 8000 8000 0.199 0.027  7.37  4.62 29.89/ 0.87
0.050 3201147 128 8000 8000 0.472 0.062  7.63  5.61 13.02/ 0.38
0.100 6401100 128 8000 8000 0.975 0.103  9.48  7.04  7.84/ 0.23
dense     all 256 8000 8000 1.416 0.046 30.86  9.10
0.001   64106 256 8000 8000 0.033 0.003  9.68  2.26 411.18/13.32
0.005  320390 256 8000 8000 0.118 0.015  7.65  3.94 91.56/ 2.97
0.010  639310 256 8000 8000 0.219 0.029  7.63  5.02 49.26/ 1.60
0.020 1279465 256 8000 8000 0.412 0.054  7.64  5.72 26.26/ 0.85
0.050 3199623 256 8000 8000 1.076 0.124  8.71  7.20 11.46/ 0.37
0.100 6395132 256 8000 8000 2.015 0.210  9.60  8.09  6.75/ 0.22
<function asfortranarray at 0x2999c08>
dense     all 128 1000 1000 0.012 0.001 21.55  3.40
0.001     984 128 1000 1000 0.001 0.006  0.08  0.01  1.96/ 0.09
0.005    5031 128 1000 1000 0.001 0.007  0.16  0.03  1.69/ 0.08
0.010   10075 128 1000 1000 0.002 0.007  0.25  0.03  1.65/ 0.08
0.020   20122 128 1000 1000 0.003 0.007  0.48  0.06  1.74/ 0.08
0.050   50006 128 1000 1000 0.008 0.007  1.15  0.15  1.78/ 0.08
0.100  100040 128 1000 1000 0.014 0.008  1.75  0.21  1.52/ 0.07
dense     all 128 2000 2000 0.047 0.002 23.00  4.08
0.001    4011 128 2000 2000 0.001 0.006  0.18  0.03  7.41/ 0.32
0.005   20105 128 2000 2000 0.004 0.006  0.67  0.07  7.62/ 0.33
0.010   40044 128 2000 2000 0.006 0.008  0.82  0.13  6.27/ 0.27
0.020   80069 128 2000 2000 0.012 0.011  1.10  0.28  4.30/ 0.19
0.050  200073 128 2000 2000 0.030 0.012  2.48  0.63  3.96/ 0.17
0.100  400544 128 2000 2000 0.054 0.012  4.33  1.27  3.79/ 0.16
dense     all 128 4000 4000 0.193 0.006 30.15  5.10
0.001   15845 128 4000 4000 0.003 0.007  0.47  0.10 27.33/ 0.91
0.005   79642 128 4000 4000 0.012 0.008  1.56  0.43 25.29/ 0.84
0.010  160122 128 4000 4000 0.023 0.010  2.45  0.94 20.30/ 0.67
0.020  320339 128 4000 4000 0.045 0.013  3.54  1.67 15.28/ 0.51
0.050  799471 128 4000 4000 0.118 0.020  5.93  3.62  9.67/ 0.32
0.100 1600523 128 4000 4000 0.235 0.032  7.42  4.99  6.10/ 0.20
dense     all 256 4000 4000 0.358 0.012 30.33  7.35
0.001   15872 256 4000 4000 0.006 0.008  0.79  0.45 46.23/ 1.52
0.005   79963 256 4000 4000 0.025 0.011  2.36  1.33 34.13/ 1.13
0.010  160045 256 4000 4000 0.045 0.014  3.24  2.07 25.84/ 0.85
0.020  319143 256 4000 4000 0.088 0.020  4.49  3.02 18.26/ 0.60
0.050  799789 256 4000 4000 0.218 0.035  6.18  4.75 10.13/ 0.33
0.100 1600608 256 4000 4000 0.474 0.058  8.14  6.19  6.15/ 0.20
dense     all 128 8000 8000 0.767 0.023 32.79  5.85
0.001   64045 128 8000 8000 0.012 0.009  1.36  0.75 89.44/ 2.73
0.005  320039 128 8000 8000 0.052 0.016  3.33  2.13 49.36/ 1.51
0.010  640493 128 8000 8000 0.104 0.021  4.97  3.17 36.79/ 1.12
0.020 1283456 128 8000 8000 0.209 0.038  5.49  4.03 20.08/ 0.61
0.050 3199762 128 8000 8000 0.468 0.068  6.84  5.04 11.20/ 0.34
0.100 6395342 128 8000 8000 0.982 0.111  8.88  6.69  6.93/ 0.21
dense     all 256 8000 8000 1.418 0.046 30.90  7.77
0.001   64152 256 8000 8000 0.023 0.015  1.55  0.92 94.73/ 3.07
0.005  319075 256 8000 8000 0.104 0.023  4.50  2.91 61.25/ 1.98
0.010  639807 256 8000 8000 0.223 0.036  6.11  4.32 38.94/ 1.26
0.020 1280282 256 8000 8000 0.399 0.061  6.52  5.16 23.17/ 0.75
0.050 3200336 256 8000 8000 0.988 0.133  7.40  6.22 10.62/ 0.34
0.100 6399484 256 8000 8000 1.969 0.219  9.00  7.67  6.48/ 0.21

%sparse nnz    m n   k    cpu(s) gpu(s) sp_cpu/sp_gpu, sp_cpu_fct/sp_gpu_fct, d_cpu/sp_gpu, d_cpu/sp_cpu d_gpu/sp_gpu
<function asfortranarray at 0x11a4c08>
dense      all 128 1024 3072 0.047 0.002 29.98  1.56
WARNING (theano.gof.cmodule): not all the following op(s) implement c_code_cache_version(). This makes them recompiled for each process.[<theano.sandbox.gpuarray.sparse.GpuDotCsrDense object at 0x7d3e050>]
0.0005     191 128 1024 3072 0.028 0.000 99.89  1.76 167.73  1.68  5.59
0.0010     414 128 1024 3072 0.021 0.000 58.36  1.22 131.01  2.24  4.37
0.0050    1975 128 1024 3072 0.024 0.001 24.97  1.33 49.71  1.99  1.66
0.0100    3976 128 1024 3072 0.027 0.002 16.91  1.17 30.00  1.77  1.00
0.0500   19828 128 1024 3072 0.045 0.003 14.74  1.83 15.47  1.05  0.52
dense      all 128 2048 3072 0.075 0.003 24.37  0.66
0.0005     168 128 2048 3072 0.097 0.000 199.46  3.05 154.71  0.78  6.35
0.0010     409 128 2048 3072 0.106 0.001 145.00  3.56 103.11  0.71  4.23
0.0050    1877 128 2048 3072 0.116 0.002 64.74  3.91 42.05  0.65  1.73
0.0100    3879 128 2048 3072 0.111 0.003 36.66  2.28 24.92  0.68  1.02
0.0500   19629 128 2048 3072 0.145 0.006 23.51  2.61 12.18  0.52  0.50
dense      all 128 4096 3072 0.152 0.005 30.13  0.54
0.0005     181 128 4096 3072 0.264 0.001 392.72 10.59 225.86  0.58  7.50
0.0010     398 128 4096 3072 0.266 0.001 240.66  8.40 137.23  0.57  4.56
0.0050    1966 128 4096 3072 0.273 0.004 77.61  8.52 43.19  0.56  1.43
0.0100    3902 128 4096 3072 0.290 0.006 48.04  9.12 25.17  0.52  0.84
0.0500   19789 128 4096 3072 0.391 0.012 31.47  9.68 12.24  0.39  0.41
dense      all 128 8192 3072 0.346 0.009 36.97  0.60
0.0005     202 128 8192 3072 0.509 0.001 447.10 11.82 303.46  0.68  8.21
0.0010     388 128 8192 3072 0.510 0.002 280.15 11.16 189.68  0.68  5.13
0.0050    1916 128 8192 3072 0.522 0.007 78.26 10.40 51.82  0.66  1.40
0.0100    3762 128 8192 3072 0.551 0.012 47.88  9.21 30.04  0.63  0.81
0.0500   19640 128 8192 3072 0.788 0.024 32.34 11.00 14.18  0.44  0.38
dense      all 512 8192 3072 1.146 0.036 32.02  1.88
0.0005     783 512 8192 3072 0.530 0.003 209.34  8.88 452.60  2.16 14.13
0.0010    1582 512 8192 3072 0.563 0.004 127.24 10.01 258.89  2.03  8.08
0.0050    7793 512 8192 3072 0.589 0.018 32.34  8.22 62.93  1.95  1.97
0.0100   15637 512 8192 3072 0.700 0.036 19.52  7.81 31.94  1.64  1.00
0.0500   78607 512 8192 3072 1.217 0.089 13.64  8.42 12.85  0.94  0.40
dense      all 128 16384 3072 0.609 0.018 33.55  0.53
0.0005     205 128 16384 3072 1.022 0.002 510.01 11.16 303.60  0.60  9.05
0.0010     387 128 16384 3072 1.064 0.003 308.71 11.27 176.55  0.57  5.26
0.0050    2022 128 16384 3072 1.052 0.014 76.07 10.48 43.99  0.58  1.31
0.0100    3895 128 16384 3072 1.121 0.024 46.04  9.76 25.00  0.54  0.74
0.0500   19709 128 16384 3072 1.523 0.048 31.51 11.12 12.59  0.40  0.38
dense      all 512 16384 3072 2.230 0.071 31.37  1.80
0.0005     762 512 16384 3072 1.125 0.005 236.96  8.78 469.88  1.98 14.98
0.0010    1537 512 16384 3072 1.081 0.008 135.88  8.38 280.32  2.06  8.94
0.0050    7850 512 16384 3072 1.223 0.038 32.06  7.75 58.44  1.82  1.86
0.0100   15707 512 16384 3072 1.380 0.071 19.47  7.36 31.45  1.62  1.00
0.0500   78774 512 16384 3072 2.434 0.179 13.63  8.20 12.48  0.92  0.40
dense      all 128 1024 1024 0.013 0.001 22.67  1.35
0.0005      61 128 1024 1024 0.007 0.000 36.92  1.65 68.46  1.85  3.02
0.0010     135 128 1024 1024 0.007 0.000 33.63  1.31 61.46  1.83  2.71
0.0050     651 128 1024 1024 0.008 0.000 24.97  0.96 42.08  1.69  1.86
0.0100    1338 128 1024 1024 0.008 0.000 19.43  1.15 30.12  1.55  1.33
0.0500    6559 128 1024 1024 0.014 0.001 14.15  1.89 13.32  0.94  0.59
dense      all 128 2048 2048 0.050 0.002 24.03  0.71
0.0005     141 128 2048 2048 0.060 0.000 130.71  3.17 108.60  0.83  4.52
0.0010     266 128 2048 2048 0.061 0.001 110.07  3.34 90.98  0.83  3.79
0.0050    1354 128 2048 2048 0.063 0.001 48.59  2.73 38.75  0.80  1.61
0.0100    2513 128 2048 2048 0.066 0.002 33.01  2.78 24.87  0.75  1.04
0.0500   13146 128 2048 2048 0.092 0.005 18.61  2.66 10.13  0.54  0.42
dense      all 128 4096 4096 0.205 0.007 30.61  0.53
0.0005     265 128 4096 4096 0.354 0.001 380.02  9.43 220.08  0.58  7.19
0.0010     541 128 4096 4096 0.361 0.001 260.76 10.05 148.01  0.57  4.83
0.0050    2649 128 4096 4096 0.351 0.005 65.81  8.66 38.36  0.58  1.25
0.0100    5208 128 4096 4096 0.375 0.007 51.02  9.07 27.84  0.55  0.91
0.0500   26234 128 4096 4096 0.477 0.019 25.72  8.81 11.03  0.43  0.36
dense      all 128 8192 8192 0.806 0.025 32.64  0.53
0.0005     562 128 8192 8192 1.442 0.003 461.07 12.28 257.69  0.56  7.89
0.0010    1046 128 8192 8192 1.402 0.005 264.18 11.44 151.89  0.57  4.65
0.0050    5245 128 8192 8192 1.517 0.021 70.82 11.11 37.63  0.53  1.15
0.0100   10568 128 8192 8192 1.459 0.038 38.64 10.00 21.35  0.55  0.65
0.0500   52619 128 8192 8192 1.838 0.109 16.89  8.22  7.41  0.44  0.23
dense      all 512 8192 8192 2.934 0.095 30.79  1.85
0.0005    2156 512 8192 8192 1.455 0.009 161.25 11.31 325.05  2.02 10.56
0.0010    4248 512 8192 8192 1.401 0.017 83.81 10.81 175.55  2.09  5.70
0.0050   21104 512 8192 8192 1.552 0.066 23.69  8.59 44.78  1.89  1.45
0.0100   41755 512 8192 8192 1.825 0.123 14.81  7.63 23.81  1.61  0.77
0.0500  209821 512 8192 8192 3.215 0.298 10.79  7.77  9.85  0.91  0.32

   #dot(vector with zero, matrix) speed up vs vector without zero
   #StructuredDot() faster then dot, via opt op
   #test usmm vs dot speed

"""
    prob = [0.0005, 0.001,
            0.005, 0.01,
            0.05]
    pkl_file = "test_sparse.py.speed_result.pkl"
    pkl_file = os.path.join(os.path.split(__file__)[0], pkl_file)
    results = {}
    if os.path.exists(pkl_file):
        pkl_f = open(pkl_file, 'r')
        try:
            f = cPickle.load(pkl_f)
            results = f
        except EOFError:
            pass
        finally:
            pkl_f.close()

    #m = minibatch, 10-20
    #n = n hid
    #k = n input
    shapes = [
        # First layer, 3*1024(rgb or 32x32 images, svhn) input size
        # Output layer 10 classes
        (128, 1*1024, 3*1024),
        (128, 2*1024, 3*1024),
        (128, 4*1024, 3*1024),
        (128, 8*1024, 3*1024),
        (512, 8*1024, 3*1024),
        (128, 16*1024, 3*1024),
        (512, 16*1024, 3*1024),
        (128, 24*1024, 3*1024),
        (512, 24*1024, 3*1024),
        (128, 32*1024, 3*1024),
        (512, 32*1024, 3*1024),
        (128, 64*1024, 3*1024),
        (512, 64*1024, 3*1024),
        #Hiden layer to hiden layer
        (128, 1*1024, 1*1024),
        (128, 2*1024, 2*1024),
        (128, 4*1024, 4*1024),
        (128, 8*1024, 8*1024),
        (512, 8*1024, 8*1024),
        (128, 8*1024, 16*1024),
        (512, 8*1024, 16*1024),
        (128, 16*1024, 16*1024),
        (512, 16*1024, 16*1024),
        #Output layer
        (128, 10, 1*1024),
        (128, 10, 2*1024),
        (128, 10, 4*1024),
        (128, 10, 8*1024),
        (512, 10, 8*1024),
        (128, 10, 16*1024),
        (512, 10, 16*1024),
        (128, 10, 32*1024),
        (512, 10, 32*1024),
        (128, 10, 64*1024),
        (512, 10, 64*1024),
    ]

    def time(sym_inputs, out, op, inputs,
             m_gpu=mode_with_gpu, results={}, key=None):
        keyc = key + ('cpu',)
        keyg = key + ('gpu',)
        if keyc in results:
            c = results[keyc]
            res = None
        else:
            c = theano.compile.profiling.ProfileStats(atexit_print=False)
            f = theano.function(sym_inputs, out,
                                mode=mode_without_gpu, profile=c)
            for i in range(10):
                res = f(*inputs)
            results[keyc] = c
        if keyg in results:
            g = results[keyg]
            gpu_res = None
        else:
            g = theano.compile.profiling.ProfileStats(atexit_print=False)
            f = theano.function(sym_inputs, out,
                                mode=m_gpu, profile=g)
            for i in range(100):
                res_gpu = f(*inputs)
            results[keyg] = g

        if res is not None and res_gpu is not None:
            utt.assert_allclose(res, res_gpu)

        return c, g

    print "%sparse    nnz   m n   k    cpu(s) gpu(s) sp_cpu/sp_gpu, sp_cpu_fct/sp_gpu_fct, d_cpu/sp_gpu, d_cpu/sp_cpu d_gpu/sp_gpu"

    def usmm(x, y):
        return theano.sparse.opt.usmm(1., x, y, 0.)

    def usmm_csc_dense(x, y):
        return theano.sparse.opt.usmm_csc_dense(
            1., sparse.csm_data(x),
            sparse.csm_indices(x),
            sparse.csm_indptr(x),
            sparse.csm_shape(x)[0], y,
            theano.tensor.zeros((x.shape[0], y.shape[1]), dtype=x.dtype))

    for op, f1, f2 in [
        ((usmm, theano.sparse.Usmm), 'csr', 'dense'),
        ((usmm_csc_dense, theano.sparse.opt.UsmmCscDense), 'csc', 'dense'),
#        (theano.sparse.basic.StructuredDot, 'csr', 'dense'),
        # The first input must be sparse.
        #(theano.sparse.basic.StructuredDot, 'dense', 'csc'),
        (theano.sparse.basic.Dot, 'csr', 'dense'),
        (theano.sparse.basic.Dot, 'dense', 'csc'),
    ]:
        if isinstance(op, tuple):
            op_call = op[0]
            op = op[1]
        else:
            op_call = op()
        print "%s(%s, %s)" % (op, f1, f2)
        x = theano.sparse.tests.test_basic.sparse_random_inputs(
            f1, (10, 10), out_dtype='float32')[0][0]
        xd = theano.tensor.fmatrix()
        y = theano.sparse.tests.test_basic.sparse_random_inputs(
            f2, (10, 10), out_dtype='float32')[0][0]
        yd = theano.tensor.fmatrix()
        out = op_call(x, y)
        outd = theano.dot(xd, yd)
        #TODO uncomment
        for order in [numpy.ascontiguousarray,
                      numpy.asfortranarray]:
            print order
            for m, n, k in shapes:
                if (k * n) * 4 > 350*2**20:
                    print "skip shape m,n,k:", m, n, k
                    continue
                x1 = numpy.random.rand(m, k).astype('float32')
                y1 = numpy.random.rand(k, n).astype('float32')
                x1 = order(x1)  # order(y1) don't always raise an error.
                y1 = order(y1)
                key = (theano.tensor.dot, 'dense', 'dense', order, m, n, k)
                cd, gd = time([xd, yd], outd, None, [x1, y1],
                              m_gpu=mode_with_cuda, results=results, key=key)
                cd_t = cd.class_time()[theano.tensor.blas.Dot22]
                gd_t = gd.class_time()[theano.sandbox.cuda.blas.GpuDot22]
                cd_i = cd.class_callcount()[theano.tensor.blas.Dot22]
                gd_i = gd.class_callcount()[theano.sandbox.cuda.blas.GpuDot22]
                print "dense      all %d %d %d %.3f %.3f %6.2f %6.2f" % (
                    m, n, k, cd_t/cd_i, gd_t/gd_i, (cd_t/cd_i) / (gd_t/gd_i),
                    (cd.vm_call_time/cd_i)/(gd.vm_call_time/gd_i))
                for p in prob:
                    key = (op, f1, f2, order, m, n, k, p)
                    if "dense" != f1:
                        x1 = sparse.tests.test_basic.sparse_random_inputs(
                            f1, (m, k), out_dtype='float32', p=p)[1][0]
                    if "dense" != f2:
                        y1 = sparse.tests.test_basic.sparse_random_inputs(
                            f2, (k, n), out_dtype='float32', p=p)[1][0]
                    if hasattr(x1, 'nnz'):
                        nnz = x1.nnz
                    elif hasattr(y1, 'nnz'):
                        nnz = y1.nnz
                    else:
                        raise Exception()
                    try:
                        c, g = time([x, y], out, None, [x1, y1],
                                    results=results, key=key)
                    except Exception, ex:
                        print repr(ex)
                        print "The previous exception happened with config"
                        print "%.4f %7d %d %d %d" % (
                            p, nnz, m, n, k)
                        continue
                    c_t = c.class_time()[op]
                    c_i = c.class_callcount()[op]
                    if GpuDotCsrDense in g.class_time():
                        g_t = g.class_time()[GpuDotCsrDense]
                        g_i = g.class_callcount()[GpuDotCsrDense]
                    else:
                        g_t = -1
                        g_i = 1
                    print "%.4f %7d %d %d %d %.3f %.3f %6.2f %6.2f %6.2f %6.2f %6.2f" % (
                        p, nnz,
                        m, n, k, c_t/c_i, g_t/g_i,
                        (c_t/c_i) / (g_t/g_i),  # sp_cpu/sp_gpu
                        (c.vm_call_time/c_i)/(g.vm_call_time/g_i),  # sp_cpu_fct/sp_gpu_fct
                        (cd_t/cd_i)/(g_t/g_i),  # d_cpu/sp_gpu
                        (cd_t/cd_i)/(c_t/c_i),  # d_cpu/sp_cpu
                        (gd_t/gd_i)/(g_t/g_i),  # d_gpu/sp_gpu
                        )

                pkl_f = open(pkl_file, 'w')
                cPickle.dump(results, pkl_f)
                pkl_f.close()