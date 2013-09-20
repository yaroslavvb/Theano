#TODO skip if scipy.sparse isn't present.

import numpy

from theano.gof.python25 import any
import theano.sandbox.cuda as cuda_ndarray
import theano.sandbox.gpuarray
from theano.sandbox.gpuarray.sparse import GpuDotCsrDense
import theano.sparse
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
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding(
        'gpuarray')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including(
        'gpuarray')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding(
        'gpuarray')


def test_local_gpu_dot_csr_dense():
    x = theano.sparse.csr_matrix(dtype='float32')
    y = theano.tensor.fmatrix()
    out = theano.sparse.basic._dot(x, y)

    f = theano.function([x, y], out, mode=mode_without_gpu)
    f_gpu = theano.function([x, y], out, mode=mode_with_gpu)
    theano.printing.debugprint(f)
    theano.printing.debugprint(f_gpu)
    assert any(isinstance(x.op, GpuDotCsrDense)
               for x in f_gpu.maker.fgraph.toposort())

    x1 = theano.sparse.tests.test_basic.sparse_random_inputs(
        'csr', (3, 4), out_dtype='float32')[1][0]
    y1 = numpy.random.rand(4, 5).astype('float32')

    for x_val, y_val in [(x1, y1)]:
        res = f(x_val, y_val)
        res_gpu = f_gpu(x_val, y_val)
        utt.assert_allclose(res, res_gpu)

    #Assert not compatible shape raise error
    y_val = numpy.random.rand(5, 5).astype('float32')
    try:
        res_gpu = f_gpu(x_val, y_val)
        assert False
    except ValueError:
        pass


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
"""
    x = theano.sparse.csr_matrix(dtype='float32')
    y = theano.tensor.fmatrix()
    out = theano.sparse.basic._dot(x, y)

    f = theano.function([x, y], out, mode=mode_without_gpu)
    f_gpu = theano.function([x, y], out, mode=mode_with_gpu)
    theano.printing.debugprint(f)
    theano.printing.debugprint(f_gpu)
    assert any(isinstance(x.op, GpuDotCsrDense)
               for x in f_gpu.maker.fgraph.toposort())

    x1 = theano.sparse.tests.test_basic.sparse_random_inputs(
        'csr', (4000, 4000), out_dtype='float32')[1][0]
    y1 = numpy.random.rand(4000, 128).astype('float32')

    for x_val, y_val in [(x1, y1)]:
        res = f(x_val, y_val)
        res_gpu = f_gpu(x_val, y_val)
        utt.assert_allclose(res, res_gpu)