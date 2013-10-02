import numpy
import scipy.sparse

import theano
from theano import config, gof, sparse
from theano.gof.python25 import any
import theano.sandbox.gpuarray as gpuarray
from theano.sandbox.gpuarray.basic_ops import (
    as_gpuarray_variable, host_from_gpu, gpu_from_host, GpuArrayType)
from theano.sandbox.gpuarray.opt import register_opt
from theano.sparse.basic import _is_sparse_variable


class GpuDotCsrDense(gof.Op):
    """
    Gpu version of: dot(sparse csr, dense) -> dense

    call CUSPARSE

    TODO: assert that the input matrix is sorted and uniq
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x_val, x_ind, x_ptr, x_shape, y):
        x_val = as_gpuarray_variable(x_val)
        x_ind = as_gpuarray_variable(x_ind)
        x_ptr = as_gpuarray_variable(x_ptr)
        x_shape = theano.tensor.as_tensor_variable(x_shape)
        y = as_gpuarray_variable(y)
        assert numpy.intc == numpy.int32, ("We suppose that the c type 'int'"
                                           " is equivalent to int32, but this"
                                           " is false!")
        assert x_val.dtype == 'float32' and x_val.ndim == 1
        assert x_ind.dtype == 'int32' and x_ind.ndim == 1
        assert x_ptr.dtype == 'int32' and x_ptr.ndim == 1
        assert x_shape.dtype == 'int32' and x_shape.ndim == 1
        assert y.dtype == 'float32' and y.ndim == 2

        bz = (False, y.type.broadcastable[1])
        out = GpuArrayType(broadcastable=bz,
                           dtype=x_val.dtype)()
        return gof.Apply(self, [x_val, x_ind, x_ptr, x_shape, y],
                         [out])

    def perform(self, node, inputs, out):
        inputs = [numpy.asarray(x) for x in inputs]
        x = scipy.sparse.csr_matrix(tuple(inputs[:3]), shape=inputs[3])
        y = inputs[-1]
        out[0][0] = gpuarray.pygpu.asarray(x * y)

    def c_code_cache_version(self):
        return ()
        return (1,)

    def c_headers(self):
        # std.cout doesn't require the '%' symbol to print stuff...
        # so it works much better with python's string-substitution stuff.
        return ['<cuda_runtime.h>', '<cusparse_v2.h>',
                '<compyte/extension.h>',
                'cuda.h',
                'compyte/util.h'  # needed for GpuArray_ITEMSIZE
        ]

    def c_libraries(self):
        return ['cusparse', 'cudart']

    # code_cache_version is built by subclasses from
    #  build_gemm_version

    def _c_header_dirs(self):
        ret = []
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret

    def _c_lib_dirs(self):
        ret = []
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'lib'))
        return ret

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx);
        CUdeviceptr (*cuda_get_ptr)(gpudata *g);
        cusparseHandle_t cusparseHandle = 0;
        cusparseMatDescr_t descr = 0;
        """

    def c_init_code(self):
        return ['cuda_get_ctx = (CUcontext (*)(void *ctx))compyte_get_extension("cuda_get_ctx");',
                'cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))compyte_get_extension("cuda_get_ptr");',
"""
        // We try to init it here to don't init it in the
        //Theano fct call to don't mess with profiling.
        // But as we can't raise error here, we try again
        // in c_code, to raise the good error.
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);
        if (cusparseStatus == CUSPARSE_STATUS_SUCCESS) {
          assert(cusparseHandle != 0);
          /* Get the mat description */
          cusparseStatus = cusparseCreateMatDescr(&descr);
          if (cusparseStatus == CUSPARSE_STATUS_SUCCESS) {
            assert(descr != 0);
            cusparseStatus = cusparseSetMatType(descr,
                             CUSPARSE_MATRIX_TYPE_GENERAL);
            if (cusparseStatus == CUSPARSE_STATUS_SUCCESS) {
              cusparseStatus = cusparseSetMatIndexBase(descr,
                             CUSPARSE_INDEX_BASE_ZERO);
              if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
                descr = 0;
              }
            }else{
                descr = 0;
            }
          }else{
            descr = 0;
          }
        }else{
          cusparseHandle = 0;
        }
                """

        ]

    def c_code(self, node, name, inputs, outputs, sub):
        x_val, x_ind, x_ptr, x_shape, y = inputs
        out, = outputs
        fail = sub['fail']
        code = """
    const float alpha = 1;
    const float beta = 0;

    cusparseStatus_t cusparseStatus;
    int %(name)serr;
    size_t x_shp0 = ((dtype_%(x_shape)s *)PyArray_DATA(%(x_shape)s))[0];
    size_t x_shp1 = ((dtype_%(x_shape)s *)PyArray_DATA(%(x_shape)s))[1];
    size_t out_dims[2] = {x_shp0,
                          %(y)s->ga.dimensions[1]};
    PyGpuArrayObject* usable_y = NULL;

    if (x_shp1 != %(y)s->ga.dimensions[0])
    {
        PyErr_Format(
                PyExc_ValueError,
                "GpuDotCsrDense: input shape incompatible:"
                " (%%d, %%d) and (%%d, %%d)",
                x_shp0, x_shp1,
                %(y)s->ga.dimensions[0], %(y)s->ga.dimensions[1]);
        %(fail)s
    }

    /* Get handle to the CUSPARSE context */
    if (cusparseHandle == 0){
        printf("create cusparse handle\\n");
        cusparseStatus = cusparseCreate(&cusparseHandle);

        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrDense: cusparseCreate() failed");
            %(fail)s
        }
        assert(cusparseHandle != 0);
    }
    /* Get the mat description */
    if (descr == 0){
        printf("create cusparse desct\\n");

        cusparseStatus = cusparseCreateMatDescr(&descr);

        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrDense: cusparseCreateMatDescr() failed");
            cusparseDestroy(cusparseHandle);
            cusparseHandle = 0;
            %(fail)s
        }
        assert(descr != 0);
        cusparseStatus = cusparseSetMatType(descr,
                             CUSPARSE_MATRIX_TYPE_GENERAL);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
             PyErr_SetString(
             PyExc_RuntimeError,
                        "GpuDotCsrDense: cusparseSetMatType() failed");
             cusparseDestroyMatDescr(descr);
             cusparseDestroy(cusparseHandle);
             descr = 0;
             cusparseHandle = 0;
             %(fail)s
        }

        cusparseStatus = cusparseSetMatIndexBase(descr,
                             CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "GpuDotCsrDense: cusparseSetMatIndexBase() failed");
            cusparseDestroyMatDescr(descr);
            cusparseDestroy(cusparseHandle);
             descr = 0;
             cusparseHandle = 0;
            %(fail)s
        }
    }
    //if nvcc >= 5.5, use cusparseScsrmm2 call.
    if (%(y)s->ga.strides[0] != GpuArray_ITEMSIZE(&%(y)s->ga))
    {
        usable_y = pygpu_copy(%(y)s, GA_F_ORDER);
        if (NULL == usable_y){
            cusparseDestroyMatDescr(descr);
            cusparseDestroy(cusparseHandle);
            descr = 0;
            cusparseHandle = 0;
            //pygpu_copy should have set the error message.
            %(fail)s
        }
    }else{
        usable_y = %(y)s;
        Py_INCREF(usable_y);
    }

    //TODO reuse!
    Py_XDECREF(%(out)s);
    %(out)s = pygpu_empty(2, out_dims, %(y)s->ga.typecode, GA_F_ORDER,
        pygpu_default_context(),
        (PyObject *)&PyGpuArrayType);
    if (%(out)s == NULL) {
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        Py_CLEAR(usable_y);
        // pygpu_empty should have set an error message
        %(fail)s
    }
#if 1
        cusparseStatus = cusparseScsrmm(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            out_dims[0], out_dims[1], x_shp1,
            %(x_val)s->ga.dimensions[0], &alpha, descr,
            (float*) cuda_get_ptr(%(x_val)s->ga.data),
            (int*)cuda_get_ptr(%(x_ptr)s->ga.data),
            (int*)cuda_get_ptr(%(x_ind)s->ga.data),
            (float*)cuda_get_ptr(usable_y->ga.data),
            usable_y->ga.strides[1]/GpuArray_ITEMSIZE(&usable_y->ga), // ldb
            &beta,
            (float*)cuda_get_ptr(%(out)s->ga.data),
            out_dims[0]); //ldc suppose out is f contiguous
#else
        cusparseStatus = cusparseScsrmm2(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            out_dims[0], out_dims[1], x_shp1,
            %(x_val)s->ga.dimensions[0], &alpha, descr,
            (float*) cuda_get_ptr(%(x_val)s->ga.data),
            (int*)cuda_get_ptr(%(x_ptr)s->ga.data),
            (int*)cuda_get_ptr(%(x_ind)s->ga.data),
            (float*)cuda_get_ptr(usable_y->ga.data),
            usable_y->ga.strides[1]/GpuArray_ITEMSIZE(usable_y), // ldb
            &beta,
            (float*)cuda_get_ptr(%(out)s->ga.data),
            out_dims[0]); //ldc suppose out is f contiguous
#endif

    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        Py_CLEAR(%(out)s);
        Py_CLEAR(usable_y);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        char * err_msg;
        if (cusparseStatus == CUSPARSE_STATUS_NOT_INITIALIZED){
            err_msg = "CUSPARSE_STATUS_NOT_INITIALIZED";
        }else if (cusparseStatus == CUSPARSE_STATUS_ALLOC_FAILED){
            err_msg = "CUSPARSE_STATUS_ALLOC_FAILED";
        }else if (cusparseStatus == CUSPARSE_STATUS_INVALID_VALUE){
            err_msg = "CUSPARSE_STATUS_INVALID_VALUE";
        }else if (cusparseStatus == CUSPARSE_STATUS_ARCH_MISMATCH){
            err_msg = "CUSPARSE_STATUS_ARCH_MISMATCH";
        }else if (cusparseStatus == CUSPARSE_STATUS_EXECUTION_FAILED){
            err_msg = "CUSPARSE_STATUS_EXECUTION_FAILED";
        }else if (cusparseStatus == CUSPARSE_STATUS_INTERNAL_ERROR){
            err_msg = "CUSPARSE_STATUS_INTERNAL_ERROR";
        }else if (cusparseStatus == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED){
            err_msg = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        }else{
            err_msg = "Unknow error code";
        }
        PyErr_Format(
                PyExc_RuntimeError,
                "GpuDotCsrDense: cusparseScsrmm[2]() returned status %%d (%%s)",
                cusparseStatus, err_msg);
        %(fail)s
    }
        """ % locals()
        if config.gpuarray.sync:
            code += "GpuArray_sync(%(out)s);" % locals()
        code += """
    //TODO remove!!!
    cudaThreadSynchronize();

    if (cudaSuccess != cudaGetLastError())
    {
        Py_CLEAR(%(out)s);
        Py_CLEAR(usable_y);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrDense: cudaGetLastError() failed after cusparseScsrmm()");
        %(fail)s
    }

    Py_CLEAR(usable_y);
        """ % locals()
        return code


@register_opt()
@gof.local_optimizer([sparse.basic._dot])
def local_gpu_dot_csr_dense(node):
    """
    move to gpu sparse.[Structured]Dot(csr, dense) and sparse.Dot(dense, csc)
    """
    # check for sparse.Dot(csr, dense)
    if (isinstance(node.op, (sparse.Dot, sparse.StructuredDot)) and
        _is_sparse_variable(node.inputs[0]) and
        not _is_sparse_variable(node.inputs[1])):
        if not any([i.owner and
                    i.owner.op == host_from_gpu
                    for i in node.inputs]):
            return False

        a, b = node.inputs
        if (a.type.format == 'csr' and a.dtype == b.dtype
            and a.dtype == 'float32'):

            a_val, a_ind, a_ptr, a_shape = sparse.csm_properties(a)
            b = gpu_from_host(b)
            out = GpuDotCsrDense()(a_val, a_ind, a_ptr, a_shape, b)
            return [host_from_gpu(out)]
    # check for sparse.Dot(dense, csc)
    elif (isinstance(node.op, sparse.Dot) and
        _is_sparse_variable(node.inputs[1]) and
        not _is_sparse_variable(node.inputs[0])):
        if not any([i.owner and
                    i.owner.op == host_from_gpu
                    for i in node.inputs]):
            return False

        a, b = node.inputs
        if (b.type.format == 'csc' and a.dtype == b.dtype
            and a.dtype == 'float32'):
            a = gpu_from_host(a.T)
            b_val, b_ind, b_ptr, b_shape = sparse.csm_properties(b.T)
            # The .T introduce the host_from_gpu
            out = GpuDotCsrDense()(b_val, b_ind, b_ptr, b_shape, a).T
            return [out]
    return False


class GpuDotCsrCsrCsr(gof.Op):
    """
    GPU version of: dot(csr, csr)->csr

    call CUSPARSE

    TODO: assert that the input matrix is sorted and uniq
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x_val, x_ind, x_ptr, x_shape,
                  y_val, y_ind, y_ptr, y_shape):
        x_val = as_gpuarray_variable(x_val)
        x_ind = as_gpuarray_variable(x_ind)
        x_ptr = as_gpuarray_variable(x_ptr)
        x_shape = theano.tensor.as_tensor_variable(x_shape)
        y_val = as_gpuarray_variable(y_val)
        y_ind = as_gpuarray_variable(y_ind)
        y_ptr = as_gpuarray_variable(y_ptr)
        y_shape = theano.tensor.as_tensor_variable(y_shape)

        assert numpy.intc == numpy.int32, ("We suppose that the c type 'int'"
                                           " is equivalent to int32, but this"
                                           " is false!")
        assert x_val.dtype == 'float32' and x_val.ndim == 1
        assert x_ind.dtype == 'int32' and x_ind.ndim == 1
        assert x_ptr.dtype == 'int32' and x_ptr.ndim == 1
        assert x_shape.dtype == 'int32' and x_shape.ndim == 1
        assert y_val.dtype == 'float32' and y_val.ndim == 1
        assert y_ind.dtype == 'int32' and y_ind.ndim == 1
        assert y_ptr.dtype == 'int32' and y_ptr.ndim == 1
        assert y_shape.dtype == 'int32' and y_shape.ndim == 1

        bz = (False,)
        out_val = GpuArrayType(broadcastable=bz,
                               dtype=x_val.dtype)()
        out_ind = GpuArrayType(broadcastable=bz,
                               dtype='int32')()
        out_ptr = GpuArrayType(broadcastable=bz,
                               dtype='int32')()
        out_shape = theano.tensor.ivector()
        return gof.Apply(self, [x_val, x_ind, x_ptr, x_shape,
                                y_val, y_ind, y_ptr, y_shape],
                         [out_val, out_ind, out_ptr, out_shape])

    def perform(self, node, inputs, out):
        inputs = [numpy.asarray(x) for x in inputs]
        x = scipy.sparse.csr_matrix(tuple(inputs[:3]), shape=inputs[3])
        y = scipy.sparse.csr_matrix(tuple(inputs[4:7]), shape=inputs[7])
        o = x * y
        out[0][0] = gpuarray.pygpu.asarray(o.data)
        out[1][0] = gpuarray.pygpu.asarray(o.indices)
        out[2][0] = gpuarray.pygpu.asarray(o.indptr)
        out[3][0] = theano._asarray(o.shape, dtype='int32')

    def c_code_cache_version(self):
        return ()
        return (1,)

    def c_headers(self):
        # std.cout doesn't require the '%' symbol to print stuff...
        # so it works much better with python's string-substitution stuff.
        return ['<cuda_runtime.h>', '<cusparse_v2.h>',
                '<compyte/extension.h>', 'cuda.h', 'compyte/util.h']

    def c_libraries(self):
        return ['cusparse', 'cudart']

    # code_cache_version is built by subclasses from
    #  build_gemm_version

    def _c_header_dirs(self):
        ret = []
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret

    def _c_lib_dirs(self):
        ret = []
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'lib'))
        return ret

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx);
        CUdeviceptr (*cuda_get_ptr)(gpudata *g);
        cusparseHandle_t cusparseHandle = 0;
        cusparseMatDescr_t descr = 0;
        """

    def c_init_code(self):
        return ['cuda_get_ctx = (CUcontext (*)(void *ctx))compyte_get_extension("cuda_get_ctx");',
                'cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))compyte_get_extension("cuda_get_ptr");',
                """
        // We try to init it here to don't init it in the
        // Theano fct call to don't mess with profiling.
        // But as we can't raise error here, we try again
        // in c_code, to raise the good error.
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);
        if (cusparseStatus == CUSPARSE_STATUS_SUCCESS) {
          assert(cusparseHandle != 0);
          /* Get the mat description */
          cusparseStatus = cusparseCreateMatDescr(&descr);
          if (cusparseStatus == CUSPARSE_STATUS_SUCCESS) {
            assert(descr != 0);
            cusparseStatus = cusparseSetMatType(descr,
                             CUSPARSE_MATRIX_TYPE_GENERAL);
            if (cusparseStatus == CUSPARSE_STATUS_SUCCESS) {
              cusparseStatus = cusparseSetMatIndexBase(descr,
                             CUSPARSE_INDEX_BASE_ZERO);
              if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
                descr = 0;
              }
            }else{
                descr = 0;
            }
          }else{
            descr = 0;
          }
        }else{
          cusparseHandle = 0;
        }
                """
        ]

    def c_code(self, node, name, inputs, outputs, sub):
        #TODO support other dtype then float32
        (x_val, x_ind, x_ptr, x_shape,
         y_val, y_ind, y_ptr, y_shape) = inputs
        z_val, z_ind, z_ptr, z_shape = outputs
        fail = sub['fail']
        code = """
    const float alpha = 1;
    const float beta = 0;

    cusparseStatus_t cusparseStatus;
    int %(name)serr;
    const size_t x_shp0 = ((dtype_%(x_shape)s *)PyArray_DATA(%(x_shape)s))[0];
    const size_t x_shp1 = ((dtype_%(x_shape)s *)PyArray_DATA(%(x_shape)s))[1];
    const size_t y_shp0 = ((dtype_%(y_shape)s *)PyArray_DATA(%(y_shape)s))[0];
    const size_t y_shp1 = ((dtype_%(y_shape)s *)PyArray_DATA(%(y_shape)s))[1];
    size_t out_dims[2] = {x_shp0,
                          y_shp1};
    size_t m_p_1 = x_shp0 + 1;
    const size_t nnzX = %(x_val)s->ga.dimensions[0];
    const size_t nnzY = %(y_val)s->ga.dimensions[0];
    int nnzZ = -1;
    int *nnzTotalDevHostPtr = &nnzZ;
    size_t nnzZ_size_t[1];
    npy_intp shape_shape[] = {2};

    if (x_shp1 != y_shp0)
    {
        PyErr_Format(
                PyExc_ValueError,
                "GpuDotCsrCsrCsr: input shape incompatible:"
                " (%%d, %%d) and (%%d, %%d)",
                x_shp0, x_shp1,
                y_shp0, y_shp1);
        %(fail)s
    }

    /* Get handle to the CUSPARSE context */
    if (cusparseHandle == 0){
        printf("create cusparse handle\\n");
        cusparseStatus = cusparseCreate(&cusparseHandle);

        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrCsrCsr: cusparseCreate() failed");
            %(fail)s
        }
        assert(cusparseHandle != 0);
    }
    /* Get the mat description */
    if (descr == 0){
        printf("create cusparse desct\\n");

        cusparseStatus = cusparseCreateMatDescr(&descr);

        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrCsrCsr: cusparseCreateMatDescr() failed");
            cusparseDestroy(cusparseHandle);
            cusparseHandle = 0;
            %(fail)s
        }
        assert(descr != 0);
        cusparseStatus = cusparseSetMatType(descr,
                             CUSPARSE_MATRIX_TYPE_GENERAL);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
             PyErr_SetString(
             PyExc_RuntimeError,
                        "GpuDotCsrCsrCsr: cusparseSetMatType() failed");
             cusparseDestroyMatDescr(descr);
             cusparseDestroy(cusparseHandle);
             descr = 0;
             cusparseHandle = 0;
             %(fail)s
        }

        cusparseStatus = cusparseSetMatIndexBase(descr,
                             CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "GpuDotCsrCsrCsr: cusparseSetMatIndexBase() failed");
            cusparseDestroyMatDescr(descr);
            cusparseDestroy(cusparseHandle);
             descr = 0;
             cusparseHandle = 0;
            %(fail)s
        }
    }

    //TODO reuse!
    Py_XDECREF(%(z_val)s);
    Py_XDECREF(%(z_ind)s);
    Py_XDECREF(%(z_ptr)s);
    Py_XDECREF(%(z_shape)s);

    %(z_shape)s = (PyArrayObject*) PyArray_SimpleNew(1, shape_shape, NPY_INT32);
    if (!%(z_shape)s) {
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        PyErr_SetString(PyExc_MemoryError, "cpu alloc failed");
        %(fail)s
    }
    ((npy_int32*)PyArray_GETPTR1(%(z_shape)s, 0))[0] = out_dims[0];
    ((npy_int32*)PyArray_GETPTR1(%(z_shape)s, 1))[0] = out_dims[1];

    %(z_ptr)s = pygpu_empty(1, &m_p_1, %(y_ptr)s->ga.typecode, GA_C_ORDER,
        pygpu_default_context(),
        (PyObject *)&PyGpuArrayType);
    if (NULL == %(z_ptr)s) {
        Py_CLEAR(%(z_val)s);
        Py_CLEAR(%(z_ind)s);
        Py_CLEAR(%(z_ptr)s);
        Py_CLEAR(%(z_shape)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        // pygpu_empty set an error message
        %(fail)s
    }

    %(name)serr = cusparseXcsrgemmNnz(cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                x_shp0, y_shp1, x_shp1,
                descr, nnzX,
                (int*)cuda_get_ptr(%(x_ptr)s->ga.data),
                (int*)cuda_get_ptr(%(x_ind)s->ga.data),
                descr, nnzX,
                (int*)cuda_get_ptr(%(y_ptr)s->ga.data),
                (int*)cuda_get_ptr(%(y_ind)s->ga.data),
                descr,
                (int*)cuda_get_ptr(%(z_ptr)s->ga.data),
                &nnzZ);
    if (%(name)serr != GA_NO_ERROR) {
        Py_CLEAR(%(z_val)s);
        Py_CLEAR(%(z_ind)s);
        Py_CLEAR(%(z_ptr)s);
        Py_CLEAR(%(z_shape)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        PyErr_SetString(
            PyExc_MemoryError,
            "GpuDotCsrCsrCsr: cusparseXcsrgemmNnz() failed.");
        %(fail)s
    }
    if (NULL != nnzTotalDevHostPtr){
            nnzZ = *nnzTotalDevHostPtr;
    }else{
        assert(0);
/*        cudaMemcpy(&nnzZ , csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, csrRowPtrC  , sizeof(int), cudaMemcpyDeviceToHost);
        nnzZ -= baseC;
        */
    }

    //cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzZ);
    nnzZ_size_t[0] = nnzZ;
    %(z_ind)s = pygpu_empty(1, nnzZ_size_t, %(y_ind)s->ga.typecode, GA_C_ORDER,
        pygpu_default_context(),
        (PyObject *)&PyGpuArrayType);
    if (NULL == %(z_ind)s) {
        Py_CLEAR(%(z_val)s);
        Py_CLEAR(%(z_ind)s);
        Py_CLEAR(%(z_ptr)s);
        Py_CLEAR(%(z_shape)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        // pygpu_empty set an error message
        %(fail)s
    }

    //cudaMalloc((void**)&csrValC   , sizeof(float)*nnzZ);
    %(z_val)s = pygpu_empty(1, nnzZ_size_t, %(y_val)s->ga.typecode, GA_C_ORDER,
        pygpu_default_context(),
        (PyObject *)&PyGpuArrayType);
    if (NULL == %(z_ind)s) {
        Py_CLEAR(%(z_val)s);
        Py_CLEAR(%(z_ind)s);
        Py_CLEAR(%(z_ptr)s);
        Py_CLEAR(%(z_shape)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        // pygpu_empty set an error message
        %(fail)s
    }

    cusparseStatus = cusparseScsrgemm(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            out_dims[0], out_dims[1], x_shp1, //m, n, k
            descr, nnzX,
            (float*)cuda_get_ptr(%(x_val)s->ga.data),
            (int*)cuda_get_ptr(%(x_ptr)s->ga.data),
            (int*)cuda_get_ptr(%(x_ind)s->ga.data),
            descr,  nnzY,
            (float*)cuda_get_ptr(%(y_val)s->ga.data),
            (int*)cuda_get_ptr(%(y_ptr)s->ga.data),
            (int*)cuda_get_ptr(%(y_ind)s->ga.data),
            descr,
            (float*)cuda_get_ptr(%(z_val)s->ga.data),
            (int*)cuda_get_ptr(%(z_ptr)s->ga.data),
            (int*)cuda_get_ptr(%(z_ind)s->ga.data));

    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        Py_CLEAR(%(z_val)s);
        Py_CLEAR(%(z_ind)s);
        Py_CLEAR(%(z_ptr)s);
        Py_CLEAR(%(z_shape)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        char * err_msg;
        if (cusparseStatus == CUSPARSE_STATUS_NOT_INITIALIZED){
            err_msg = "CUSPARSE_STATUS_NOT_INITIALIZED";
        }else if (cusparseStatus == CUSPARSE_STATUS_ALLOC_FAILED){
            err_msg = "CUSPARSE_STATUS_ALLOC_FAILED";
        }else if (cusparseStatus == CUSPARSE_STATUS_INVALID_VALUE){
            err_msg = "CUSPARSE_STATUS_INVALID_VALUE";
        }else if (cusparseStatus == CUSPARSE_STATUS_ARCH_MISMATCH){
            err_msg = "CUSPARSE_STATUS_ARCH_MISMATCH";
        }else if (cusparseStatus == CUSPARSE_STATUS_EXECUTION_FAILED){
            err_msg = "CUSPARSE_STATUS_EXECUTION_FAILED";
        }else if (cusparseStatus == CUSPARSE_STATUS_INTERNAL_ERROR){
            err_msg = "CUSPARSE_STATUS_INTERNAL_ERROR";
        }else if (cusparseStatus == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED){
            err_msg = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        }else{
            err_msg = "Unknow error code";
        }
        PyErr_Format(
                PyExc_RuntimeError,
                "GpuDotCsrCsrCsr: cusparseScsrgemm() returned status %%d (%%s)",
                cusparseStatus, err_msg);
        %(fail)s
    }
        """ % locals()
        if config.gpuarray.sync:
            code += "GpuArray_sync(%(out)s);" % locals()
        code += """
    cudaThreadSynchronize();

    if (cudaSuccess != cudaGetLastError())
    {
        Py_CLEAR(%(z_val)s);
        Py_CLEAR(%(z_ind)s);
        Py_CLEAR(%(z_ptr)s);
        Py_CLEAR(%(z_shape)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrCsrCsr: cudaGetLastError() failed after cusparseScsrmm()");
        %(fail)s
    }
        """ % locals()
        return code
        pass


@register_opt()
@gof.local_optimizer([sparse.basic.true_dot])
def local_gpu_dot_csr_csr_csr(node):
    """
    #move to gpu sparse.TrueDot(csr, csr) -> csr

    """
    # check for sparse.TrueDot(csr, csr)
    if (isinstance(node.op, sparse.TrueDot) and
        _is_sparse_variable(node.inputs[0]) and
        _is_sparse_variable(node.inputs[1])):
#        if not any([i.owner and
#                    i.owner.op == host_from_gpu
#                    for i in node.inputs]):
#            return False

        a, b = node.inputs
        if (a.type.format == 'csr' and b.type.format == 'csr' and
            a.dtype == b.dtype and a.dtype == 'float32'):

            a_val, a_ind, a_ptr, a_shape = sparse.csm_properties(a)
            b_val, b_ind, b_ptr, b_shape = sparse.csm_properties(b)
            out_vec = GpuDotCsrCsrCsr()(a_val, a_ind, a_ptr, a_shape, b_val, b_ind, b_ptr, b_shape)
            out = sparse.CSR(*out_vec)
            return [out]
    return False
