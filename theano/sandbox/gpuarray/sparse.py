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
        return ['<cuda_runtime.h>', '<cusparse_v2.h>', '<compyte/extension.h>', 'cuda.h'] # cuda.h?

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
    GpuArray usable_y_stack;
    GpuArray* usable_y = NULL;

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
        usable_y = &usable_y_stack;
        if (%(name)serr != GA_NO_ERROR) {
            PyErr_SetString(
                PyExc_MemoryError,
                "GpuDotCsrDense: Can't allocate device memory for transposed input.");
            %(fail)s
        }
        if (GpuArray_copy(usable_y, &(%(y)s->ga), GA_F_ORDER) != GA_NO_ERROR){
            PyErr_SetString(
                PyExc_ValueError,
                "GpuDotCsrDense: call to GpuArray_copy() failed");
            %(fail)s
        }
    }else{
        usable_y = &(%(y)s->ga);
    }

    //TODO reuse!
    Py_XDECREF(%(out)s);
    %(out)s = new_GpuArray((PyObject *)&PyGpuArrayType,
        pygpu_default_context(), Py_None);
    if (%(out)s == NULL) {
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        // new_GpuArray calls __new__ which will set an error message
        // if it returns NULL.
        %(fail)s
    }
    //TODO: If next call commented, segfault.

    %(name)serr = GpuArray_empty(&%(out)s->ga,
        pygpu_default_context()->ops,
        pygpu_default_context()->ctx,
        %(y)s->ga.typecode,
        2, out_dims, GA_F_ORDER);
    if (%(name)serr != GA_NO_ERROR) {
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        Py_CLEAR(%(out)s);
        PyErr_SetString(
            PyExc_MemoryError,
            "GpuDotCsrDense: Can't allocate device memory for result.");
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
            (float*)cuda_get_ptr(usable_y->data),
            usable_y->strides[1]/GpuArray_ITEMSIZE(usable_y), // ldb
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
            (float*)cuda_get_ptr(usable_y->data),
            usable_y->ga.strides[1]/GpuArray_ITEMSIZE(usable_y), // ldb
            &beta,
            (float*)cuda_get_ptr(%(out)s->ga.data),
            out_dims[0]); //ldc suppose out is f contiguous
#endif

    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        Py_CLEAR(%(out)s);
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
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
        PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrDense: cudaGetLastError() failed after cusparseScsrmm()");
        %(fail)s
    }

    if (usable_y == &usable_y_stack)
    {
        GpuArray_clear(usable_y);
    }
        """ % locals()
        return code


@register_opt()
@gof.local_optimizer([sparse.basic._dot])
def local_gpu_dot_csr_dense(node):
    """
    move to gpu sparse.Dot(csr, dense) and sparse.Dot(dense, csc)
    """
    # check for sparse.Dot(csr, dense)
    if (isinstance(node.op, sparse.Dot) and
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
    # check for sparse.Dot(csc, dense)
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
