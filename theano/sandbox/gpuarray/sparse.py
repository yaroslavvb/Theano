import numpy
import scipy.sparse

import theano
from theano import gof, sparse
from theano.gof.python25 import any
import theano.sandbox.gpuarray as gpuarray
from theano.sandbox.gpuarray.basic_ops import (
    as_gpuarray_variable, host_from_gpu, gpu_from_host, GpuArrayType)
from theano.sandbox.gpuarray.opt import register_opt
from theano.sparse.basic import _is_sparse_variable


class GpuDotCsrDense(gof.Op):
    """
    Sparse version of dot(sparse csr, dense)

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
        assert x_val.dtype == 'float32' and x_val.ndim == 1
        assert x_ind.dtype == 'int32' and x_ind.ndim == 1
        assert x_ptr.dtype == 'int32' and x_ptr.ndim == 1
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
        return ['cusparse', 'cudart'] #        return ['cudart', 'cuda']

    # code_cache_version is built by subclasses from
    #  build_gemm_version

    def _c_compile_args(self):
        return ldflags(libs=False, flags=True)

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
                'cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))compyte_get_extension("cuda_get_ptr");']

    def c_code(self, node, name, inputs, outputs, sub):
        x_val, x_ind, x_ptr, x_shape, y = inputs
        out, = outputs
        fail = sub['fail']
        return """
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
            %(fail)s
        }
    }
    //if nvcc >= 5.5, use cusparseScsrmm2 call.
    if (%(y)s->ga.strides[0] != GpuArray_ITEMSIZE(&%(y)s->ga))
    {
        usable_y = &usable_y_stack;
        %(name)serr = GpuArray_empty(usable_y,
            pygpu_default_context()->ops,
            pygpu_default_context()->ctx,
            %(y)s->ga.typecode,
            2,
            %(y)s->ga.dimensions,
            GA_F_ORDER);
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
    //%(y)s->ga.ops->property(NULL, %(y)s->ga.data, NULL, GA_BUFFER_PROP_REFCNT, &refcnt);
    //printf("y refcnt=%%u\\n", refcnt);
    //usable_y->ops->property(NULL, usable_y->data, NULL, GA_BUFFER_PROP_REFCNT, &refcnt);
    //printf("usable_y refcnt=%%u\\n", refcnt);
    Py_XDECREF(%(out)s);
    %(out)s = new_GpuArray((PyObject *)&PyGpuArrayType,
        pygpu_default_context(), Py_None);
    if (%(out)s == NULL) {
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        // new_GpuArray calls __new__ which will set an error message
        // if it returns NULL.
        %(fail)s
    }
    //TODO: If next call commented, segfault.
    %(name)serr = GpuArray_empty(&%(out)s->ga,
        pygpu_default_context()->ops,
        pygpu_default_context()->ctx,
        %(y)s->ga.typecode, //get_typecode((PyObject *)PyArray_DESCR(%%(name)s_tmp)),
        2,
        out_dims, //(size_t *)PyArray_DIMS(%%(inp)s),
        GA_F_ORDER);
    if (%(name)serr != GA_NO_ERROR) {
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
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
            (int*)cuda_get_ptr(%(x_ptr)s->ga.data), // TODO check that the input dtype is equiv to c int.
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
            (int*)cuda_get_ptr(%(x_ptr)s->ga.data), // TODO check that the input dtype is equiv to c int.
            (int*)cuda_get_ptr(%(x_ind)s->ga.data),
            (float*)cuda_get_ptr(usable_y->data),
            usable_y->ga.strides[1]/GpuArray_ITEMSIZE(usable_y), // ldb
            &beta,
            (float*)cuda_get_ptr(%(out)s->ga.data),
            out_dims[0]); //ldc suppose out is f contiguous
#endif
    //cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);

    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
        Py_CLEAR(%(out)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        PyErr_Format(
                PyExc_RuntimeError,
                "GpuDotCsrDense: cusparseScsrmm[2]() returned status %%d",
                cusparseStatus);
        %(fail)s
    }
    //TODO remove!!!
    cudaThreadSynchronize();
    //CNDA_THREAD_SYNC;
    if (cudaSuccess != cudaGetLastError())
    {
        Py_CLEAR(%(out)s);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        PyErr_SetString(
                PyExc_RuntimeError,
                "GpuDotCsrDense: cudaGetLastError() failed after cusparseScsrmm()");
        %(fail)s
    }

    if (usable_y == &usable_y_stack)
    {
        GpuArray_clear(usable_y);
        //usable_y->ops->property(NULL, usable_y->data, NULL, GA_BUFFER_PROP_REFCNT, &refcnt);
        //printf("usable_y=%%p\\n", usable_y);
        //printf("usable_y->ops=%%p\\n", usable_y->ops);
    }
    if (0){
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cusparseHandle);
        descr = 0;
        cusparseHandle = 0;
    }
        """ % locals()
        pass


@register_opt()
@gof.local_optimizer([sparse.basic._dot])
def local_gpu_dot_csr_dense(node):
    # check for elemwise(..., host_from_gpu, ...)
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
    return False
