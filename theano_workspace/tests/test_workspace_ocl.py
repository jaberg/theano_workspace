import functools
import unittest

import numpy as np
import pyopencl as cl

import theano
from theano import tensor
from theano_workspace.workspace import SimpleWorkspace
from theano_workspace.workspace_ocl import WorkspaceOCL
from theano_workspace.workspace_ocl import pre_alloc_reg, gen_kernel_reg

def pre_alloc_elemwise(queue, node, storage_map):
    for inp in node.inputs:
        if inp not in storage_map:
            raise NotImplementedError()
    # XXX figure out the max size on each dim
    # XXX use the output dtype
    x = storage_map[node.inputs[0]][0]
    for out in node.outputs:
        storage_map[out] = [cl.array.empty_like(x)]
pre_alloc_reg[tensor.Elemwise] = pre_alloc_elemwise

def gen_kernel_elemwise(queue, node, storage_map):
    x = storage_map[node.inputs[0]][0]
    y = storage_map[node.outputs[0]][0]
    N, = x.shape
    assert x.dtype == 'float64'
    fn = cl.Program(queue.context, """
            __kernel void fn(
                __global const double *x,
                __global double *y
                         )
            {
                int gid = get_global_id(0);
                for (int ii = get_global_id(0); ii < %(N)s; ii += get_global_size(0))
                {
                    y[ii] = exp(x[ii]);
                }
            }
    """ % locals()).build().fn
    args = (queue, (N,), None, x.data, y.data)
    return fn, args
gen_kernel_reg[tensor.Elemwise] = gen_kernel_elemwise



class TestElemwise1(unittest.TestCase):
    def test_exp(self):
        x = tensor.vector('x')
        ws = SimpleWorkspace()
        ws[x] = [1, 2]
        ws.add_method('f', updates=[(x, tensor.exp(x)),])

        wso = WorkspaceOCL(ws)
        wso.f()
        print 'wso[x] =', wso[x]
        assert np.allclose(wso[x], np.exp([1., 2.]))


