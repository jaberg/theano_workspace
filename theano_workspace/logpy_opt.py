import numpy
import theano
from theano import tensor
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_specialize
from theano.tensor.blas import local_optimizer

from theano.tensor.basic import get_scalar_constant_value
from theano.tensor.basic import NotScalarConstantError
from theano.tensor.basic import TensorVariable

from logpy import (
    eq,
    conde,
    run,
    var,
    membero,
    goalify,
    )
from logpy.core import (
    lall as logical_all,
    EarlyGoalError,
    )
from logpy.variables import (
    Var,
    vars,
    )

from logpy.unify import(
    register_unify_object,
    register_unify_object_attrs,
    reify,
    reify_dispatch,
    reify_generator,
    reify_object,
    unify_dispatch,
    unify_object,
    unify_object_attrs,
    unify,
    )

# XXX  need context manager to get rid of this
if 1: # DEBUGGING W OPS BUILT WITH RAW_INIT
    theano.gof.Apply.__repr__ = object.__repr__
    theano.gof.Apply.__str__ = object.__str__

register_unify_object_attrs(theano.Apply, ['op', 'inputs', 'outputs'])
register_unify_object_attrs(tensor.TensorVariable, ['type', 'owner'])
register_unify_object_attrs(tensor.IncSubtensor, [
    'idx_list', 'inplace', 'set_instead_of_inc',
    'destroyhandler_tolerate_aliased'])



def raw_init(cls, **kwargs):
    rval = object.__new__(cls)
    rval.__dict__.update(kwargs)
    return rval

def shape_dim(shape_of):
    def shape_dim_i(x, i):
        try:
            return int(get_scalar_constant_value(shape_of[x][i]))
        except NotScalarConstantError:
            raise EarlyGoalError()
    return shape_dim_i


class LogpyTensorVar(tensor.TensorVariable, Var):
    def assoc(self, d, other):
        if 0:
            if self.type.dtype:
                if self.type.dtype != other.type.dtype:
                    return False
            if self.type.ndim:
                if self.type.ndim != other.type.ndim:
                    return False
        d = d.copy()
        d[self] = other
        return d

class LogpyNumpyInteger(int, Var):
    # put these built-ins up front to hide int versions
    def __str__(self):
        return "~" + str(self.token)
    __repr__ = __str__

    def __eq__(self, other):
        rval = type(self) == type(other) and self.token == other.token
        print 'running eq', self, other, rval
        return rval

    def __hash__(self):
        return hash((type(self), self.token))


def lptensor(token, dtype, broadcastable):
    rval = LogpyTensorVar(token)
    rval.type = tensor.TensorType(dtype=dtype, broadcastable=broadcastable)
    return rval

def lpint(val, token):
    rval = LogpyNumpyInteger(val)
    rval.token = token
    return rval
    #rtype = tensor.TensorType(dtype=dtype, broadcastable=broadcastable)
    #_var_data = var(token + '_data')
    #print fakedata
    #return LogpyTensorConstant(rtype, fakedata, token)

@register_specialize
@register_canonicalize
@local_optimizer()
def logpy_cut_whole_incsubtensor(node):
    if not isinstance(node.op, tensor.IncSubtensor):
        return
    magic = 238908925034
    jj = lpint(magic, 'j')
    x = lptensor('x', 'float32', [False])
    y = lptensor('y', 'float32', [False])
    pattern = tensor.set_subtensor(x[0:jj], y)
    theano.printing.debugprint(node.outputs[0])
    theano.printing.debugprint(pattern)
    print '-' * 80
    unify(node.outputs[0], pattern, {})

    return
    # -- declare some wild variables
    rval, outputs, in_inc, in_x = vars(4)
    start, stop, step, set_instead_of_inc, inplace, dta = vars(6)
    x_shpdim = var()

    shape_dimo = goalify(
        shape_dim(node.fgraph.shape_feature.shape_of))


    # -- use them in a pattern
    matches = run(1, rval,
        eq(
            node,
            raw_init(theano.Apply,
                op=raw_init(tensor.IncSubtensor,
                    idx_list=[slice(0, stop, step)],
                    inplace=inplace,
                    set_instead_of_inc=set_instead_of_inc,
                    destroyhandler_tolerate_aliased=dta),
                inputs=[in_x, in_inc],
                outputs=outputs)
            ),
        membero(step, (1, None)),
        (eq, x_shpdim, (in_x, 0)),
        (shape_dimo, x_shpdim, stop),
        conde(
            [
                eq(set_instead_of_inc, True),
                eq(rval, (tensor.add, in_inc))],
            [
                eq(set_instead_of_inc, False),
                eq(rval, (tensor.add, in_x, in_inc))]
            ),
        )
    if matches:
        return [matches[0][0](*matches[0][1:])]

