import theano
from theano import tensor
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_specialize
from theano.tensor.blas import local_optimizer

from theano.tensor.basic import get_scalar_constant_value
from theano.tensor.basic import NotScalarConstantError

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

@register_specialize
@register_canonicalize
@local_optimizer()
def logpy_cut_whole_incsubtensor(node):
    # -- declare some wild variables
    rval, outputs, in_inc, in_x = [var() for i in [0] * 4]
    start, stop, step = [var() for i in [0] * 3]
    set_instead_of_inc, inplace, dta = [var() for i in [0] * 3]
    x_shpdim = var()

    shape_dimo = goalify(
        shape_dim(node.fgraph.shape_feature.shape_of))

    # -- use them in a pattern
    matches = run(0, rval,
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

