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
    membero
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

def match_shape_i(shape_of, x, i, shp_i):
    def goal_shape(s):
        # figure out how to call reify and unify here
        # to make it a proper goal
        if hasattr(x, 'token'):
            x_ = s[x]
        else:
            x_ = x
        if hasattr(i, 'token'):
            i_ = s[i]
        else:
            i_ = i
        if hasattr(shp_i, 'token'):
            shp_i_ = s[shp_i]
        else:
            shp_i_ = shp_i
        try:
            zval = int(get_scalar_constant_value(shape_of[x_][i_]))
        except NotScalarConstantError:
            return []
        if zval == shp_i_:
            return [s]
        return []
    return goal_shape

@register_specialize
@register_canonicalize
@local_optimizer()
def logpy_cut_whole_incsubtensor(node):
    if not isinstance(node.op, tensor.IncSubtensor):
        return
    # -- declare some wild variables
    w = dict((name, var(name)) for name in [
        'start', 'stop', 'step', 'set_instead_of_inc', 'inplace', 'dta',
        'in_x', 'in_inc', 'outputs', 'rval',
        ])

    shape_of = node.fgraph.shape_feature.shape_of

    # -- use them in a pattern
    matches = run(0, w,
        logical_all(
            eq(
                node,
                raw_init(theano.Apply,
                    op=raw_init(tensor.IncSubtensor,
                        idx_list=[slice(0, w['stop'], w['step'])],
                        inplace=w['inplace'],
                        set_instead_of_inc=w['set_instead_of_inc'],
                        destroyhandler_tolerate_aliased=w['dta']),
                    inputs=[w['in_x'], w['in_inc']],
                    outputs=w['outputs'])
                ),
            membero(w['step'], (1, None)),
            match_shape_i(shape_of, w['in_x'], 0, w['stop']),
            conde(
                [
                    eq(w['set_instead_of_inc'], True),
                    eq(w['rval'], (tensor.add, w['in_inc']))],
                [
                    eq(w['set_instead_of_inc'], False),
                    eq(w['rval'], (tensor.add, w['in_x'], w['in_inc']))]
                ),
            )
        )
    if matches:
        return [matches[0]['rval'][0](*matches[0]['rval'][1:])]
