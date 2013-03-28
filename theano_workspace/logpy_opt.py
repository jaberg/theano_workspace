import theano
import theano.tensor
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_specialize
from theano.tensor.blas import local_optimizer

from logpy import (
    eq,
    run,
    var,
    membero
    )
from logpy.core import lall as logical_all

from logpy.unify import(
    register_unify_object,
    register_unify_object_attrs,
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

class TensorShapeI(object):
    def __init__(self, tsor, shape_of, i):
        self.tsor = tsor
        self.shape_of = shape_of
        self.i = i

    @staticmethod
    def unify(u, v, s):
        print u, v
        assert 0


unify_dispatch[(TensorShapeI, TensorShapeI)] = TensorShapeI.unify

register_unify_object_attrs(theano.Apply, ['op', 'inputs', 'outputs'])
register_unify_object_attrs(theano.tensor.IncSubtensor, [
    'idx_list', 'inplace', 'set_instead_of_inc',
    'destroyhandler_tolerate_aliased'])



def raw_init(cls, **kwargs):
    rval = object.__new__(cls)
    rval.__dict__.update(kwargs)
    return rval


@register_specialize
@register_canonicalize
@local_optimizer()
def logpy_cut_whole_incsubtensor(node):
    # -- declare some wild variables
    w = dict((name, var(name)) for name in [
        'start', 'stop', 'step', 'set_instead_of_inc', 'inplace', 'dta',
        'in_x', 'in_inc', 'outputs',
        ])
    shape_of = node.fgraph.shape_feature.shape_of

    # -- use them in a pattern
    pattern = raw_init(theano.Apply,
        op=raw_init(theano.tensor.IncSubtensor,
            idx_list=[slice(0, w['stop'], w['step'])],
            inplace=w['inplace'],
            set_instead_of_inc=w['set_instead_of_inc'],
            destroyhandler_tolerate_aliased=w['dta']),
        inputs=[w['in_x'], w['in_inc']],
        outputs=w['outputs'])

    matches = run(0, w,
        logical_all(
            (eq, node, pattern),
            (eq, TensorShapeI(w['in_x'], shape_of, 0), w['stop']),
            (membero, w['step'], (1, None)),
            ))
    assert 0, ('len matches', len(matches))
    if matches:
        match = matches[0]
        if match['set_instead_of_inc']:
            return [match['in_inc']]
        else:
            return [match['in_x'] + match['in_inc']]
