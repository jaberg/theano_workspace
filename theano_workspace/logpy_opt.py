import theano
import theano.tensor

from logpy import (
    eq,
    run,
    var,
    )
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


register_unify_object_attrs(theano.Apply, ['op', 'inputs', 'outputs'])
register_unify_object_attrs(theano.tensor.IncSubtensor, [
    'idx_list', 'inplace', 'set_instead_of_inc',
    'destroyhandler_tolerate_aliased'])


def raw_init(cls, **kwargs):
    rval = object.__new__(cls)
    rval.__dict__.update(kwargs)
    return rval

