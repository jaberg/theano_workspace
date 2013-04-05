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
    lall,
    EarlyGoalError,
    )
from logpy.variables import (
    Var,
    vars,
    variables,
    )

from logpy.unify import(
    reify,
    reify_generator,
    unify_dispatch,
    unify,
    )

from logpy.unifymore import(
    register_unify_object_attrs,
    register_unify_object,
    reify_object,
    unify_object,
    unify_object_attrs,
    more_unify_dispatch,
    more_reify_dispatch,
    )

unify_dispatch.update(more_unify_dispatch)

register_unify_object_attrs(theano.Apply, ['op', 'inputs'])
register_unify_object_attrs(tensor.TensorVariable, ['type', 'owner', 'name'])
register_unify_object_attrs(tensor.IncSubtensor, [
    'idx_list', 'inplace', 'set_instead_of_inc',
    'destroyhandler_tolerate_aliased'])
register_unify_object_attrs(tensor.Subtensor, ['idx_list'])
register_unify_object_attrs(tensor.Reshape, ['ndim'])
register_unify_object_attrs(tensor.Dot, [])


def raw_init(cls, **kwargs):
    rval = object.__new__(cls)
    rval.__dict__.update(kwargs)
    return rval

def shape_dim(shape_of):
    def shape_dim_i(x, i):
        #print 'shape keys', shape_of.keys()
        #print 'args (x, i):', x, i
        try:
            return x.data.shape[i]
        except AttributeError:
            pass
        try:
            return int(get_scalar_constant_value(shape_of[x][i]))
        except NotScalarConstantError:
            pass
        try:
            return shape_of[x][i].eval()
        except:
            return -1 # an unsatisfiable shape
    return shape_dim_i


def goalifyN(func):
    funco = goalify(func)
    def goalo(args, result):
        tmp = var()
        return (lall,
            (eq, tmp, args),
            (funco, tmp, result))
    return goalo

def getattrrec(x, *attrs):
    if attrs:
        return getattrrec(getattr(x, attrs[0]), *attrs[1:])
    else:
        return x

getattro = goalifyN(getattr)
getattrreco = goalifyN(getattrrec)

def isinstanceo(a, b):
    def deco(s):
        if isinstance(a, b):
            yield s
    return deco


def logpy_optimization(f):
    def deco(node):
        rval, goals = f(node) #-- XXX shape features require node
        matches = run(1, rval, *goals)
        return matches[0] if matches else None
    deco.__name__ = f.__name__
    return deco

@register_specialize
@register_canonicalize
@local_optimizer()
@logpy_optimization
def logpy_cut_whole_setsubtensor(node):
    # TODO: how to design re-usable patterns? (dtype, ndim, etc.)
    shape_dimo = goalifyN(
        shape_dim(node.fgraph.shape_feature.shape_of))
    #jj = lpint(238908925034, 'j') # -- a number that cannot occur in the graph
    x = tensor.vector()
    y = tensor.vector()
    jj = 12345
    with variables(x, y, jj) :
        rval = [y]
        goals = (
            (eq, node.outputs[0], tensor.set_subtensor(x[0:jj], y)),
            (shape_dimo, (x, 0), jj),
            )
    return rval, goals

@register_specialize
@register_canonicalize
@local_optimizer()
@logpy_optimization
def logpy_cut_subtensor(node):
    # TODO: how to design re-usable patterns? (dtype, ndim, etc.)
    shape_dimo = goalifyN(
        shape_dim(node.fgraph.shape_feature.shape_of))
    #jj = lpint(238908925034, 'j') # -- a number that cannot occur in the graph
    x = tensor.vector()
    jj = 12345
    with variables(x, jj) :
        rval = [x]
        goals = (
            (eq, node.outputs[0], x[:jj]),
            (shape_dimo, (x, 0), jj),
            )
    return rval, goals


#@register_specialize
@register_canonicalize
@local_optimizer()
def logpy_remove_dot_scalar_matrix(node):
    # TODO: how to design re-usable patterns? (dtype, ndim, etc.)
    shape_of = node.fgraph.shape_feature.shape_of
    shape_dimo = goalifyN(
        shape_dim(shape_of))
    ndimo = goalify(lambda x: getattr(x, 'ndim'))
    x = theano.tensor.matrix() # -- XXX type should not matter
    y = theano.tensor.matrix() # -- XXX type should not matter
    if isinstance(node.op, theano.tensor.Dot):
        with variables(x, y):
            #theano.printing.debugprint(tensor.dot(x, y))
            result = run(1, (x, y),
                    (eq, node, tensor.dot(x, y).owner),
                    (ndimo, x, 2),
                    (shape_dimo, (x, 0), 1),
                    (shape_dimo, (x, 1), 1),
               )
        if result:
            xx, yy = result[0]
            #print 'MATCHED xx!', xx, shape_of[xx], xx.type
            #print 'MATCHED yy!', yy, shape_of[yy], yy.type
            #theano.printing.debugprint(xx)
            return [tensor.addbroadcast(xx, 0, 1).dimshuffle() * yy]

from logpy import fact, Relation

x = tensor.vector('x')
from theano.tensor import exp, log
rules = [
        (x + x, 2*x),
        (x * x, x**2),
        (exp(log(x)), x),
        (log(exp(x)), x),
        ]
vars = [x]
reduces = Relation('reduces')
for source, target in rules:
    fact(reduces, source, target)

def simplify(expr):
    source, target = var(), var()
    with variables(*vars):
        result = run(0, target, (reduces, source, target),
                                (eq, expr, source))
    return result

