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

# XXX  need context manager to get rid of this
if 1: # DEBUGGING W OPS BUILT WITH RAW_INIT
    theano.gof.Apply.__repr__ = object.__repr__
    theano.gof.Apply.__str__ = object.__str__

register_unify_object_attrs(theano.Apply, ['op', 'inputs'])
register_unify_object_attrs(tensor.TensorVariable, ['type', 'owner', 'name'])
register_unify_object_attrs(tensor.IncSubtensor, [
    'idx_list', 'inplace', 'set_instead_of_inc',
    'destroyhandler_tolerate_aliased'])
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
            return -1 # an unsatisfiable shape
    return shape_dim_i


class LogpyTensorVar(Var, tensor.TensorVariable):

    def _logpy_reify_dispatch(self, s):
        return s[self]

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
    # -- int has to go first, before Var so that __new__ doesn't complain,
    # but then these methods from Var have to be duplicated here...
    # maybe logpy Var could be built from one class that implements __new__
    # and a mix-in that provides these?
    def __str__(self):
        return "~" + str(self.token)
    __repr__ = __str__

    def __eq__(self, other):
        return type(self) == type(other) and self.token == other.token

    def __hash__(self):
        return hash((type(self), self.token))


def lptensor(token, dtype, broadcastable):
    rval = LogpyTensorVar(token)
    rval.type = tensor.TensorType(dtype=dtype, broadcastable=broadcastable)
    rval.name = token
    return rval

def lpint(val, token):
    rval = LogpyNumpyInteger(val)
    rval.token = token
    return rval
    #rtype = tensor.TensorType(dtype=dtype, broadcastable=broadcastable)
    #_var_data = var(token + '_data')
    #print fakedata
    #return LogpyTensorConstant(rtype, fakedata, token)

def goalifyN(func):
    funco = goalify(func)
    def goalo(args, result):
        tmp = var()
        return (lall,
            (eq, tmp, args),
            (funco, tmp, result))
    return goalo

def logpy_optimization(f):
    def deco(node):
        rval, goals = f(node) #-- XXX shape features require node
        matches = run(1, rval, *goals)
        return matches[0] if matches else None
    deco.__name__ == f.__name__
    return deco

@register_specialize
@register_canonicalize
@local_optimizer()
@logpy_optimization
def logpy_cut_whole_incsubtensor(node):
    # TODO: how to design re-usable patterns? (dtype, ndim, etc.)
    shape_dimo = goalifyN(
        shape_dim(node.fgraph.shape_feature.shape_of))
    jj = lpint(238908925034, 'j') # -- a number that cannot occur in the graph
    x = lptensor('x', 'float32', [False])
    y = lptensor('y', 'float32', [False])
    x0 = var('x0')
    rval = [y]
    goals = (
            (eq, node.outputs[0], tensor.set_subtensor(x[0:jj], y)),
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

