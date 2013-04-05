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

#@register_canonicalize
@local_optimizer()
def logpy_group_incsubtensor(node):
    # TODO: how to design re-usable patterns? (dtype, ndim, etc.)

    shape_of = node.fgraph.shape_feature.shape_of
    shape_dimo = goalifyN(
        shape_dim(shape_of))
    ndimo = goalify(lambda x: getattr(x, 'ndim'))
    x = node.outputs[0].type()
    if x.ndim == 0:
        return
    y = x[0].type()
    z = tensor.set_subtensor(x[1001], y)
    incs = []
    orig_out = node.outputs[0]
    while node:
        with variables(x, y, 1001):
            match = run(1, (x, y, 1001), (eq, node.outputs[0], z))
            if match:
                xx, yy, ii = match[0]
                incs.append((ii, xx, yy))
                node = xx.owner
                continue
        break
    if not incs:
        return
    incs.sort()
    if zip(*incs)[0] == tuple(range(shape_dim(shape_of)(xx, 0))):
        iin = tensor.concatenate([
            tensor.shape_padleft(yy)
            for ii, _, yy in incs])
        print 'INCS', incs
        return [iin]

#@register_canonicalize
@local_optimizer()
def logpy_lift_dimshuffle_throughsubtensor(node):
    if isinstance(node.op, tensor.DimShuffle):
        x, = node.inputs
        if x.owner and isinstance(x.owner.op, tensor.Subtensor):
            v = x.owner.inputs[0]
            if len(x.owner.inputs) > 1:
                # TODO
                return
            # v[vidx].dimshuffle(*xpat)
            # -> v.dimshuffle(*xpat2)[vidx2]
            xpat = node.op.new_order
            vidx = x.owner.op.idx_list
            if not len(vidx) == 1 or not isinstance(vidx[0], int):
                # TODO
                #print 'xtype', x.type
                #print 'x', x
                #print 'x', vidx, xpat

                return
            assert v.ndim == x.ndim + 1
            new_order = [0] + ['x' if ii == 'x' else (ii + 1)
                         for ii in xpat]
            vn = v.dimshuffle(*new_order)

            new_idx = [slice(None, None, None) for ii in new_order]
            for ii, idx in enumerate(vidx):
                ni = new_order.index(ii)
                new_idx[ni] = idx
            while new_idx[-1] == slice(None, None, None):
                new_idx.pop()

            #print 'x', vidx, xpat, '->', 'x', new_order, new_idx
            return [vn.__getitem__(*new_idx)]

#@register_canonicalize
@local_optimizer()
def logpy_join(node):
    if isinstance(node.op, tensor.Join):
        axis = node.inputs[0]
        tsrs = node.inputs[1:]
        if len(tsrs) < 2:
            return

        for i, (t0, t1) in enumerate(zip(tsrs[:-1], tsrs[1:])):
            reb_op = tensor.Rebroadcast((0, 0))
            x0 = reb_op(t0.type())
            x1 = reb_op(t1.type())
            op0 = var('op0')
            with variables(x0, x1):
                op(x[i], x[i+1])
                match = run(
                    1, [x0, x1, op0],
                    (eq, [t0, t1], [reb_op(x0), reb_op(x1)]),
                    (getattrreco, (x0, 'owner', 'op'), op0),
                    (getattrreco, (x1, 'owner', 'op'), op0),
                    (isinstanceo, op0, tensor.Elemwise),

                   )
                if match:
                    print 'MATCH', match
                else:
                    return


#@register_canonicalize
@local_optimizer()
def logpy_lift_reshape_through_subtensor(node):
    if isinstance(node.op, tensor.Reshape):
        x = node.inputs[0]
        if x.owner and isinstance(x.owner.op, tensor.Subtensor):
            v = x.owner.inputs[0]
            if len(x.owner.inputs) > 1:
                # TODO
                return
            vidx = x.owner.op.idx_list
            if not len(vidx) == 1:
                # TODO
                return
            wtf = [v.shape[0]] + node.inputs[1:]
            return [v.reshape(wtf, ndim=len(wtf)).__getitem__(vidx[0])]


if 0:

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

