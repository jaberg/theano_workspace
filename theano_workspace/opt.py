from collections import OrderedDict
import numpy as np
import theano

from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_specialize
from theano.tensor.blas import local_optimizer
from theano.tensor.blas import Optimizer

IncSubtensor = theano.tensor.IncSubtensor
Subtensor = theano.tensor.Subtensor
Reshape = theano.tensor.Reshape
Elemwise = theano.tensor.Elemwise
from theano.tensor.basic import get_scalar_constant_value
from theano.tensor.basic import NotScalarConstantError
from theano import tensor

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


def optimizer_from_any(specifier):
    if isinstance(specifier, basestring):
        try:
            dct = theano.compile.mode.predefined_optimizers
            query = dct[specifier]
        except KeyError:
            raise ValueError('Optimizer %s not in %s' % (
                specifier, dct))
        return theano.compile.mode.optdb.query(query)
    elif isinstance(specifier, theano.gof.Query):
        return theano.compile.mode.optdb.query(specifier)
    else:
        # TODO probably not implemented error is more appropriate
        raise TypeError(specifier)


class RefactorSubtensors(Optimizer):
    """
    May ops can process entire tensors at once if it has already been
    specified that they should process each slice of a tensor.

    op(x[:a]), op([a:b]), op(x[b:]) -> A = op(x), A[:a], A[a:b] A[b:]

    Graphs of the latter form are easier to implement fast because op(x) can
    be parallel internally.

    """
    mergers = []

    @classmethod
    def add_merger(cls, f):
        cls.mergers.append(f)
        return f

    def add_requirements(self, fgraph):
        fgraph.attach_feature(theano.gof.toolbox.ReplaceValidate())

    @staticmethod
    def downstream_op_iter(idxs):
        """Key routine in recognizing refactor opportunities.

        idxs: a list of (int i, node n) pairs such that each node is some
            x[i].owner, for one base variable x.

        Returns: ((op, pos, itypes), nodes)
            Each nodes[i] is an op(..., x[i], ...) where the relevant slice of
            x shows up in position `pos` in the inputs to every node.
        """
        ops = OrderedDict()
        if range(len(idxs)) == list(zip(*idxs)[0]):
            for i0, n in idxs:
                for client_apply, pos_in_client in n.outputs[0].clients:
                    key = (client_apply.op, pos_in_client)
                    otypes = (tuple(i.type for i in client_apply.outputs),)
                    assert len(set(otypes)) == 1
                    key += (otypes[0],)
                    key += (tuple(tuple(i.broadcastable)
                                  for i in client_apply.inputs),)
                    ops.setdefault(key, []).append(client_apply)
            for key, ins in ops.items():
                #print key
                #print len(ins)
                #print len(idxs)
                #print ins
                if len(ins) == len(idxs):
                    yield (key, ins)
        else:
            # TODO work with this case
            pass

    def foo(self, x, subtensor_clients):
        # -- potentially merge the subtensor clients of x

        if len(subtensor_clients) <= 1:
            # -- leave this for another optimization
            return

        if any(len(st.inputs) > 1 for st in subtensor_clients):
            # -- TODO: support non-constant indexing ranges
            return

        # if we're dealing with x[i] subtensors
        if all(((len(n.op.idx_list) == 1)
                and isinstance(n.op.idx_list[0], int)
                )
                for n in subtensor_clients):
            idxs = [(n.op.idx_list[0], n)
                for n in subtensor_clients]
            idxs.sort()
            assert len(idxs) > 1
            shpf = shape_dim(self.fgraph.shape_feature.shape_of)
            if len(idxs) != shpf(x, 0):
                return
            def opr_fn():
                for op_pos_rval in self.downstream_op_iter(idxs):
                    #print op_pos_rval
                    for fn in self.mergers:
                        yield op_pos_rval, fn
                    print 'failed to merge', op_pos_rval[0]
            for opr, fn in opr_fn():
                #print 'thus, opr', opr, fn
                replacements = fn(x, opr)
                if replacements:
                    print 'REPLACEMENTS', replacements
                    self.fgraph.replace_all_validate(replacements,
                        reason='RefactorSubtensors')
                    self.nb_replacement += len(replacements)
                    return True


    def apply(self, fgraph):
        self.fgraph = fgraph
        self.nb_iter = 0
        self.nb_replacement = 0
        self.nb_replacement_didn_t_remove = 0
        self.nb_inconsistency_make = 0
        self.nb_inconsistency_replace = 0
        self.time_canonicalize = 0
        self.time_factor_can = 0
        self.time_factor_list = 0
        self.time_toposort = 0

        did_something = True
        #print '-- START -- '

        Subtensor = theano.tensor.Subtensor

        # XXX: make sure we don't replace all elemwise(subtensor) with
        # subtensor(elemwise) because most of the time that would be a bad
        # idea!

        while did_something:
            self.nb_iter += 1

            subtensors = [n for n in fgraph.toposort()
                    if isinstance(n.op, Subtensor)]

            # x -> x[a], x[b], ...
            xs_with_subtensor = {}
            for n in subtensors:
                xs_with_subtensor.setdefault(n.inputs[0], []).append(n)

            did_something = any(self.foo(x, subtensor_clients)
                    for x, subtensor_clients in xs_with_subtensor.items())

        #theano.printing.debugprint(fgraph.outputs)
        #print '-- DONE -- '
        return (self,
                self.nb_iter,
                self.nb_replacement,
                self.nb_replacement_didn_t_remove,
                self.nb_inconsistency_make, 
                self.nb_inconsistency_replace,
                self.time_canonicalize, 
                self.time_factor_can,
                self.time_factor_list, 
                self.time_toposort)

    def logic(self):
        # todo make this work with logpy
        pattern = op(x[a:b], u), op(x[b:c], v), op(x[c:d], w)
        result = op(x, concatenate(u, v, w))
        
rst = RefactorSubtensors()
theano.compile.mode.optdb.register('refactor_subtensors',
        rst,
        0, 'fast_compile', 'fast_run')

@rst.add_merger
def refactor_subtensor_elemwise(x, op_pos_rval):
    (op, xpos, otype, itypes), xclients = op_pos_rval
    print 'merge?', op
    if not isinstance(op, Elemwise):
        return
    # each xclient[i] is something like
    # op(a, b, x[i], c)
    # 
    new_inputs = []
    for ipos, ibc in enumerate(itypes):
        if ipos == xpos:
            iin = x
        else:
            iin = tensor.concatenate([
                tensor.shape_padleft(xcl.inputs[ipos])
                for xcl in xclients])
        assert iin.broadcastable[1:] == ibc
        new_inputs.append(iin)
    new_output0 = op(*new_inputs)

    replacements = [(xcl.outputs[0], new_output0[i])
        for i, xcl in enumerate(xclients)]
    #print 'RTYPES', [(a.type, b.type) for a, b in replacements]
    #print 'INPUTS', [a.type for a in new_inputs]
    return replacements


