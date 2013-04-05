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


def optimizer_from_any(specifier):
    if isinstance(specifier, basestring):
        try:
            dct = theano.compile.mode.predefined_optimizers
            query = dct[specifier]
        except KeyError:
            raise ValueError('Optimizer %s not in %s' % (
                specifier, dct))
        return theano.compile.mode.optdb.query(query)
    else:
        # TODO probably not implemented error is more appropriate
        raise TypeError(specifier)


class RefactorSubtensors(Optimizer):
    """
    op(x[a:b]), op(x[b:c]) -> A = op(x[a:c]), A[0:b-a], A[b-a:c-a]

    When some elementwise operation is applied separately to neighbouring
    parts of a tensor, this optimization rearranges things so that the
    elementwise operation is only applied once, and the result is split.
    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(theano.gof.toolbox.ReplaceValidate())

    @staticmethod
    def op_pos(idxs):
        ops = {}
        for i0, n in idxs:
            for client_apply, pos_in_client in n.outputs[0].clients:
                if isinstance(client_apply.op, Elemwise):
                    key = (client_apply.op, pos_in_client)
                    key += (tuple(i.type for i in client_apply.inputs),)
                    ops.setdefault(key, []).append(client_apply)
        for key, ins in ops.items():
            if len(ins) == len(idxs):
                return key, ins


    def foo(self, x, subtensor_clients):
        # -- potentially merge the subtensor clients of x

        if len(subtensor_clients) <= 1:
            # -- leave this for another optimization
            return

        if any(len(st.inputs) > 1 for st in subtensor_clients):
            # -- TODO: support non-constant indexing ranges
            return

        if all(((len(n.op.idx_list) == 1)
                and isinstance(n.op.idx_list[0], int)
                )
                for n in subtensor_clients):
            idxs = [(n.op.idx_list[0], n)
                for n in subtensor_clients]
            idxs.sort()
            assert len(idxs) > 1
            # TODO: support incomplete intervals
            if range(len(idxs)) == list(zip(*idxs)[0]):
                op_pos_rval = self.op_pos(idxs)
                if not op_pos_rval:
                    return
                # each xclient[i] is something like
                # op(a, b, x[i], c)
                # 
                (op, xpos, itypes), xclients = op_pos_rval
                new_inputs = []
                for ipos, itype in enumerate(itypes):
                    if ipos == xpos:
                        iin = x[:len(idxs)]
                    else:
                        iin = tensor.concatenate([
                            tensor.shape_padleft(xcl.inputs[ipos])
                            for xcl in xclients])
                    assert iin.broadcastable[1:] == itype.broadcastable, (
                        ipos, itype)
                    new_inputs.append(iin)
                new_output0 = op(*new_inputs)

                replacements = [(xcl.outputs[0], new_output0[i])
                    for i, xcl in enumerate(xclients)]
                #print 'REPLACEMENTS', replacements
                #print 'RTYPES', [(a.type, b.type) for a, b in replacements]
                #print 'INPUTS', [a.type for a in new_inputs]

                self.fgraph.replace_all_validate(replacements,
                    reason='RefactorSubtensors')
                self.nb_replacement += len(replacements)
                return True

                new_clients = []
                print op_pos
                #op, pos = op_pos
                theano.printing.debugprint(op_pos[1][0].outputs)
                theano.printing.debugprint(op_pos[1][1].outputs)

            if 0:
                print 'op inpos', 
                replacements = []
                to_go = set()
                # -- check for common operations on these slices.
                # TODO: check for *some* matches
                for start, stop, subt_node in ranges:
                    for client_apply, pos_in_client in subt_node.outputs[0].clients:
                        if len(client_apply.outputs) > 1:
                            raise NotImplementedError()
                        client_op = client_apply.op
                        if isinstance(client_op, theano.tensor.Elemwise):
                            new_inputs = list(client_apply.inputs)

                            # XXX: need to simultaneously replace
                            # all new_inputs that our subtensor
                            # merge is going to affect. If we are
                            # merging e.g.
                            #   add(x[1:2], y[1:2])
                            #   add(x[2:4], y[2:4])
                            #   -> add(x[1:4], y[1:4])[0:1]
                            #      add(x[1:4], y[1:4])[1:3]
                            # then we need to replace both of
                            # x and y.

                            new_inputs[pos_in_client] = x
                            new_out = client_op(*new_inputs)[start:stop]
                            replacements.append((client_apply.outputs[0], new_out))
                            assert client_apply.outputs[0] not in to_go
                            to_go.add(client_apply.outputs[0])
                if replacements:
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
        



theano.compile.mode.optdb.register('refactor_subtensors',
        RefactorSubtensors(),
        0, 'fast_compile', 'fast_run')


class Match(object):
    def __init__(self, name, cls,
            match_fn=None,
            unmatch_fn=None,
            **attrs):
        self._name = name
        self._cls = cls
        self._attrs = attrs
        self._match_fn = match_fn
        self._unmatch_fn = unmatch_fn

    def __call__(self, *inputs):
        if hasattr(self, '_inputs'):
            raise TypeError('wrong usage, call Match once')
        self._inputs = inputs
        return self

    def match(self, node, assignment):
        def ok(*args):
            #print 'OK ' + ' '.join([str(a) for a in args])
            pass
        def fail(*args):
            #print '-- ' + ' '.join([str(a) for a in args])
            pass

        if node is None:
            fail('not matching null node')
            return
        if isinstance(node.op, self._cls):
            ok('matched class', node.op, self._cls)
        else:
            fail('not matching wrong class', node.op, self._cls)
            return
        for attr, varname in self._attrs.items():
            opval = getattr(node.op, attr)
            if varname in assignment:
                if assignment[varname] == opval:
                    ok('matched attrib', varname, opval)
                elif (isinstance(opval, (list, tuple)) and
                    isinstance(assignment[varname], (list, tuple))
                    and tuple(opval) == tuple(assignment[varname])):
                    ok('matched attrib', varname, opval)
                else:
                    fail('not matching attribute', varname, opval)
                    return
            else:
                ok('assigning attrib', varname, opval)
                assignment[varname] = opval
        assignment[self._name] = node
        if len(node.inputs) != len(self._inputs):
            raise ValueError('input count')
        for invar, arg in zip(node.inputs, self._inputs):
            if isinstance(arg, Match):
                assignment = arg.match(invar.owner, assignment)
                if not assignment:
                    return
            elif isinstance(arg, MatchConstant):
                assignment = arg.match(invar, assignment)
                if assignment:
                    ok('matched constant', arg, invar)
                else:
                    fail('failed to match constant', arg, invar)
                    return
            elif isinstance(arg, basestring):
                if arg in assignment:
                    if assignment[arg] == invar:
                        ok('matched free var', arg, invar)
                    else:
                        fail('wrong free var', arg, invar)
                        return
                else:
                    ok('assigning free var', arg, invar)
                    assignment[arg] = invar
            else:
                raise NotImplementedError(arg)
        return assignment

class MatchConstant(object):
    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    def match(self, var, assignment):
        try:
            value = get_scalar_constant_value(var)
        except NotScalarConstantError:
            return
        if self.val is None:
            # -- any constant value will do
            assignment[self.name] = value
        else:
            if self.val == value:
                assignment[self.name] = value
            else:
                return
        return assignment


@register_canonicalize
@local_optimizer()
def local_consolidate_incsubtensor(node):
    # TODO: check to see if an IncSubtensor among the clients of node would
    # match, and abort if that's true, so we leave the work to the parent.
    # It's not clear if we can get away with not running the entire function
    # fo the parent though... maybe it's better just to write an Optimizer
    # class that iterates correctly and doesn't do too many replacements?
    #
    # -- Empirically, the current local_optimizer seems to be iterating from
    # outputs to inputs, which is actually the order we want.

    template = Match('is1', IncSubtensor, idx_list='i1', set_instead_of_inc='s/i')(
        Match('is2', IncSubtensor, idx_list='i2', set_instead_of_inc='s/i')(
            'x',
            Match('st2', Subtensor, idx_list='i2')('v')),
        Match('st1', Subtensor, idx_list='i1')('v'))

    assignment = template.match(node, {})
    assignments = []
    while assignment:
        assignments.append(assignment)
        assignment = template.match(assignment['is2'], {})

    if not assignments:
        return

    #incsubtensors = [a['is1'] for a in assignments] + [assignments[-1]['is2']]
    if any([len(a['i1']) > 1 for a in assignments]):
        # NotImplemented
        return
    if any([len(a['i2']) > 1 for a in assignments]):
        # NotImplemented
        return
    if any([not isinstance(a['i1'][0], slice) for a in assignments]):
        # NotImplemented
        return
    if any([not isinstance(a['i2'][0], slice) for a in assignments]):
        # NotImplemented
        return
    if any([a['i2'][0].step not in (1, None) for a in assignments]):
        # NotImplementedError
        # It should be possible to detect interleaved access patterns that are
        # guaranteed to provide full coverage (or even just enough coverage to
        # justify doing a more parallized summation)
        # e.g. a[::2] += b[::2] and a[1::2] += b[1::2]
        #      -> (a + b)[::2] and (a + b)[1::2]
        # Like all applications of this optimization, it is meant to really
        # pay off in conjunction with local_cut_whole_incsubtensor, which
        # arises in the graphs formed by SharedStorageWorkspace.
        return

    def start_stop_node(a):
        return (a.op.idx_list[0].start, a.op.idx_list[0].stop, a)

    incsubtensors = ([start_stop_node(a['is1']) for a in assignments]
            + [start_stop_node(assignments[-1]['is2'])])
    incsubtensors.sort()

    # -- ensure we have a contiguous range
    if not all(ssn0[1] == ssn1[0]
            for ssn0, ssn1 in zip(incsubtensors[:-1], incsubtensors[1:])):
        return

    start = incsubtensors[0][0]
    stop = incsubtensors[-1][1]
    x = assignments[-1]['x']
    v = assignments[-1]['v']
    set_instead_of_inc = assignments[-1]['s/i']
    rval = theano.tensor.inc_subtensor(
        x[start:stop],
        v[start:stop],
        set_instead_of_inc=set_instead_of_inc,
        )
    return [rval]


