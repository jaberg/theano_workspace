

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


