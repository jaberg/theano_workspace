# XXX COMPLETELY UNTESTED, DEFINITELY NOT FUNCTIONAL

class Function(object):
    """
    Special case of Workspace for implementing a single callable expression

    TODO: Provides support for structuring outputs as nested list, dict, etc.
    """
    # XXX COMPLETELY UNTESTED
    def __init__(self, ws, inputs, outputs, dests, fn_name):
        self._ws = ws
        self._inputs = inputs
        self._outputs = outputs
        self._dests = dests
        self._fn_name = fn_name

    def __call__(self, *args):
        assert len(self._inputs) == len(args)
        for var, val in zip(self._inputs, args):
            self._ws[var] = val
        self._ws.compiled_updates[self._fn_name]()
        # TODO: unflatten dictionaries, singles, nested stuff, etc.
        return [self[var] for var in self._dests]


def function(inputs, outputs, ws_cls=Workspace):
    ws = ws_cls()
    dests = [o.type() for o in outputs]
    for var in inputs + dests:
        ws[var] = None
    ws.add_compiled_update('__call__', zip(dests, outputs))
    return Function(ws, inputs, outputs, dests, '__call__')

