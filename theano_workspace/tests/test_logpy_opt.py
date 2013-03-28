from theano_workspace.logpy_opt import raw_init
import theano
from theano import tensor

from logpy import (
    eq,
    run,
    var,
    )

def test_logpy():
    x = tensor.vector()
    y = tensor.vector()
    z = tensor.inc_subtensor(x[1:3], y)
    node = z.owner

    # otw theano chokes on var attributes when nose tries to print a traceback
    # XXX this should be un-monkey-patched after the test runs by e.g. a
    # context manager decorator
    theano.gof.Apply.__repr__ = object.__repr__
    theano.gof.Apply.__str__ = object.__str__

    w = dict((name, var(name)) for name in [
        'start', 'stop', 'step', 'set_instead_of_inc', 'inputs', 'outputs',
        'inplace', 'whole_op', 'dta',
        ])

    pattern = raw_init(theano.Apply,
        op=raw_init(theano.tensor.IncSubtensor,
            idx_list=[slice(w['start'], w['stop'], w['step'])],
            inplace=w['inplace'],
            set_instead_of_inc=w['set_instead_of_inc'],
            destroyhandler_tolerate_aliased=w['dta']),
        inputs=w['inputs'],
        outputs=w['outputs'])

    match, = run(0, w, (eq, node, pattern))

    assert match['stop'] == 3
    assert match['inputs'] == [x, y]

