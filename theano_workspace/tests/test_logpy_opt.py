from theano_workspace.logpy_opt import raw_init, simplify
import theano
from theano import tensor

import logpy
from logpy import (
    eq,
    run,
    var, isvar
    )

from logpy.variables import variables

def test_context_manager():
    x = tensor.vector()
    y = tensor.vector()
    z = tensor.inc_subtensor(x[1:3], y)

    xp = tensor.vector()
    yp = tensor.vector()
    zp = tensor.inc_subtensor(xp[1:1234], yp)

    vars = (1234, xp, yp)

    with variables(*vars):
        match, = run(0, vars, (eq, z, zp))

    assert match == (3, x, y)


def theq(a, b):
    """ Theano equality - compare by string representation """
    sa = theano.printing.debugprint(a, file='str')
    sb = theano.printing.debugprint(b, file='str')

    if sa==sb:
        return True
    else:
        print sa
        print sb
        return False

def test_simplify():
    y = tensor.vector('y')
    assert theq(simplify(y+y)[0], 2*y)
    assert theq(simplify(y*y)[0], y**2)
    assert theq(simplify(tensor.exp(tensor.log(y**3)))[0], y**3)


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


def test_slice_sanity():
    class Foo(int, object):
        def __eq__(self, other):
            return False

    class Bar(int, logpy.variables.Var):
        # put these built-ins up front to hide int versions
        def __str__(self):
            return "~" + str(self.token)
        __repr__ = __str__

        def __eq__(self, other):
            return type(self) == type(other) and self.token == other.token

    def __hash__(self):
        return hash((type(self), self.token))

    assert not slice(1) == slice(Foo())
    bar = Bar(4)
    bar.token = 'hello'
    assert not slice(5) == slice(bar)
